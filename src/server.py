"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import litert_lm
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts

HF_REPO = os.environ.get("HF_REPO", "litert-community/gemma-4-E2B-it-litert-lm")
HF_FILENAME = os.environ.get("HF_FILENAME", "gemma-4-E2B-it.litertlm")


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "You MUST always use the respond_to_user tool to reply. "
    "First transcribe exactly what the user said, then write your response."
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

engine = None
tts_backend = None


def load_models():
    global engine, tts_backend
    print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU,
        vision_backend=litert_lm.Backend.GPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")

    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Per-connection tool state captured via closure
    tool_result = {}

    def respond_to_user(transcription: str, response: str) -> str:
        """Respond to the user's voice message.

        Args:
            transcription: Exact transcription of what the user said in the audio.
            response: Your conversational response to the user. Keep it to 1-4 short sentences.
        """
        tool_result["transcription"] = transcription
        tool_result["response"] = response
        return "OK"

    # Conversation is created lazily on the first inbound message so we
    # can use the client-supplied system prompt. litert-lm only supports
    # one session per engine and can't be torn down+recreated mid-
    # connection — if the client edits the prompt, it closes+reopens the
    # WebSocket to get a fresh conversation.
    conversation = None
    current_system_prompt = SYSTEM_PROMPT

    def ensure_conversation(system_prompt: str) -> bool:
        """Create the conversation if it doesn't exist yet. Returns True
        on first creation. If a session already exists and the requested
        prompt differs, we log a note but keep the existing conversation
        (the frontend is expected to reconnect on prompt change)."""
        nonlocal conversation, current_system_prompt
        resolved = (system_prompt or "").strip() or SYSTEM_PROMPT
        if conversation is None:
            current_system_prompt = resolved
            conversation = engine.create_conversation(
                messages=[{"role": "system", "content": current_system_prompt}],
                tools=[respond_to_user],
            )
            conversation.__enter__()
            print(f"Conversation started ({len(current_system_prompt)} chars).")
            return True
        if resolved != current_system_prompt:
            print(
                "System prompt changed mid-session — ignored. "
                "Reconnect to apply the new prompt."
            )
        return False

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()

            # Optional language hint — Supertonic uses it; Kokoro ignores it.
            lang = (msg.get("lang") or "en").lower()
            # Optional voice preset. Empty / unknown → backend's default.
            voice = (msg.get("voice") or "").strip() or None

            # Lazy-create (or log mismatch for) the conversation based on
            # the client-supplied system prompt.
            ensure_conversation(msg.get("system_prompt") or "")

            # Accept images as a list of {source, blob} items. Fall back to
            # the legacy single `image` field so the static index.html UI
            # (which predates this change) still works.
            images = list(msg.get("images") or [])
            if not images and msg.get("image"):
                images = [{"source": "camera", "blob": msg["image"]}]
            images = [i for i in images if i.get("blob")]

            content = []
            if msg.get("audio"):
                content.append({"type": "audio", "blob": msg["audio"]})
            for item in images:
                content.append({"type": "image", "blob": item["blob"]})

            source_labels = {
                "camera": "camera",
                "screen": "screen",
                "pdf": "a PDF page",
                "video": "a video frame",
            }
            sources = [
                source_labels.get(i.get("source", ""), i.get("source") or "image")
                for i in images
            ]

            def _join(items):
                if len(items) <= 1:
                    return items[0] if items else ""
                return ", ".join(items[:-1]) + " and " + items[-1]

            has_audio = bool(msg.get("audio"))
            if has_audio and sources:
                content.append({"type": "text", "text": f"The user just spoke to you (audio) while also showing you {_join(sources)}. Respond to what they said, referencing what you see if relevant."})
            elif has_audio:
                content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
            elif sources:
                content.append({"type": "text", "text": f"The user is showing you {_join(sources)}. Describe what you see."})
            else:
                content.append({"type": "text", "text": msg.get("text", "Hello!")})

            # Steer Gemma to respond in the user's selected language so the
            # downstream TTS (which is already told the language via `lang`)
            # is actually speaking matching words.
            lang_names = {
                "en": "English",
                "ko": "Korean",
                "es": "Spanish",
                "pt": "Portuguese",
                "fr": "French",
            }
            if lang in lang_names and lang != "en":
                content[-1]["text"] += f" You MUST respond in {lang_names[lang]}."

            # LLM inference
            t0 = time.time()
            tool_result.clear()
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: conversation.send_message({"role": "user", "content": content})
            )
            llm_time = time.time() - t0

            # Extract response from tool call or fallback to raw text
            if tool_result:
                strip = lambda s: s.replace('<|"|>', "").strip()
                transcription = strip(tool_result.get("transcription", ""))
                text_response = strip(tool_result.get("response", ""))
                print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response}")
            else:
                transcription = None
                text_response = response["content"][0]["text"]
                print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
            if transcription:
                reply["transcription"] = transcription
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                print("Interrupted before TTS, skipping audio")
                continue

            # Streaming TTS: split into sentences and send chunks progressively
            sentences = split_sentences(text_response)
            if not sentences:
                sentences = [text_response]

            tts_start = time.time()

            # Signal start of audio stream
            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                    break

                # Generate audio for this sentence
                pcm = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda s=sentence: tts_backend.generate(
                        s, voice=voice, lang=lang
                    ),
                )

                if interrupted.is_set():
                    break

                # Convert to 16-bit PCM and send as base64
                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                    "index": i,
                }))

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": round(tts_time, 2),
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        if conversation is not None:
            try:
                conversation.__exit__(None, None, None)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
