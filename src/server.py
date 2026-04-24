"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import litert_lm
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts

HF_REPO = os.environ.get(
    "HF_REPO", "typomonster/supergemma4-e4b-abliterated-litert-lm"
)
HF_FILENAME = os.environ.get(
    "HF_FILENAME", "supergemma4-e4b-abliterated.litertlm"
)


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()
SYSTEM_PROMPT = (
    "You are Jarvis, a helpful on-device voice assistant. The user speaks "
    "to you through a microphone and may also share their camera, screen, "
    "a PDF page, or a video frame. Listen carefully and reference specific "
    "details from what you see and hear — concrete nouns, numbers, names, "
    "what's on screen, what's in the camera frame. "
    "Respond in a natural conversational tone that works as spoken audio. "
    "Aim for 2-5 sentences per reply: long enough to be substantive, short "
    "enough not to drone. Vary your phrasing across turns and never repeat a "
    "previous answer verbatim. Avoid generic greetings or fillers unless the "
    "user actually greeted you, and answer the user's question directly — "
    "don't lecture or moralise. "
    "You MUST always use the respond_to_user tool. Put an exact transcription "
    "of the user's audio in `transcription` and your reply in `response`."
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

engine = None
conversation = None
tts_backend = None
# Shared across the global conversation; cleared before each send.
tool_result: dict = {}
# Last assistant reply — used to nudge the model away from byte-identical
# repeats when the same or similar user prompt arrives twice in a row.
last_reply: str = ""
# Set by a client-side `{type: "reset"}` message. On the next user turn
# we prepend a banner telling Gemma to disregard prior context and
# clear repeat-breaker state. litert-lm won't let us tear down the
# session for a hard reset, so this is a soft reset via prompting.
pending_reset: bool = False

# Pin each framework to one thread: serialises non-thread-safe
# litert-lm calls and keeps Gemma/Supertonic Metal contexts apart.
_llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parlor-llm")
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parlor-tts")


def _respond_to_user(transcription: str, response: str) -> str:
    """Respond to the user's voice message.

    Args:
        transcription: Exact transcription of what the user said in the audio.
        response: Your conversational reply (2-5 natural sentences).
            Reference what you heard and saw concretely. Don't start with
            generic greetings unless the user greeted you. Vary phrasing
            across turns — never repeat a previous reply verbatim. Answer
            directly without lecturing.
    """
    tool_result["transcription"] = transcription
    tool_result["response"] = response
    return "OK"


def _load_engine() -> None:
    global engine
    print(f"Loading model from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU,
        vision_backend=litert_lm.Backend.GPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")


def _load_tts() -> None:
    global tts_backend
    tts_backend = tts.load()


def _ensure_conversation(lang: str) -> None:
    """Lazy-create the engine-wide conversation on the first turn.

    `extra_context.language` is only honoured if the model's chat
    template references it — the per-turn text prefix is the real
    language steering.
    """
    global conversation
    if conversation is not None:
        return
    hint = (lang or "en").strip().lower() or "en"
    conversation = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=[_respond_to_user],
        extra_context={"language": hint},
    )
    conversation.__enter__()
    print(f"Conversation started (language hint = {hint!r}).")


def _llm_turn(content: list, lang: str) -> dict:
    """One queued LLM turn: ensure conversation, clear tool_result, send."""
    _ensure_conversation(lang)
    tool_result.clear()
    return conversation.send_message({"role": "user", "content": content})


@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_event_loop()
    # Load each framework on the thread that will own it.
    await loop.run_in_executor(_llm_executor, _load_engine)
    await loop.run_in_executor(_tts_executor, _load_tts)
    try:
        yield
    finally:
        _llm_executor.shutdown(wait=False)
        _tts_executor.shutdown(wait=False)


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
    global last_reply, pending_reset
    await ws.accept()

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
                elif msg.get("type") == "reset":
                    global pending_reset, last_reply
                    pending_reset = True
                    last_reply = ""
                    print("Client requested conversation reset.")
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

            # Prepended to this turn since we can't swap the real
            # system message after conversation creation.
            system_override = (msg.get("system_prompt") or "").strip()

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
                content.append({"type": "text", "text": f"The user just spoke to you (audio). They are also sharing {_join(sources)}, but only reference the visual content if the user's question is actually about what they're showing — otherwise answer their question directly without commenting on the image."})
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

            # Soft reset requested by the client — tell the model to
            # disregard all prior context on this turn. litert-lm won't
            # let us physically drop the session, so we rely on the
            # model respecting the banner. Flag is one-shot.
            if pending_reset:
                content[-1]["text"] = (
                    "[CONVERSATION RESET. The user has ended the previous"
                    " conversation and started a new one. Disregard all"
                    " prior turns — they are no longer relevant. Respond"
                    " only to the message below.]\n\n"
                    + content[-1]["text"]
                )
                pending_reset = False

            # Break deterministic repeat loops: if the last assistant
            # reply is non-empty, tell the model to not reproduce it.
            if last_reply:
                content[-1]["text"] += (
                    f" Your previous reply was: \"{last_reply}\" — do not"
                    " repeat that wording. If the user is asking the same"
                    " question again, rephrase and give a genuinely different"
                    " answer."
                )

            if system_override:
                content[-1]["text"] = (
                    f"[Instructions for this conversation: {system_override}]\n\n"
                    + content[-1]["text"]
                )

            # LLM inference (serialised on the LLM worker).
            t0 = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                _llm_executor, _llm_turn, content, lang
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

            # Remember this reply so the next turn's prompt can nudge
            # the model away from reproducing it verbatim.
            last_reply = text_response

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

                # TTS on the pinned worker (avoids Metal overlap with LLM).
                pcm = await asyncio.get_event_loop().run_in_executor(
                    _tts_executor,
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
