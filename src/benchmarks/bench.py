"""End-to-end benchmark — connects to the running server over WebSocket.

Start the server first:  uv run python server.py
Then run:                uv run python benchmarks/bench.py

Measures real latencies including network, serialization, and all server overhead.
"""

import asyncio
import base64
import json
import os
import time
import wave

import numpy as np
import websockets


SERVER_URL = os.environ.get("SERVER_URL", "ws://localhost:8000/ws")


# ── Test fixtures ──────────────────────────────────────────────────────────


def make_wav_b64(duration_s: float, sample_rate: int = 16000) -> str:
    """Create a WAV file as base64 string."""
    samples = np.sin(2 * np.pi * 440 * np.arange(int(sample_rate * duration_s)) / sample_rate)
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def make_jpg_b64(width: int = 320, height: int = 240) -> str:
    """Create a JPEG image as base64 string."""
    import io
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ── WebSocket client ───────────────────────────────────────────────────────


async def send_and_receive(ws, payload: dict) -> dict:
    """Send a message and collect all responses until audio_end."""
    t0 = time.time()
    await ws.send(json.dumps(payload))

    text_msg = None
    audio_bytes = 0
    tts_time = None

    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)

        if msg["type"] == "text":
            text_msg = msg
            text_msg["_recv_time"] = time.time() - t0
        elif msg["type"] == "audio_chunk":
            audio_bytes += len(msg.get("audio", ""))
        elif msg["type"] == "audio_start":
            pass
        elif msg["type"] == "audio_end":
            tts_time = msg.get("tts_time")
            break

    total_time = time.time() - t0
    return {
        "text": text_msg.get("text", "") if text_msg else "",
        "transcription": text_msg.get("transcription") if text_msg else None,
        "llm_time": text_msg.get("llm_time", 0) if text_msg else 0,
        "ttft": text_msg.get("ttft"),
        "decode_tokens": text_msg.get("decode_tokens"),
        "decode_time": text_msg.get("decode_time"),
        "tok_per_sec": text_msg.get("tok_per_sec"),
        "tts_time": tts_time,
        "total_time": round(total_time, 2),
        "text_recv_time": round(text_msg["_recv_time"], 2) if text_msg else 0,
        "audio_kb": round(audio_bytes * 3 / 4 / 1024, 1),  # base64 -> bytes -> KB
    }


# ── Print helpers ──────────────────────────────────────────────────────────


def print_header():
    print(f"{'Test':<22} {'LLM':>6} {'TTS':>6} {'Total':>6}  {'Response'}")
    print("-" * 80)


def print_row(name: str, r: dict):
    resp = r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
    extra = ""
    if r.get("ttft") is not None:
        extra = f" (prefill {r['ttft']}s · {r['decode_tokens']}tok · {r['tok_per_sec']}tok/s)"
    print(
        f"{name:<22} {r['llm_time']:>5.2f}s {r['tts_time'] or 0:>5.2f}s "
        f"{r['total_time']:>5.2f}s  {resp}{extra}"
    )


# ── Test suites ────────────────────────────────────────────────────────────


async def main():
    # Prepare test data
    audio_2s = make_wav_b64(2.0)
    audio_5s = make_wav_b64(5.0)
    image = make_jpg_b64()

    # ── Individual turns (new connection each = fresh conversation) ─────

    print("=" * 80)
    print("BENCHMARK: Individual turns (fresh connection each)")
    print("=" * 80)
    print_header()

    tests = [
        ("Text only", {"text": "Tell me a fun fact about the ocean."}),
        ("Audio 2s", {"audio": audio_2s}),
        ("Audio 5s", {"audio": audio_5s}),
        ("Image only", {"image": image}),
        ("Image + Audio 2s", {"audio": audio_2s, "image": image}),
        ("Image + Audio 5s", {"audio": audio_5s, "image": image}),
    ]

    for name, payload in tests:
        async with websockets.connect(SERVER_URL) as ws:
            r = await send_and_receive(ws, payload)
            print_row(name, r)

    # ── Multi-turn conversation (same connection) ──────────────────────

    print()
    print("=" * 80)
    print("BENCHMARK: Multi-turn conversation (same connection)")
    print("=" * 80)
    print_header()

    turns = [
        ("Turn 1: img+aud 1s", {"audio": make_wav_b64(1.0), "image": image}),
        ("Turn 2: img+aud 2s", {"audio": audio_2s, "image": image}),
        ("Turn 3: img+aud 3s", {"audio": make_wav_b64(3.0), "image": image}),
        ("Turn 4: img+aud 5s", {"audio": audio_5s, "image": image}),
        ("Turn 5: img+aud 2s", {"audio": audio_2s, "image": image}),
    ]

    async with websockets.connect(SERVER_URL) as ws:
        for name, payload in turns:
            r = await send_and_receive(ws, payload)
            print_row(name, r)

    # ── Correctness checks ─────────────────────────────────────────────

    print()
    print("=" * 80)
    print("CORRECTNESS")
    print("=" * 80)

    async with websockets.connect(SERVER_URL) as ws:
        # Tool calling works
        r = await send_and_receive(ws, {"text": "Hello, nice to meet you!"})
        print(f"  Tool called:        {'PASS' if r['transcription'] else 'FAIL'}")
        print(f"  Has response:       {'PASS' if r['text'] and len(r['text']) > 0 else 'FAIL'}")
        print(f"  No raw delimiters:  {'PASS' if '<|\"|>' not in r['text'] else 'FAIL'}")

        # Image description works
        r = await send_and_receive(ws, {"image": image})
        has_desc = r["text"] and len(r["text"]) > 10
        print(f"  Image described:    {'PASS' if has_desc else 'FAIL'}")

        # Audio works
        r = await send_and_receive(ws, {"audio": audio_2s})
        print(f"  Audio processed:    {'PASS' if r['text'] and len(r['text']) > 0 else 'FAIL'}")

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
