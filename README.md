# Parlor

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine.

Parlor uses [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) for understanding speech and vision, and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech. You talk, show your camera, and it talks back, all locally.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

# Why?

I'm [self-hosting a totally free voice AI](https://www.fikrikarim.com/bule-ai-initial-release/) on my home server to help people learn speaking English. It has hundreds of monthly active users, and I've been thinking about how to keep it free while making it sustainable.

The obvious answer: run everything on-device, eliminating any server cost. Six months ago I needed an RTX 5090 to run just the voice models in real-time.

Google just released a super capable small model that I can run on my M3 Pro in real-time, with vision too! Sure you can't do agentic coding with this, but it is a game-changer for people learning a new language. Imagine a few years from now that people can run this locally on their phones. They can point their camera at objects and talk about them. And this model is multi-lingual, so people can always fallback to their native language if they want. This is essentially what OpenAI demoed a few years ago.

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
FastAPI server
    ├── Gemma 4 E2B via LiteRT-LM (GPU)  →  understands speech + vision
    └── Kokoro TTS (MLX on Mac, ONNX on Linux)  →  speaks back
    │
    │  WebSocket (streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Voice Activity Detection** in the browser ([Silero VAD](https://github.com/ricky0123/vad)). Hands-free, no push-to-talk.
- **Barge-in.** Interrupt the AI mid-sentence by speaking.
- **Sentence-level TTS streaming.** Audio starts playing before the full response is generated.

## Requirements

- Python 3.12+
- Node.js 20+ (for the Next.js frontend)
- macOS with Apple Silicon, or Linux with a supported GPU
- ~3 GB free RAM for the model

## Quick start

Parlor has two pieces: a **FastAPI** backend (`src/`) and a **Next.js** frontend (`web/`). In development you run both — Next proxies `/ws` to FastAPI so the browser talks to a single origin.

```bash
git clone https://github.com/typomonster/parlor-multilingual.git
cd parlor-multilingual
```

**Terminal 1 — backend:**

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

cd src
uv sync
uv run server.py
```

**Terminal 2 — frontend:**

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), grant camera and microphone access, and start talking.

Models are downloaded automatically on first run (~2.6 GB for Gemma 4 E2B, plus TTS models).

### How the dev proxy works

`web/next.config.ts` declares a rewrite from `/ws` → `http://localhost:8000/ws`, so the browser opens `ws://localhost:3000/ws` and Next.js proxies the upgrade through to FastAPI. Override the backend target with `BACKEND_URL=http://host:port npm run dev` if you want to point at a different server.

## Configuration

### Backend (`src/`)

| Variable           | Default                                    | Description                                                               |
| ------------------ | ------------------------------------------ | ------------------------------------------------------------------------- |
| `MODEL_PATH`       | auto-download from HuggingFace             | Path to a local `gemma-4-*.litertlm` file                                 |
| `HF_REPO`          | `litert-community/gemma-4-E2B-it-litert-lm`| HuggingFace repo to pull the model from                                   |
| `HF_FILENAME`      | `gemma-4-E2B-it.litertlm`                  | File within `HF_REPO` to download                                         |
| `PORT`             | `8000`                                     | FastAPI port                                                              |
| `TTS_ENGINE`       | `supertonic`                               | `supertonic` (multilingual — en/ko/es/pt/fr) or `kokoro` (English, faster) |
| `KOKORO_ONNX`      | unset                                      | Force the Kokoro ONNX backend on Apple Silicon                            |
| `SUPERTONIC_ONNX`  | unset                                      | Force the Supertonic ONNX backend on Apple Silicon                        |
| `SUPERTONIC_MLX_REPO`  | `typomonster/supertonic-2-mlx`         | HF repo for the Supertonic MLX checkpoint                                 |
| `SUPERTONIC_ONNX_REPO` | `Supertone/supertonic-2`               | HF repo for the Supertonic ONNX checkpoint                                |

#### TTS engines

Parlor ships with two interchangeable TTS backends, selected at runtime via `TTS_ENGINE`:

| Engine                | Languages                    | Apple Silicon               | Linux / x86             |
| --------------------- | ---------------------------- | --------------------------- | ----------------------- |
| `supertonic` (default)| `en`, `ko`, `es`, `pt`, `fr` | `mlx-audio` fork[^st] (MLX) | Supertonic ONNX Runtime |
| `kokoro`              | English only                 | `mlx-audio` (MLX)           | `kokoro-onnx`           |

Force ONNX instead of MLX on macOS with `SUPERTONIC_ONNX=1` or `KOKORO_ONNX=1`. Language is selected automatically from the frontend's active locale (Supertonic only — Kokoro ignores it). Set `TTS_ENGINE=kokoro` if you want the lighter English-only engine.

[^st]: `mlx-audio @ git+https://github.com/typomonster/mlx-audio` — the Supertonic-MLX support is pinned to this fork until it's upstreamed.

#### Supported models

| Model                                             | `HF_REPO`                                     | `HF_FILENAME`               |
| ------------------------------------------------- | --------------------------------------------- | --------------------------- |
| **Gemma 4 E2B** _(default — ~3 GB, fastest)_      | `litert-community/gemma-4-E2B-it-litert-lm`   | `gemma-4-E2B-it.litertlm`   |
| **Gemma 4 E4B** _(larger, higher quality)_        | `litert-community/gemma-4-E4B-it-litert-lm`   | `gemma-4-E4B-it.litertlm`   |

Switch models by setting both env vars, e.g.:

```bash
HF_REPO=litert-community/gemma-4-E4B-it-litert-lm \
HF_FILENAME=gemma-4-E4B-it.litertlm \
uv run server.py
```

### Frontend (`web/`)

| Variable      | Default                 | Description                             |
| ------------- | ----------------------- | --------------------------------------- |
| `BACKEND_URL` | `http://localhost:8000` | Target for the `/ws` proxy in dev/build |

## Performance (Apple M3 Pro)

| Stage                            | Time          |
| -------------------------------- | ------------- |
| Speech + vision understanding    | ~1.8-2.2s     |
| Response generation (~25 tokens) | ~0.3s         |
| Text-to-speech (1-3 sentences)   | ~0.3-0.7s     |
| **Total end-to-end**             | **~2.5-3.0s** |

Decode speed: ~83 tokens/sec on GPU (Apple M3 Pro).

## Project structure

```
parlor/
├── src/                       # Python backend
│   ├── server.py              # FastAPI WebSocket server + Gemma 4 inference
│   ├── tts.py                 # Platform-aware TTS (MLX on Mac, ONNX on Linux)
│   ├── pyproject.toml         # Python dependencies
│   └── benchmarks/
│       ├── bench.py           # End-to-end WebSocket benchmark
│       └── benchmark_tts.py   # TTS backend comparison
└── web/                       # Next.js frontend (TypeScript + Tailwind + shadcn/ui)
    ├── app/
    │   ├── page.tsx           # Main UI — VAD, camera, audio playback
    │   ├── layout.tsx
    │   └── globals.css        # App styles + Tailwind tokens
    ├── components/ui/         # shadcn/ui components
    └── next.config.ts         # /ws rewrite → FastAPI
```

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Supertonic](https://huggingface.co/Supertone/supertonic-2) multilingual TTS by Supertone
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection

## License

[Apache 2.0](LICENSE)
