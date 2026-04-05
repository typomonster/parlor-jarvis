# Benchmarks — Apple M3 Pro 18GB

All benchmarks run on LiteRT-LM v0.10.1, Gemma 4 E2B (2.3B effective, 2.58 GB model).

## Python API — GPU (WebGPU) Setup

As of litert-lm v0.10.1, `Backend.GPU` works in Python despite docs saying "upcoming".
The pip package ships a WebGPU accelerator that uses Metal under the hood on macOS.
Audio backend is model-constrained to CPU only (`INVALID_ARGUMENT` if set to GPU).

```python
engine = litert_lm.Engine(
    MODEL_PATH,
    backend=litert_lm.Backend.GPU,
    vision_backend=litert_lm.Backend.GPU,
    audio_backend=litert_lm.Backend.CPU,  # model requires CPU
)
```

**How to verify GPU is active:** check stderr for `RegisterAccelerator: name=GPU WebGPU`.

**Known warning (non-fatal):** `Could not load symbol LiteRtTopKWebGpuSampler_UpdateConfig` —
falls back to CPU sampling, GPU compute still active. Does not affect performance.

## Synthetic Benchmarks (1024 prefill, 256 decode)

Same methodology as official HuggingFace benchmarks. Text-only, no multimodal encoders.
Run via `litert_lm.Benchmark()` Python API.

### GPU (WebGPU/Metal) vs CPU

| Metric | GPU (cached) | GPU (first run) | CPU | GPU vs CPU |
|--------|-------------|-----------------|-----|------------|
| **Prefill** | **2,662 tok/s** | 431 tok/s | 215 tok/s | **12.4x** |
| **Decode** | **83 tok/s** | 70 tok/s | 19 tok/s | **4.4x** |
| **TTFT** | **0.40s** | 2.39s | 4.82s | **12.1x** |
| Init time | ~0.8s | ~2.1s | ~0.8s | - |

**First-run shader penalty:** WebGPU compiles Metal shaders on first inference after process
start (~2.4s TTFT). Subsequent runs use cached shaders (~0.4s TTFT). The server does a warmup
`send_message("Hi")` at startup to pre-compile shaders.

### Comparison with official HuggingFace benchmarks

| Platform | Prefill (tok/s) | Decode (tok/s) | TTFT |
|----------|-----------------|----------------|------|
| **M3 Pro (ours, WebGPU)** | **2,662** | **83** | **0.40s** |
| M4 Max (official) | 7,835 | 160 | 0.1s |
| iPhone 17 Pro (official) | 2,878 | 56.5 | 0.3s |
| S26 Ultra (official) | 3,808 | 52.1 | 0.3s |
| RTX 4090 (official) | 11,234 | 143 | 0.1s |

Our M3 Pro is in line with expectations — comparable to iPhone 17 Pro for prefill,
faster decode (83 vs 56.5) due to higher memory bandwidth.

## Native Metal vs WebGPU (Python)

The pip package only ships `libLiteRtWebGpuAccelerator.dylib`. The C++ build includes
native Metal (`libLiteRtMetalAccelerator.dylib`) which shows ~6x faster prefill in the
C++ binary. However:

- **Native Metal crashes (SIGSEGV) in Python** — the prebuilt Metal lib has ABI
  incompatibilities with the pip-installed Python nanobind bindings.
- **DYLD_LIBRARY_PATH doesn't work** — macOS dyld reads it at process start, and
  litert_lm uses dlopen relative to its package directory, not the env var.
- **WebGPU with cached shaders matches Metal performance** — once shaders are compiled,
  WebGPU prefill (2,662 tok/s) is comparable to native Metal. The gap only shows on first run.

**Conclusion:** stick with WebGPU from pip. No benefit to native Metal for Python users.

## Multimodal Encoder Costs (Real-World)

These are measured by isolating each modality and subtracting text-only baseline.
The encoders run their own forward passes BEFORE token prefill begins.

| Component | Time | Notes |
|-----------|------|-------|
| **Image encoder (SigLIP, GPU)** | **~0.86s** | Biggest single cost. 320px input upscaled to ~912x672, produces ~268 tokens (default budget 280). Resolution doesn't matter — always fills budget. |
| **Audio encoder (Conformer, CPU)** | **~0.1-0.4s** | 305M param conformer. ~25 tokens/sec of audio, capped at 750 tokens (30s). Forced to CPU by model constraint. |
| **Tool grammar overhead** | **~0.45s** | Constrained decoding setup for tool call parsing. Paid on EVERY `send_message`, not just the first. Internal to LiteRT-LM. |
| **Text prefill (GPU)** | **~0.08s** | For ~300 tokens (system + prompt). Fast. |

### Gemma 4 Input Token Counts

Tokenizer: SentencePiece, 262,144 vocab. Extracted from `.litertlm` bundle at offset 32768.

**Image tokens:** Configurable budget (70/140/280/560/1120). Default is **280**.
For a 320x240 JPEG at default budget: upscaled to 912x672 → **266 soft tokens + 2 special = 268 total**.
Input resolution doesn't matter — image is always rescaled to fill the budget.

**Audio tokens:** ~**25 tokens per second** of audio (40ms per token), capped at 750 tokens (30s).
- 1s → ~25 tokens
- 2s → ~50 tokens
- 5s → ~125 tokens

**Text:** System prompt ~51 tokens, tool definition ~177 tokens, user prompt ~30 tokens.

## Real-World Pipeline Latency (GPU, E2B)

Measured via `bench.py` against running server over WebSocket. Image + Audio every turn.

### Per-Turn Breakdown

| Stage | Time |
|-------|------|
| Image encoder (SigLIP) | ~0.86s |
| Tool constrained decoding overhead | ~0.45s |
| Audio encoder (2-5s clip, CPU) | ~0.1-0.4s |
| Text + history prefill | ~0.1-0.4s |
| **Total TTFT / prefill** | **~1.8-2.2s** |
| Decode (~25 tokens at 80 tok/s) | ~0.3s |
| **Total LLM** | **~2.1-2.5s** |
| TTS (Kokoro MLX, 1-3 sentences) | ~0.3-0.7s |
| **Total end-to-end** | **~2.5-3.0s** |

### Multi-Turn Context Growth

From bench.py (same WebSocket connection, image + audio every turn):

| Turn | LLM Time | Notes |
|------|----------|-------|
| Turn 1 | ~1.8s | Fresh conversation |
| Turn 2 | ~2.2s | +prior turn context |
| Turn 3 | ~2.5s | Context growing |
| Turn 4 | ~2.2s | |
| Turn 5 | ~2.2s | |

TTFT grows ~0.1-0.2s per turn as conversation history accumulates.

## Bottleneck Summary

| Bottleneck | Cost per turn | Fixable? |
|------------|---------------|----------|
| **Image encoder (SigLIP)** | 0.86s | Maybe — reduce image budget from 280→70 (~0.2s), but not configurable in Python API |
| **Tool grammar** | 0.45s | No — internal to LiteRT-LM constrained decoding |
| **Audio encoder (CPU)** | 0.1-0.4s | No — model constraint forces CPU |
| **Context growth** | 0.1-0.2s/turn | Could limit conversation history length |
| **Decode** | ~0.3s | Already near hardware limit (80 tok/s) |
| **TTS** | 0.3-0.7s | Already on GPU (MLX) |

## Benchmarking

```bash
# Synthetic text-only benchmark (Python API)
python3 -c "
import litert_lm, os
bench = litert_lm.Benchmark(
    os.path.expanduser('~/workspace/LiteRT-LM/run_dir/gemma-4-E2B-it.litertlm'),
    backend=litert_lm.Backend.GPU, prefill_tokens=1024, decode_tokens=256)
r = bench.run()
print(f'Prefill: {r.last_prefill_tokens_per_second:.0f} tok/s')
print(f'Decode: {r.last_decode_tokens_per_second:.0f} tok/s')
print(f'TTFT: {r.time_to_first_token_in_second:.3f}s')
"

# End-to-end benchmark against running server
# Start server: uv run python server.py
# Then: uv run python bench.py
```
