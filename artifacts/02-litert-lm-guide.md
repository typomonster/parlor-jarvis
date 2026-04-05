# LiteRT-LM — Inference Engine for Gemma 4

## What is LiteRT-LM?

Google's **production-ready, open-source inference framework** for deploying LLMs on edge devices. Powers on-device GenAI in Chrome, Chromebook Plus, and Pixel Watch.

- **License**: Apache 2.0
- **GitHub**: https://github.com/google-ai-edge/LiteRT-LM
- **Latest release**: v0.10.1 (April 3, 2026)
- **Stars**: ~1,105
- **Core language**: C++ with bindings for Python, Kotlin, Swift (in dev)
- **Build system**: Bazel 7.6.1
- **Model format**: `.litertlm` bundles (pre-converted, hosted on HuggingFace at `litert-community/`)

## Why LiteRT-LM for Our Demo

| Feature | LiteRT-LM | Ollama | llama.cpp | MLX |
|---------|-----------|--------|-----------|-----|
| Gemma 4 E4B model size | **3.65 GB** | ~6 GB | ~6 GB | ~5 GB |
| Native audio input | **Yes** | No | No | No |
| Native vision input | **Yes** | Yes | Yes | Yes |
| Metal GPU on Mac | **Yes** (C++ API) | Yes | Yes | Yes |
| Python GPU | **Yes** (WebGPU/Metal, v0.10.1) | Yes | N/A | Yes |
| Production pedigree | Chrome, Pixel Watch | Dev tool | Library | Research |

**Key advantages**: Smallest model size (aggressive mixed-precision quant), native audio (no separate STT needed), Google-optimized for Google's own model.

## Supported Models

| Model | Type | Size on Disk |
|-------|------|-------------|
| **Gemma 4 E4B** | Chat | **3.65 GB** |
| **Gemma 4 E2B** | Chat | 2.58 GB |
| Gemma 3n E4B | Chat | 4.24 GB |
| Gemma 3n E2B | Chat | 2.97 GB |
| Phi-4-mini | Chat | 3.9 GB |
| Qwen 2.5 1.5B | Chat | 1.6 GB |

## Performance Benchmarks (Gemma 4 E2B)

1024 prefill tokens + 256 decode tokens:

| Platform | Backend | Prefill (tok/s) | Decode (tok/s) | TTFT | Memory |
|----------|---------|-----------------|----------------|------|--------|
| **MacBook Pro M4 Max** | GPU | **7,835** | **160.2** | **0.1s** | 1,623 MB |
| MacBook Pro M4 Max | CPU | 901 | 41.6 | 1.1s | 736 MB |
| Samsung S26 Ultra | GPU | 3,808 | 52.1 | 0.3s | 676 MB |
| iPhone 17 Pro | GPU | 2,878 | 56.5 | 0.3s | 1,450 MB |
| NVIDIA RTX 4090 | GPU | 11,234 | 143.4 | - | 913 MB |
| Raspberry Pi 5 | CPU | 133 | 7.6 | 7.8s | 1,546 MB |

No M3 Pro benchmarks published, but expect GPU decode in the 60-100+ tok/s range.

## API Language Support

| Language | Status | Best For |
|----------|--------|----------|
| **C++** | **Stable** | High-performance native (native Metal GPU) |
| Python | Stable | Prototyping & production (CPU + GPU via WebGPU/Metal). Native Metal crashes in Python due to ABI mismatch. |
| Kotlin | Stable | Android apps |
| Swift | **In Dev** | iOS/macOS (no public code yet) |

## macOS Prebuilt Libraries

Located at `prebuilt/macos_arm64/`:

| Library | Purpose |
|---------|---------|
| `libLiteRt.dylib` | Core runtime |
| `libLiteRtMetalAccelerator.dylib` | Metal GPU backend |
| `libLiteRtWebGpuAccelerator.dylib` | WebGPU backend |
| `libGemmaModelConstraintProvider.dylib` | Constrained decoding |

These are runtime-loaded plugins. The engine itself must be built from source.

## Build on macOS

```bash
# Prerequisites
xcode-select --install
brew install bazelisk

# Clone
git clone https://github.com/google-ai-edge/LiteRT-LM.git ~/workspace/LiteRT-LM
cd ~/workspace/LiteRT-LM
git checkout v0.10.1
git lfs pull

# Build with GPU support
bazel build //runtime/engine:litert_lm_main \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false

# Setup runtime directory
mkdir -p run_dir
cp bazel-bin/runtime/engine/litert_lm_main run_dir/
cp prebuilt/macos_arm64/*.dylib run_dir/

# Download model
pip install litert-lm  # or: uv tool install litert-lm
litert-lm run --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm

# Run
cd run_dir
./litert_lm_main --backend=gpu --model_path=../gemma-4-E4B-it.litertlm
```

## C++ API Reference

### Three-Layer Architecture

```
Engine (model loading, hardware backend)
  └── Session (stateful KV cache, prefill/decode) — low-level
  └── Conversation (chat API, prompt templates, multimodal) — high-level (recommended)
```

### Engine Creation

```cpp
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"

using namespace litert::lm;

auto model_assets = ModelAssets::Create("/path/to/gemma-4-E4B-it.litertlm");

auto engine_settings = EngineSettings::CreateDefault(
    *std::move(model_assets),
    Backend::GPU,                        // main LLM backend
    /*vision_backend=*/Backend::GPU,     // for image input
    /*audio_backend=*/Backend::CPU       // for audio input
);

auto engine = EngineFactory::CreateDefault(*std::move(engine_settings));
```

### Conversation API (Multimodal)

```cpp
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"

// Create with system prompt
auto config = ConversationConfig::Builder()
    .SetPreface(JsonPreface{
        .messages = nlohmann::ordered_json::array({
            {{"role", "system"}, {"content", "You are a helpful assistant."}}
        })
    })
    .Build(*engine);

auto conversation = Conversation::Create(*engine, *config);

// Text-only
auto response = conversation->SendMessage(
    json{{"role", "user"}, {"content", "Hello!"}});

// Image + text (webcam frame)
auto response = conversation->SendMessage(json{
    {"role", "user"},
    {"content", json::array({
        {{"type", "image"}, {"blob", base64_jpeg_data}},
        {{"type", "text"}, {"text", "What do you see?"}}
    })}
});

// Audio + text (microphone)
auto response = conversation->SendMessage(json{
    {"role", "user"},
    {"content", json::array({
        {{"type", "audio"}, {"blob", base64_wav_data}},
        {{"type", "text"}, {"text", "What did the user say?"}}
    })}
});

// Audio + image + text (both!)
auto response = conversation->SendMessage(json{
    {"role", "user"},
    {"content", json::array({
        {{"type", "audio"}, {"blob", base64_wav_data}},
        {{"type", "image"}, {"blob", base64_jpeg_data}},
        {{"type", "text"}, {"text", "The user is speaking to you while showing you their camera. Respond naturally."}}
    })}
});
```

### Streaming Token Generation

```cpp
conversation->SendMessageAsync(
    json{{"role", "user"}, {"content", "Write me a poem"}},
    [](absl::StatusOr<Message> message) {
        if (!message.ok()) return;
        auto& json_msg = std::get<JsonMessage>(*message);
        if (json_msg.is_null()) return;  // stream complete
        for (const auto& content : json_msg["content"]) {
            std::cout << content["text"].get<std::string>() << std::flush;
        }
    }
);
engine->WaitUntilDone(absl::Minutes(5));
```

### C API (for FFI / HTTP Server)

File: `/c/engine.h` — pure C with `extern "C"` linkage.

```c
// Streaming callback type
typedef void (*LiteRtLmStreamCallback)(
    void* callback_data, const char* chunk, bool is_final, const char* error_msg);

// Key functions
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings* settings);
LiteRtLmConversation* litert_lm_conversation_create(LiteRtLmEngine* engine, ...);

// Streaming multimodal message
int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation,
    const char* message_json,         // JSON string with multimodal content
    const char* extra_context,
    LiteRtLmStreamCallback callback,
    void* callback_data);

// Cancel mid-generation
void litert_lm_conversation_cancel_process(LiteRtLmConversation* conversation);
```

## Key Header Files

| Header | Purpose |
|--------|---------|
| `runtime/engine/engine.h` | Engine and Session interfaces |
| `runtime/engine/engine_factory.h` | EngineFactory |
| `runtime/engine/engine_settings.h` | EngineSettings, SessionConfig |
| `runtime/engine/io_types.h` | InputText, InputImage, InputAudio, Responses |
| `runtime/conversation/conversation.h` | Conversation (high-level chat) |
| `runtime/conversation/io_types.h` | JsonMessage, Message, Preface |
| `runtime/executor/executor_settings_base.h` | Backend enum, ModelAssets |
| `c/engine.h` | Pure C API for FFI |

## Model-Specific Processing

LiteRT-LM includes a dedicated `Gemma4DataProcessor` that automatically handles:
- Gemma 4 Jinja prompt templates
- Image preprocessing (resize, tokenize per configured budget)
- Audio preprocessing (mel-spectrogram extraction)
- Tool calling / function calling
- Constrained decoding

This is selected automatically based on model metadata — no manual configuration needed.

## Additional Features

- **Speculative decoding**: `enable_speculative_decoding` parameter (v0.10.1)
- **Constrained decoding**: regex, JSON Schema, Lark grammars
- **Function calling**: built-in with auto-execution
- **Jinja prompt templates**: auto-loaded from model metadata
- **Conversation history**: automatic incremental prompt rendering
- **Text scoring**: log-likelihood scores via `run_text_scoring()`

## Sources
- https://github.com/google-ai-edge/LiteRT-LM
- https://ai.google.dev/edge/litert-lm/overview
- HuggingFace: litert-community/gemma-4-E4B-it-litert-lm
