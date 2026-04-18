"""Platform-aware TTS.

Two engines are supported, selected by the ``TTS_ENGINE`` env var:

- ``supertonic`` (default) — multilingual (en / ko / es / pt / fr). Uses
  the Supertonic models through ``mlx-audio`` on Apple Silicon and the
  Supertonic ONNX runtime elsewhere. Force ONNX with ``SUPERTONIC_ONNX=1``.
- ``kokoro`` — English-only, fast. Uses ``mlx-audio`` on Apple Silicon
  and ``kokoro-onnx`` elsewhere. Force ONNX with ``KOKORO_ONNX=1``.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Optional

import numpy as np


SUPPORTED_LANGS = ("en", "ko", "es", "pt", "fr")


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface.

    ``generate`` returns a 1-D float32 numpy array of PCM samples at
    ``self.sample_rate``. ``multilingual`` indicates whether a ``lang``
    hint has any effect — Kokoro backends ignore it.
    """

    sample_rate: int = 24000
    multilingual: bool = False

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.05,
        lang: str = "en",
    ) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────
# Kokoro
# ─────────────────────────────────────────────────────────────────────

KOKORO_DEFAULT_VOICE = "af_heart"


class KokoroMLXBackend(TTSBackend):
    """Kokoro via mlx-audio on Apple Silicon (GPU)."""

    def __init__(self) -> None:
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: trigger phonemizer / spacy init so the first real call
        # doesn't eat the pipeline cost.
        list(self._model.generate(text="Hello", voice=KOKORO_DEFAULT_VOICE, speed=1.0))

    def generate(self, text, voice=None, speed=1.1, lang="en"):
        results = list(
            self._model.generate(
                text=text,
                voice=voice or KOKORO_DEFAULT_VOICE,
                speed=speed,
            )
        )
        return np.concatenate([np.array(r.audio) for r in results])


class KokoroONNXBackend(TTSBackend):
    """Kokoro via kokoro-onnx on CPU. Used on Linux and as a Mac fallback."""

    def __init__(self) -> None:
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text, voice=None, speed=1.1, lang="en"):
        pcm, _sr = self._model.create(
            text,
            voice=voice or KOKORO_DEFAULT_VOICE,
            speed=speed,
        )
        return pcm


# ─────────────────────────────────────────────────────────────────────
# Supertonic (multilingual)
# ─────────────────────────────────────────────────────────────────────

SUPERTONIC_DEFAULT_VOICE = "M1"


class SupertonicMLXBackend(TTSBackend):
    """Supertonic via the mlx-audio fork (https://github.com/typomonster/mlx-audio).

    Requires the Supertonic checkpoint in MLX format, defaults to
    ``typomonster/supertonic-2-mlx`` (override with ``SUPERTONIC_MLX_REPO``).
    """

    multilingual = True

    def __init__(self) -> None:
        from mlx_audio.tts import load

        repo = os.environ.get("SUPERTONIC_MLX_REPO", "typomonster/supertonic-2-mlx")
        self._model = load(repo)
        self.sample_rate = self._model.sample_rate
        voices = list(self._model.available_voices)
        self._default_voice = (
            SUPERTONIC_DEFAULT_VOICE
            if SUPERTONIC_DEFAULT_VOICE in voices
            else (voices[0] if voices else SUPERTONIC_DEFAULT_VOICE)
        )
        # Warmup pass so first real request is hot.
        list(
            self._model.generate(
                "Hello",
                voice=self._default_voice,
                lang="en",
                speed=1.05,
                steps=5,
            )
        )

    def generate(self, text, voice=None, speed=1.05, lang="en"):
        if lang not in SUPPORTED_LANGS:
            lang = "en"
        v = voice or self._default_voice
        if v not in self._model.available_voices:
            v = self._default_voice
        pieces = [
            np.asarray(r.audio)
            for r in self._model.generate(
                text,
                voice=v,
                lang=lang,
                speed=speed,
                steps=5,
            )
        ]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)


class SupertonicONNXBackend(TTSBackend):
    """Supertonic via ONNX Runtime (CPU). Linux default / Mac fallback.

    Downloads the Supertonic ONNX checkpoint from HuggingFace
    (``Supertone/supertonic-2`` by default — override with
    ``SUPERTONIC_ONNX_REPO``) and drives it through the shared
    ``supertonic_helper`` module ported from the upstream reference.
    """

    multilingual = True

    def __init__(self) -> None:
        from huggingface_hub import snapshot_download

        # Local import so the module isn't eagerly required on Mac installs.
        import supertonic_helper as sth

        repo = os.environ.get("SUPERTONIC_ONNX_REPO", "Supertone/supertonic-2")
        ckpt_dir = snapshot_download(repo)

        self._ckpt_dir = Path(ckpt_dir)
        self._tts = sth.load_text_to_speech(str(self._ckpt_dir), use_gpu=False)
        self._load_style = sth.load_voice_style
        self.sample_rate = self._tts.sample_rate
        self._voice_cache: dict[str, object] = {}
        self._default_voice = SUPERTONIC_DEFAULT_VOICE

    def _resolve_voice_path(self, voice: str) -> str:
        voices_dir = self._ckpt_dir / "voice_styles"
        p = voices_dir / f"{voice}.json"
        if p.exists():
            return str(p)
        # Fall back to whatever the checkpoint ships with.
        fallback = next(voices_dir.glob("*.json"), None)
        if fallback is None:
            raise FileNotFoundError(f"No voice styles found under {voices_dir}")
        return str(fallback)

    def _style_for(self, voice: str):
        if voice in self._voice_cache:
            return self._voice_cache[voice]
        style = self._load_style([self._resolve_voice_path(voice)])
        self._voice_cache[voice] = style
        return style

    def generate(self, text, voice=None, speed=1.05, lang="en"):
        if lang not in SUPPORTED_LANGS:
            lang = "en"
        v = voice or self._default_voice
        style = self._style_for(v)
        wav, dur = self._tts(text, lang, style, total_step=5, speed=speed)
        # Helper returns (1, T) and per-sample duration — trim to real length.
        n = int(self.sample_rate * float(dur[0]))
        return wav[0, :n].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────


def _load_kokoro() -> TTSBackend:
    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = KokoroMLXBackend()
            print(f"TTS: kokoro (MLX, sample_rate={backend.sample_rate})")
            return backend
        except ImportError:
            print("TTS: kokoro MLX unavailable, falling back to ONNX")
    backend = KokoroONNXBackend()
    print(f"TTS: kokoro (ONNX CPU, sample_rate={backend.sample_rate})")
    return backend


def _load_supertonic() -> TTSBackend:
    if _is_apple_silicon() and not os.environ.get("SUPERTONIC_ONNX"):
        try:
            backend = SupertonicMLXBackend()
            print(
                f"TTS: supertonic (MLX, sample_rate={backend.sample_rate}, multilingual)"
            )
            return backend
        except ImportError:
            print("TTS: supertonic MLX unavailable, falling back to ONNX")
    backend = SupertonicONNXBackend()
    print(
        f"TTS: supertonic (ONNX CPU, sample_rate={backend.sample_rate}, multilingual)"
    )
    return backend


def load() -> TTSBackend:
    """Load the TTS backend selected by env (``TTS_ENGINE``, default supertonic)."""
    engine = os.environ.get("TTS_ENGINE", "supertonic").strip().lower()
    if engine == "kokoro":
        return _load_kokoro()
    return _load_supertonic()
