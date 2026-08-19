"""Configuration loaded from environment variables.

All settings can be overridden via a .env file (loaded by python-dotenv) or real
environment variables. Sensible defaults are provided so the stack runs out of
the box on a GPU host.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Hugging Face ----
# A read token is only required for gated models (e.g. FLUX.1-dev).
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
# Local cache for model weights (survives container restarts / redeploys).
HF_HOME: str = os.environ.get("HF_HOME", "/models")

# ---- Model repositories (overridable) ----
# Primary model: FLUX.1-schnell (Apache-2.0, few-step, fits 12GB GPU).
MODEL_A_REPO: str = os.environ.get(
    "MODEL_A_REPO", "black-forest-labs/FLUX.1-schnell"
)
# Secondary model: SDXL base (Apache-2.0, general purpose, fits 12GB GPU).
MODEL_B_REPO: str = os.environ.get(
    "MODEL_B_REPO", "stabilityai/stable-diffusion-xl-base-1.0"
)
# TTS model: Kokoro-82M (82M params, leve e rapido, voz natural).
MODEL_C_REPO: str = os.environ.get("MODEL_C_REPO", "hexgrad/Kokoro-82M")
# TTS model: Qwen3-TTS-1.7B-CustomVoice (multilíngue, vozes nomeadas).
MODEL_D_REPO: str = os.environ.get(
    "MODEL_D_REPO", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
# ASR model: Whisper-large-v3-turbo (word-level timestamps, ~4x mais rapido).
MODEL_E_REPO: str = os.environ.get(
    "MODEL_E_REPO", "openai/whisper-large-v3-turbo"
)

# Tipo de mídia de saída por modelo (seguindo o padrão de 1 executor serializado).
# image -> PNG, tts -> WAV, asr -> JSON (transcrição).
MEDIA_TYPE: dict[str, str] = {
    "flux": "image/png",
    "model2": "image/png",
    "kokoro": "audio/wav",
    "qwen3tts": "audio/wav",
    "whisper_turbo": "application/json",
}
# Extensão de arquivo de saída por modelo.
MEDIA_EXT: dict[str, str] = {
    "flux": "png",
    "model2": "png",
    "kokoro": "wav",
    "qwen3tts": "wav",
    "whisper_turbo": "json",
}
# Task (tipo de geração) por modelo.
MODEL_TASK: dict[str, str] = {
    "flux": "image",
    "model2": "image",
    "kokoro": "tts",
    "qwen3tts": "tts",
    "whisper_turbo": "asr",
}

# ---- Generation defaults ----
DEFAULT_NUM_STEPS: int = int(os.environ.get("DEFAULT_NUM_STEPS", "4"))
DEFAULT_SEED: int = int(os.environ.get("DEFAULT_SEED", "42"))
# Default orientation for each model.
MODEL_A_DEFAULT_ORIENTATION: str = os.environ.get(
    "MODEL_A_DEFAULT_ORIENTATION", "w"
)
MODEL_B_DEFAULT_ORIENTATION: str = os.environ.get(
    "MODEL_B_DEFAULT_ORIENTATION", "w"
)

# Resolution presets (width x height) per orientation.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "w": (1024, 576),   # wide 16:9 (YouTube)
    "t": (576, 1024),   # vertical 9:16 (TikTok)
}

# ---- Runtime ----
# Host bind address inside the container.
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8000"))
# Where generated images are written.
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "/app/output")
# Precision used to load models on the GPU.
TORCH_DTYPE: str = os.environ.get("TORCH_DTYPE", "bfloat16")

# ---- Concurrency guard ----
# The GPU is serial by nature. We process ONE request at a time to avoid
# overloading it. This is enforced by a single-worker thread pool + a global
# asyncio.Lock in the API layer.
MAX_CONCURRENT: int = 1


def resolve_resolution(orientation: str) -> tuple[int, int]:
    """Return (width, height) for a given orientation code ('w' or 't')."""
    if orientation not in RESOLUTIONS:
        raise ValueError(
            f"Orientacao invalida '{orientation}'. Use 'w' (wide) ou 't' (vertical)."
        )
    return RESOLUTIONS[orientation]


def output_path(model: str, width: int, height: int, seed: int, ext: str | None = None) -> Path:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ext = ext or MEDIA_EXT.get(model, "png")
    return Path(OUTPUT_DIR) / f"{model}_{width}x{height}_seed{seed}.{ext}"
