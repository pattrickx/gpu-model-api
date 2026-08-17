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


def output_path(model: str, width: int, height: int, seed: int) -> Path:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return Path(OUTPUT_DIR) / f"{model}_{width}x{height}_seed{seed}.png"
