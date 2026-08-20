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
# TTS model: ResembleAI Chatterbox (MIT, 23+ línguas incl. PT-BR, 0.5B).
MODEL_P_REPO: str = os.environ.get("MODEL_P_REPO", "ResembleAI/chatterbox")
# ASR model: Whisper-large-v3-turbo (word-level timestamps, ~4x mais rapido).
MODEL_E_REPO: str = os.environ.get(
    "MODEL_E_REPO", "openai/whisper-large-v3-turbo"
)
# ASR model: CrisperWhisper2.0_large (fork do Whisper, timestamps precisos).
MODEL_F_REPO: str = os.environ.get(
    "MODEL_F_REPO", "nyralabs/CrisperWhisper2.0_large"
)
# ASR model: Parakeet TDT 0.6B V3 (ingles, word/char/segment nativos).
MODEL_G_REPO: str = os.environ.get(
    "MODEL_G_REPO", "nvidia/parakeet-tdt-0.6b-v3"
)
# ASR model: Qwen3-ASR-0.6B (multilíngue, word timestamps via ForcedAligner).
MODEL_H_REPO: str = os.environ.get(
    "MODEL_H_REPO", "Qwen/Qwen3-ASR-0.6B"
)
# ASR model: Qwen3-ASR-1.7B (multilíngue, word timestamps via ForcedAligner).
MODEL_I_REPO: str = os.environ.get(
    "MODEL_I_REPO", "Qwen/Qwen3-ASR-1.7B"
)
# T2I: SDXL-Turbo (1-4 steps, 512x512 fixo, guidance_scale=0 obrigatorio).
MODEL_J_REPO: str = os.environ.get("MODEL_J_REPO", "stabilityai/sdxl-turbo")
# T2I: SD-Turbo (base SD2.1, 1-4 steps, 512x512, o mais leve ~3GB VRAM).
MODEL_K_REPO: str = os.environ.get("MODEL_K_REPO", "stabilityai/sd-turbo")
# T2I: SDXL-Lightning (UNet destilado 4-step aplicado sobre o SDXL base).
MODEL_L_REPO: str = os.environ.get("MODEL_L_REPO", "ByteDance/SDXL-Lightning")
# T2I: FLUX.2-klein-4B (DiT 4B, bf16 + sequential offload; requer diffusers>=0.37).
MODEL_M_REPO: str = os.environ.get(
    "MODEL_M_REPO", "black-forest-labs/FLUX.2-klein-4B"
)
# T2I: FLUX.2-klein-4b-fp8 (checkpoint fp8 solto, dequantizado para bf16 no load).
MODEL_N_REPO: str = os.environ.get(
    "MODEL_N_REPO", "black-forest-labs/FLUX.2-klein-4b-fp8"
)
# T2I: FLUX.2-klein-4B base (nao destilado; mais steps, aceita guidance>0).
MODEL_O_REPO: str = os.environ.get(
    "MODEL_O_REPO", "black-forest-labs/FLUX.2-klein-base-4B"
)

# Checkpoint do UNet destilado do SDXL-Lightning (4 steps) e base sobre a qual
# ele e aplicado (o repo Lightning contem apenas UNet/LoRA, nao um pipeline).
SDXL_LIGHTNING_CKPT: str = os.environ.get(
    "SDXL_LIGHTNING_CKPT", "sdxl_lightning_4step_unet.safetensors"
)
SDXL_LIGHTNING_BASE: str = os.environ.get(
    "SDXL_LIGHTNING_BASE", "stabilityai/stable-diffusion-xl-base-1.0"
)
# Arquivo unico de pesos fp8 do FLUX.2-klein.
FLUX2_KLEIN_FP8_FILE: str = os.environ.get(
    "FLUX2_KLEIN_FP8_FILE", "flux-2-klein-4b-fp8.safetensors"
)
# Repo que fornece a config/componentes do pipeline FLUX.2-klein (usado tanto
# pelo bf16 quanto pelo fp8, que traz apenas o transformer).
FLUX2_KLEIN_BASE: str = os.environ.get(
    "FLUX2_KLEIN_BASE", "black-forest-labs/FLUX.2-klein-4B"
)

# Modelos T2I com resolucao FIXA imposta pelo checkpoint (turbo = 512x512).
# Para eles a orientacao e ignorada e a resolucao e forcada.
FIXED_RESOLUTION: dict[str, tuple[int, int]] = {
    "sdxl_turbo": (512, 512),
    "sd_turbo": (512, 512),
}
# Modelos destilados que EXIGEM guidance_scale=0.0.
ZERO_GUIDANCE_MODELS: tuple[str, ...] = (
    "sdxl_turbo", "sd_turbo", "sdxl_lightning",
)

# Tipo de mídia de saída por modelo (seguindo o padrão de 1 executor serializado).
# image -> PNG, tts -> WAV, asr -> JSON (transcrição).
MEDIA_TYPE: dict[str, str] = {
    "flux": "image/png",
    "model2": "image/png",
    "kokoro": "audio/wav",
    "qwen3tts": "audio/wav",
    "whisper_turbo": "application/json",
    "crisper2": "application/json",
    "parakeet": "application/json",
    "qwen3asr_06b": "application/json",
    "qwen3asr_17b": "application/json",
    "sdxl_turbo": "image/png",
    "sd_turbo": "image/png",
    "sdxl_lightning": "image/png",
    "flux2_klein": "image/png",
    "flux2_klein_fp8": "image/png",
    "flux2_klein_base": "image/png",
}
# Extensão de arquivo de saída por modelo.
MEDIA_EXT: dict[str, str] = {
    "flux": "png",
    "model2": "png",
    "kokoro": "wav",
    "qwen3tts": "wav",
    "whisper_turbo": "json",
    "crisper2": "json",
    "parakeet": "json",
    "qwen3asr_06b": "json",
    "qwen3asr_17b": "json",
    "sdxl_turbo": "png",
    "sd_turbo": "png",
    "sdxl_lightning": "png",
    "flux2_klein": "png",
    "flux2_klein_fp8": "png",
    "flux2_klein_base": "png",
}
# Task (tipo de geração) por modelo.
MODEL_TASK: dict[str, str] = {
    "flux": "image",
    "model2": "image",
    "kokoro": "tts",
    "qwen3tts": "tts",
    "whisper_turbo": "asr",
    "crisper2": "asr",
    "parakeet": "asr",
    "qwen3asr_06b": "asr",
    "qwen3asr_17b": "asr",
    "sdxl_turbo": "image",
    "sd_turbo": "image",
    "sdxl_lightning": "image",
    "flux2_klein": "image",
    "flux2_klein_fp8": "image",
    "flux2_klein_base": "image",
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


def resolve_resolution(orientation: str, model: str | None = None) -> tuple[int, int]:
    """Return (width, height) for a given orientation code ('w' or 't').

    Se o modelo tiver resolucao FIXA imposta pelo checkpoint (sdxl_turbo e
    sd_turbo sao treinados so em 512x512 e degradam fora disso), a orientacao e
    ignorada e a resolucao fixa e retornada.
    """
    if model and model in FIXED_RESOLUTION:
        return FIXED_RESOLUTION[model]
    if orientation not in RESOLUTIONS:
        raise ValueError(
            f"Orientacao invalida '{orientation}'. Use 'w' (wide) ou 't' (vertical)."
        )
    return RESOLUTIONS[orientation]


def output_path(model: str, width: int, height: int, seed: int, ext: str | None = None) -> Path:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ext = ext or MEDIA_EXT.get(model, "png")
    return Path(OUTPUT_DIR) / f"{model}_{width}x{height}_seed{seed}.{ext}"
