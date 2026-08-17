"""Model pipelines for GPU image generation.

Design notes
------------
* Models are loaded **lazily** (first request that needs them) and cached in a
  module-level dict. They stay on the GPU across requests (no reload).
* All generation runs through a single global :class:`threading.Lock` and a
  1-worker ThreadPoolExecutor, so the GPU only ever handles **one** request at a
  time. Concurrent callers queue up. This prevents overloading the GPU.
* FLUX.1-schnell is loaded with 4-bit quantization (bitsandbytes) + bf16 VAE to
  fit a 12 GB GPU; the secondary model (SDXL) uses fp16.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

import app.config as cfg

# Global, single-worker executor: enforces "one request at a time" on the GPU.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")
# Guards the in-flight GPU job in addition to the executor (defensive).
_gpu_lock = threading.Lock()

# Cached pipelines. Keyed by model id ("flux" | "model2").
_PIPES: dict[str, Any] = {}
_PIPES_LOCK = threading.Lock()


def _load_flux() -> Any:
    """Load FLUX.1-schnell with 4-bit quantization + CPU offload + VAE tiling.

    This is the recipe that runs FLUX on a 4 GB GPU (e.g. GTX 1650): the
    4-bit transformer (~6 GB) lives in host RAM, layers are moved to the GPU
    one at a time (enable_model_cpu_offload), and the VAE decodes in tiles to
    avoid the VRAM spike. Requires enough host RAM for the quantized weights.
    """
    from diffusers import FluxPipeline

    repo = cfg.MODEL_A_REPO
    nf4 = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }
    pipe = FluxPipeline.from_pretrained(
        repo,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        quantization_config=nf4,
        low_cpu_mem_usage=True,
    )
    # Keep weights on CPU; stream layers to GPU per call (fits 4 GB VRAM).
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    return pipe


def _load_model2() -> Any:
    """Load the secondary model (SDXL base, fp16) onto the GPU."""
    from diffusers import StableDiffusionXLPipeline

    repo = cfg.MODEL_B_REPO
    pipe = StableDiffusionXLPipeline.from_pretrained(
        repo,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.to("cuda")
    return pipe


_LOADERS = {"flux": _load_flux, "model2": _load_model2}
_REPOS = {"flux": cfg.MODEL_A_REPO, "model2": cfg.MODEL_B_REPO}


def _get_pipe(model: str) -> Any:
    with _PIPES_LOCK:
        if model not in _PIPES:
            _PIPES[model] = _LOADERS[model]()
        return _PIPES[model]


def models_loaded() -> dict[str, bool]:
    with _PIPES_LOCK:
        return {m: m in _PIPES for m in _LOADERS}


def _run_job(
    model: str,
    repo: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> Path:
    """Synchronous GPU job. Must run inside the single-worker executor."""
    with _gpu_lock:
        pipe = _get_pipe(model)
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        out = cfg.output_path(model, width, height, seed)
        image.save(out)
        return out


async def generate(
    model: str,
    prompt: str,
    orientation: str,
    steps: int,
    seed: int | None,
) -> dict[str, Any]:
    """Schedule a generation. Blocks until the GPU is free (serialized)."""
    width, height = cfg.resolve_resolution(orientation)
    used_seed = cfg.DEFAULT_SEED if seed is None else seed
    repo = _REPOS[model]

    # Offload the blocking GPU work to the 1-worker executor so the event loop
    # stays responsive and requests are processed strictly one at a time.
    loop = asyncio.get_event_loop()
    out_path = await loop.run_in_executor(
        _executor,
        _run_job,
        model,
        repo,
        prompt,
        width,
        height,
        steps,
        used_seed,
    )
    return {
        "model": model,
        "repo": repo,
        "prompt": prompt,
        "orientation": orientation,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "seed": used_seed,
        "image_path": str(out_path),
        "filename": out_path.name,
    }
