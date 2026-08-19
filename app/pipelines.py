"""Model pipelines for GPU image/TTS/ASR generation.

Design notes
------------
* Models are loaded **lazily** (first request that needs them) and cached in a
  module-level dict. They stay on the GPU across requests (no reload) UNTIL a
  descarga explicita (ver _unload_all) libera a VRAM ao fim de um job, conforme
  a diretriz de economia de recursos (carregar so quando usado, descarregar apos).
* All generation runs through a single global :class:`threading.Lock` and a
  1-worker ThreadPoolExecutor, so the GPU only ever handles **one** request at a
  time. Concurrent callers queue up. This prevents overloading the GPU.
* FLUX.1-schnell is loaded with 4-bit quantization (bitsandbytes) + bf16 VAE to
  fit a 12 GB GPU; the secondary model (SDXL) uses fp16.
* TTS/ASR models (kokoro, qwen3tts, whisper_turbo) follow the same lazy+unload
  policy. Qwen3-TTS runs in an isolated subprocess (qwen_worker.py) because it
  deadlocks the CUDA context when loaded inside the uvicorn process.
"""

from __future__ import annotations

import asyncio
import subprocess
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

# Cached pipelines. Keyed by model id.
_PIPES: dict[str, Any] = {}
_PIPES_LOCK = threading.Lock()


def _unload_all() -> None:
    """Libera todos os modelos da GPU e limpa o cache de VRAM.

    Segue a diretriz de economia de recursos: apos um job, os modelos sao
    descarregados para que a proxima requisicao (possivelmente de outro modelo)
    encontre a GPU livre.
    """
    with _PIPES_LOCK:
        for mid in list(_PIPES.keys()):
            pipe = _PIPES.pop(mid, None)
            del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_flux() -> Any:
    """Load FLUX.1-schnell the same way it runs on a 4 GB GPU (GTX 1650).

    Mirrors the working Windows script:
      * from_pretrained with torch_dtype=bfloat16, NO 4-bit quantization
        (quantization with bitsandbytes kept the transformer in host RAM and
        OOM-killed this low-RAM host).
      * enable_sequential_cpu_offload() moves each submodule (text encoder,
        transformer, VAE) to the GPU one at a time, so only one lives on the
        4 GB VRAM at once.
      * enable_vae_tiling() avoids the VAE decode VRAM spike.
    """
    from diffusers import FluxPipeline

    repo = cfg.MODEL_A_REPO
    pipe = FluxPipeline.from_pretrained(
        repo,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Sequential offload: one submodule on the GPU at a time (fits 4 GB VRAM).
    pipe.enable_sequential_cpu_offload()
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


def _load_kokoro() -> Any:
    """Load Kokoro-82M TTS (lightweight, fast, natural voice)."""
    import kokoro
    # Kokoro resolves its own weights; keep on CUDA for fast synthesis.
    return kokoro.KPipeline(lang_code="p")  # 'p' = portugues (multilingue)


def _load_qwen3tts() -> Any:
    """Load Qwen3-TTS-1.7B-CustomVoice (multilingual, named voices).

    Usa device_map=cuda:0 (funciona na thread principal do startup do uvicorn,
    assim como no teste isolado). O .to() direto trava o CUDA context neste
    ambiente.
    """
    from qwen_tts import Qwen3TTSModel

    repo = cfg.MODEL_D_REPO
    model = Qwen3TTSModel.from_pretrained(
        repo,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    return model


def _load_whisper_turbo() -> Any:
    """Load Whisper-large-v3-turbo ASR (word-level timestamps)."""
    from transformers import pipeline as hf_pipeline

    repo = cfg.MODEL_E_REPO
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=repo,
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    return asr


_LOADERS = {
    "flux": _load_flux,
    "model2": _load_model2,
    "kokoro": _load_kokoro,
    "qwen3tts": _load_qwen3tts,
    "whisper_turbo": _load_whisper_turbo,
}
_REPOS = {
    "flux": cfg.MODEL_A_REPO,
    "model2": cfg.MODEL_B_REPO,
    "kokoro": cfg.MODEL_C_REPO,
    "qwen3tts": cfg.MODEL_D_REPO,
    "whisper_turbo": cfg.MODEL_E_REPO,
}


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
        # FLUX.1-schnell uses guidance_scale=0 and a longer token context,
        # matching the working 4 GB script.
        extra = {}
        if model == "flux":
            extra = {"guidance_scale": 0, "max_sequence_length": 512}
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            **extra,
        ).images[0]
        out = cfg.output_path(model, width, height, seed)
        image.save(out)
        return out


def _run_tts_job(
    model: str,
    repo: str,
    text: str,
    voice: str | None,
    language: str | None,
    seed: int,
) -> Path:
    """Synchronous TTS job (text -> WAV). Runs inside the 1-worker executor.

    qwen3tts e tratado num subprocesso isolado (qwen_worker.py) porque o
    Qwen3TTSModel deadlocka o CUDA context quando carregado dentro do processo
    do uvicorn. Os demais (kokoro) carregam lazy normalmente.
    """
    if model == "qwen3tts":
        import subprocess, sys as _sys, os as _os
        worker = _os.path.join(_os.path.dirname(__file__), "qwen_worker.py")
        v = voice or "Ryan"
        lang = language or "English"
        out = cfg.output_path(model, 0, 0, seed, ext="wav")
        r = subprocess.run(
            [_sys.executable, worker, text, v, lang, str(out)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"qwen_worker falhou (rc={r.returncode}): {r.stderr[-500:]}"
            )
        return out

    with _gpu_lock:
        pipe = _get_pipe(model)
        import soundfile as sf

        if model == "kokoro":
            # Kokoro KPipeline.generate() -> generator de tuplas
            # (graphemes, phonemes, audio) OU (text, phonemes, audio);
            # o ultimo elemento e o numpy array de audio. SR fixo = 24000.
            voice = voice or "af_heart"
            generator = pipe(text, voice=voice)
            chunk = next(generator)
            audio = chunk[-1]
            sr = 24000
            ext = "wav"
        else:
            raise ValueError(f"Modelo TTS desconhecido: {model}")
        out = cfg.output_path(model, 0, 0, seed, ext=ext)
        sf.write(str(out), audio, sr)
        return out


def _run_asr_job(
    model: str,
    repo: str,
    audio_path: Path,
    language: str | None,
    seed: int,
) -> Path:
    """Synchronous ASR job (audio -> JSON transcription). 1-worker executor."""
    with _gpu_lock:
        pipe = _get_pipe(model)
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        result = pipe(
            str(audio_path),
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
        )
        import json

        text_out = result.get("text", "")
        chunks = result.get("chunks", [])
        out = cfg.output_path(model, 0, 0, seed, ext="json")
        out.write_text(
            json.dumps(
                {"text": text_out, "chunks": chunks, "language": language},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return out


async def generate(
    model: str,
    prompt: str | None = None,
    orientation: str = "w",
    steps: int = 4,
    seed: int | None = None,
    text: str | None = None,
    voice: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Schedule a generation. Blocks until the GPU is free (serialized).

    Dispatches by model task:
      * image (flux, model2)  -> _run_job  (PNG)
      * tts    (kokoro, qwen3tts) -> _run_tts_job (WAV)
      * asr    (whisper_turbo)    -> handled by transcribe() (JSON)
    """
    task = cfg.MODEL_TASK.get(model, "image")
    used_seed = cfg.DEFAULT_SEED if seed is None else seed
    repo = _REPOS[model]

    loop = asyncio.get_event_loop()

    if task == "tts":
        text = text or prompt or ""
        out_path = await loop.run_in_executor(
            _executor,
            _run_tts_job,
            model,
            repo,
            text,
            voice,
            language,
            used_seed,
        )
        ext = cfg.MEDIA_EXT.get(model, "wav")
        _unload_all()  # libera VRAM apos o job (diretriz de economia de recursos)
        return {
            "model": model,
            "repo": repo,
            "task": task,
            "media_type": cfg.MEDIA_TYPE.get(model, "audio/wav"),
            "text": text,
            "voice": voice,
            "language": language,
            "image_path": str(out_path),
            "filename": out_path.name,
        }

    # image path (default)
    width, height = cfg.resolve_resolution(orientation)
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
    _unload_all()  # libera VRAM apos o job (diretriz de economia de recursos)
    return {
        "model": model,
        "repo": repo,
        "task": "image",
        "media_type": cfg.MEDIA_TYPE.get(model, "image/png"),
        "prompt": prompt,
        "orientation": orientation,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "seed": used_seed,
        "image_path": str(out_path),
        "filename": out_path.name,
    }


async def transcribe(
    model: str,
    audio_path: Path,
    language: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Schedule an ASR transcription. Serialized like generate()."""
    used_seed = cfg.DEFAULT_SEED if seed is None else seed
    repo = _REPOS[model]
    loop = asyncio.get_event_loop()
    out_path = await loop.run_in_executor(
        _executor,
        _run_asr_job,
        model,
        repo,
        audio_path,
        language,
        used_seed,
    )
    _unload_all()  # libera VRAM apos o job (diretriz de economia de recursos)
    return {
        "model": model,
        "repo": repo,
        "task": "asr",
        "media_type": cfg.MEDIA_TYPE.get(model, "application/json"),
        "language": language,
        "image_path": str(out_path),
        "filename": out_path.name,
    }
