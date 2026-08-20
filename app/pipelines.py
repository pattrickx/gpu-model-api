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


def _load_chatterbox() -> Any:
    """Load ResembleAI Chatterbox TTS (MIT, 23+ linguas incl. PT-BR).

    Modelo 0.5B, roda em ~24s/audio na 3060 12GB. Suporta tuning de
    exaggeration/cfg_weight. Carregado sob demanda (lazy) como os outros.
    """
    from chatterbox.tts import ChatterboxTTS

    return ChatterboxTTS.from_pretrained(device="cuda")


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


def _load_crisper2() -> Any:
    """Load CrisperWhisper2.0_large ASR (fork do Whisper, timestamps precisos).

    Usa o pipeline transformers padrao de ASR (igual ao whisper_turbo); aceita
    return_timestamps='word' para timestamps de palavra.
    """
    from transformers import pipeline as hf_pipeline

    repo = cfg.MODEL_F_REPO
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=repo,
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    return asr


def _load_parakeet() -> Any:
    """Load Parakeet TDT 0.6B V3 ASR (ingles, word/char/segment nativos).

    O pipeline transformers do Parakeet TDT usa ``timestamps=True`` (e nao
    return_timestamps='word', que nao e suportado por este modelo).
    """
    from transformers import pipeline as hf_pipeline

    repo = cfg.MODEL_G_REPO
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=repo,
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    return asr


def _load_qwen3asr_06b() -> Any:
    """Load Qwen3-ASR-0.6B (multilíngue) + ForcedAligner para word timestamps.

    Usa a API propria do pacote ``qwen_asr`` (Qwen3ASRModel), nao o pipeline
    transformers padrao. O ForcedAligner gera word-level timestamps.
    """
    import torch
    from qwen_asr import Qwen3ASRModel

    repo = cfg.MODEL_H_REPO
    model = Qwen3ASRModel.from_pretrained(
        repo,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=256,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )
    return model


def _load_qwen3asr_17b() -> Any:
    """Load Qwen3-ASR-1.7B (multilíngue) + ForcedAligner para word timestamps."""
    import torch
    from qwen_asr import Qwen3ASRModel

    repo = cfg.MODEL_I_REPO
    model = Qwen3ASRModel.from_pretrained(
        repo,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=256,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )
    return model


def _vae_memory_savings(pipe: Any) -> None:
    """Aplica VAE tiling/slicing onde existir.

    Pipelines novos (Flux2KleinPipeline no diffusers 0.39) NAO expoem
    enable_vae_tiling/enable_vae_slicing no pipeline; os metodos vivem no
    proprio vae (vae.enable_tiling/enable_slicing). Tentar no pipeline levanta
    AttributeError, por isso a checagem com hasattr.
    """
    vae = getattr(pipe, "vae", None)
    for obj, meth in (
        (pipe, "enable_vae_tiling"),
        (pipe, "enable_vae_slicing"),
        (vae, "enable_tiling"),
        (vae, "enable_slicing"),
    ):
        if obj is not None and hasattr(obj, meth):
            try:
                getattr(obj, meth)()
            except Exception:  # pragma: no cover - best effort
                pass


def _load_sdxl_turbo() -> Any:
    """SDXL-Turbo: destilado 1-4 steps, 512x512 fixo, guidance_scale=0.

    Medido: ~7.9GB VRAM de pico, ~19s para 4 steps.
    """
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.MODEL_J_REPO,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing()
    _vae_memory_savings(pipe)
    pipe.to("cuda")
    return pipe


def _load_sd_turbo() -> Any:
    """SD-Turbo: base SD2.1 destilada, 512x512, o mais leve do conjunto.

    Medido: ~3.0GB VRAM de pico, ~1.9s para 4 steps.
    """
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.MODEL_K_REPO,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing()
    _vae_memory_savings(pipe)
    pipe.to("cuda")
    return pipe


def _load_sdxl_lightning() -> Any:
    """SDXL-Lightning: o repo traz apenas UNet/LoRA, nao um pipeline completo.

    Receita validada: carrega o UNet destilado de 4 steps sobre o SDXL base e
    troca o scheduler para Euler com timestep_spacing='trailing' (exigido pelo
    Lightning). guidance_scale deve ser 0.
    Medido: ~9.6GB VRAM de pico, ~4.9s para 4 steps em 1024x576.
    """
    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    base = cfg.SDXL_LIGHTNING_BASE
    unet = UNet2DConditionModel.from_config(
        UNet2DConditionModel.load_config(base, subfolder="unet")
    ).to("cuda", torch.float16)
    unet.load_state_dict(
        load_file(
            hf_hub_download(cfg.MODEL_L_REPO, cfg.SDXL_LIGHTNING_CKPT),
            device="cuda",
        )
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_attention_slicing()
    _vae_memory_savings(pipe)
    pipe.to("cuda")
    return pipe


def _load_flux2_klein() -> Any:
    """FLUX.2-klein-4B (DiT 4B) em bf16 com sequential CPU offload.

    Requer diffusers>=0.37 (Flux2KleinPipeline). Com offload sequencial o pico
    de VRAM fica em ~1.4GB e 4 steps levam ~258s em 1024x576 -- ~2.7x mais
    rapido que o FLUX.1-schnell nesta maquina.
    """
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        cfg.FLUX2_KLEIN_BASE,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_sequential_cpu_offload()
    _vae_memory_savings(pipe)
    return pipe


def _load_flux2_klein_fp8() -> Any:
    """FLUX.2-klein-4b-fp8: checkpoint fp8 solto -> dequantiza para bf16.

    O repo fp8 NAO e um repo diffusers: traz um unico safetensors com o
    transformer quantizado em fp8 E4M3 e escalas PER-TENSOR escalares
    (`*.weight_scale`, shape []). O `from_single_file` do diffusers 0.39 falha
    nesse formato com "chunk expects at least a 1-dimensional tensor", pois
    tenta fatiar o qkv fundido sem tratar as escalas.

    Receita validada: multiplica cada peso fp8 pela sua escala, descarta as
    chaves de escala, grava um safetensors bf16 temporario (cacheado em disco) e
    entrega ao conversor oficial, apontando a config para o repo klein-4B (o
    default seria FLUX.2-dev, que e gated).

    NOTA: apos dequantizar o consumo de memoria iguala o do bf16 -- este loader
    valida o checkpoint fp8, nao economiza VRAM.
    """
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    src = hf_hub_download(cfg.MODEL_N_REPO, cfg.FLUX2_KLEIN_FP8_FILE)
    cache_dir = Path(cfg.OUTPUT_DIR) / "_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dequant = cache_dir / "flux2_klein_4b_fp8_dequant.safetensors"

    if not dequant.exists():
        state = load_file(src)
        converted: dict[str, Any] = {}
        for key, value in state.items():
            if key.endswith("_scale") and value.ndim == 0:
                continue  # escala consumida junto do peso correspondente
            if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                weight = value.to(torch.bfloat16)
                scale = state.get(key + "_scale")
                if scale is not None:
                    weight = weight * scale.to(torch.bfloat16)
                converted[key] = weight
            else:
                converted[key] = (
                    value.to(torch.bfloat16) if value.is_floating_point() else value
                )
        tmp = dequant.with_suffix(".tmp")
        save_file(converted, str(tmp))
        tmp.replace(dequant)  # publica atomicamente (evita cache corrompido)
        del state, converted

    transformer = Flux2Transformer2DModel.from_single_file(
        str(dequant),
        torch_dtype=torch.bfloat16,
        config=cfg.FLUX2_KLEIN_BASE,
        subfolder="transformer",
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        cfg.FLUX2_KLEIN_BASE,
        transformer=transformer,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_sequential_cpu_offload()
    _vae_memory_savings(pipe)
    return pipe


def _load_flux2_klein_base() -> Any:
    """FLUX.2-klein-base-4B: variante NAO destilada (para fine-tune).

    Diferente dos demais FLUX.2-klein (destilados, 4 steps, guidance=0), este e
    um checkpoint base e exige MAIS steps (testado: 20) e aceita guidance>0.
    Requer diffusers>=0.37 (Flux2KleinPipeline). bf16 + offload sequencial.
    Testado isoladamente: ~720s para 20 steps em 1024x576, pico ~1.4GB VRAM.
    """
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        cfg.MODEL_O_REPO,
        token=cfg.HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_sequential_cpu_offload()
    _vae_memory_savings(pipe)
    return pipe


_LOADERS = {
    "flux": _load_flux,
    "model2": _load_model2,
    "kokoro": _load_kokoro,
    "qwen3tts": _load_qwen3tts,
    "whisper_turbo": _load_whisper_turbo,
    "crisper2": _load_crisper2,
    "parakeet": _load_parakeet,
    "qwen3asr_06b": _load_qwen3asr_06b,
    "qwen3asr_17b": _load_qwen3asr_17b,
    "sdxl_turbo": _load_sdxl_turbo,
    "sd_turbo": _load_sd_turbo,
    "sdxl_lightning": _load_sdxl_lightning,
    "flux2_klein": _load_flux2_klein,
    "flux2_klein_fp8": _load_flux2_klein_fp8,
    "flux2_klein_base": _load_flux2_klein_base,
    "chatterbox": _load_chatterbox,
}
_REPOS = {
    "flux": cfg.MODEL_A_REPO,
    "model2": cfg.MODEL_B_REPO,
    "kokoro": cfg.MODEL_C_REPO,
    "qwen3tts": cfg.MODEL_D_REPO,
    "whisper_turbo": cfg.MODEL_E_REPO,
    "crisper2": cfg.MODEL_F_REPO,
    "parakeet": cfg.MODEL_G_REPO,
    "qwen3asr_06b": cfg.MODEL_H_REPO,
    "qwen3asr_17b": cfg.MODEL_I_REPO,
    "sdxl_turbo": cfg.MODEL_J_REPO,
    "sd_turbo": cfg.MODEL_K_REPO,
    "sdxl_lightning": cfg.MODEL_L_REPO,
    "flux2_klein": cfg.MODEL_M_REPO,
    "flux2_klein_fp8": cfg.MODEL_N_REPO,
    "flux2_klein_base": cfg.MODEL_O_REPO,
    "chatterbox": cfg.MODEL_P_REPO,
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
        elif model in cfg.ZERO_GUIDANCE_MODELS:
            # sdxl_turbo / sd_turbo / sdxl_lightning sao destilados: guidance
            # classifier-free desligado e obrigatorio (com CFG saem borrados).
            extra = {"guidance_scale": 0.0}
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


def _run_edit_job(
    model: str,
    repo: str,
    prompt: str,
    image_path: Path,
    width: int,
    height: int,
    steps: int,
    seed: int,
) -> Path:
    """Synchronous IMAGE-EDITING job (text + image -> image).

    FLUX.2-klein unifica T2I e I2I num so pipeline: basta passar a imagem de
    entrada como ``image=``. Diferente do img2img classico do SD, NAO aceita
    ``strength`` (edicao in-painting-style unificada). O modelo funde
    prompt + imagem (multi-reference editing).

    Modelos destilados (flux2_klein, flux2_klein_fp8) usam guidance=0; o base
    (flux2_klein_base, nao destilado) aceita guidance>0.
    """
    from PIL import Image

    with _gpu_lock:
        pipe = _get_pipe(model)
        generator = torch.Generator("cuda").manual_seed(seed)
        init = Image.open(image_path).convert("RGB")
        extra = {}
        if model in cfg.ZERO_GUIDANCE_MODELS or model in (
            "flux2_klein", "flux2_klein_fp8"
        ):
            extra = {"guidance_scale": 0.0}
        image = pipe(
            prompt=prompt,
            image=init,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            **extra,
        ).images[0]
        out = cfg.output_path(model, width, height, seed)
        image.save(out)
        return out


async def generate_image_edit(
    model: str,
    prompt: str,
    image_data: bytes,
    orientation: str = "w",
    steps: int = 4,
    seed: int | None = None,
) -> dict[str, Any]:
    """Schedule an IMAGE-EDITING job (text + image -> PNG). Serialized on GPU."""
    import tempfile

    used_seed = cfg.DEFAULT_SEED if seed is None else seed
    repo = _REPOS[model]
    loop = asyncio.get_event_loop()

    width, height = cfg.resolve_resolution(orientation, model)
    # Salva a imagem de entrada num temp para o job sincrono ler.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_data)
        tmp_img = Path(f.name)
    try:
        out_path = await loop.run_in_executor(
            _executor,
            _run_edit_job,
            model,
            repo,
            prompt,
            tmp_img,
            width,
            height,
            steps,
            used_seed,
        )
    finally:
        tmp_img.unlink(missing_ok=True)
    _unload_all()  # libera VRAM apos o job
    return {
        "model": model,
        "repo": repo,
        "task": "image_edit",
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


def _run_tts_job(
    model: str,
    repo: str,
    text: str,
    voice: str | None,
    language: str | None,
    seed: int,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
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
        elif model == "chatterbox":
            # Chatterbox TTS: gera tensor [1, T] ou [T]; SR=24000.
            # Suporta tuning de estilo via exaggeration/cfg_weight.
            wav = pipe.generate(
                text,
                exaggeration=float(exaggeration or 0.5),
                cfg_weight=float(cfg_weight or 0.5),
            )
            if hasattr(wav, "cpu"):
                wav = wav.cpu().numpy()
            audio = wav.squeeze()
            sr = pipe.sr
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
    """Synchronous ASR job (audio -> JSON transcription). 1-worker executor.

    SEMPRE transcreve na LINGUA ORIGINAL do audio: o parametro ``language`` e
    ignorado de proposito e o Whisper detecta o idioma automaticamente. O idioma
    detectado e retornado no campo ``language`` do JSON de saida.

    O audio e lido com a biblioteca padrao ``wave`` + ``numpy`` (sem dependencias
    de libsndfile/ffmpeg do sistema) e passado ao pipeline como array numpy,
    contornando o ``ffmpeg_read`` do transformers (que falha nesta imagem por
    nao ter ffmpeg/libsndfile nativos).

    Os modelos Qwen3-ASR (qwen3asr_06b/17b) usam a API propria do pacote
    ``qwen_asr`` (nao o pipeline transformers); sao roteados para
    ``_run_qwen_asr_job``.
    """
    if model in ("qwen3asr_06b", "qwen3asr_17b"):
        return _run_qwen_asr_job(model, repo, audio_path, language, seed)

    import json
    import wave

    import numpy as np

    with _gpu_lock:
        pipe = _get_pipe(model)
        # Le o WAV com a lib padrao (PCM) e converte para float32 mono.
        with wave.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        # Normaliza para [-1, 1].
        max_val = np.iinfo(dtype).max
        audio_array = samples / max_val
        # Nao passamos 'language' -> o modelo detecta a lingua original sozinho.
        # O Parakeet TDT usa timestamps=True (return_timestamps='word' quebra).
        if model == "parakeet":
            result = pipe(
                {"array": audio_array, "sampling_rate": sr},
                timestamps=True,
            )
            text_out = result.get("text", "") if isinstance(result, dict) else str(result)
            # Parakeet TDT embute os timestamps no texto como marcadores <|t.xx|>;
            # nao ha chunks estruturados no pipeline padrao.
            chunks = []
            detected = None  # Parakeet e monolingue (ingles)
        else:
            result = pipe(
                {"array": audio_array, "sampling_rate": sr},
                return_timestamps="word",
            )
            text_out = result.get("text", "")
            chunks = result.get("chunks", [])
            detected = result.get("language") or None

        out = cfg.output_path(model, 0, 0, seed, ext="json")
        out.write_text(
            json.dumps(
                {"text": text_out, "chunks": chunks, "language": detected},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return out


def _run_qwen_asr_job(
    model: str,
    repo: str,
    audio_path: Path,
    language: str | None,
    seed: int,
) -> Path:
    """Synchronous ASR job para Qwen3-ASR (0.6B/1.7B) via pacote ``qwen_asr``.

    SEMPRE transcreve na LINGUA ORIGINAL: ``language=None`` deixa o modelo
    detectar o idioma. O ForcedAligner (carregado junto) gera word-level
    timestamps (return_time_stamps=True). O texto e os timestamps sao salvos
    no JSON de saida.
    """
    import json

    import numpy as np
    import soundfile as sf
    import wave

    with _gpu_lock:
        pipe = _get_pipe(model)
        # Le o audio e salva como WAV (a API do qwen_asr aceita path ou array).
        with wave.open(str(audio_path), "rb") as wf:
            n_ch = wf.getnchannels()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        audio = samples / np.iinfo(dtype).max
        tmp_wav = str(audio_path) + ".qwen.wav"
        sf.write(tmp_wav, audio, sr)
        try:
            results = pipe.transcribe(
                audio=[tmp_wav], language=None, return_time_stamps=True
            )
        finally:
            Path(tmp_wav).unlink(missing_ok=True)
        r = results[0]
        text_out = getattr(r, "text", "") or ""
        detected = getattr(r, "language", None)
        # Word timestamps do ForcedAligner (ForcedAlignItem com start/end).
        stamps = getattr(r, "time_stamps", []) or []
        chunks = [
            {
                "text": getattr(s, "text", ""),
                "start_time": getattr(s, "start_time", None),
                "end_time": getattr(s, "end_time", None),
            }
            for s in stamps
        ]
        out = cfg.output_path(model, 0, 0, seed, ext="json")
        out.write_text(
            json.dumps(
                {"text": text_out, "chunks": chunks, "language": detected},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return out


async def generate_image(
    model: str,
    prompt: str,
    orientation: str = "w",
    steps: int = 4,
    seed: int | None = None,
) -> dict[str, Any]:
    """Schedule an IMAGE generation. Blocks until the GPU is free (serialized).

    image (flux, model2) -> _run_job (PNG).
    """
    used_seed = cfg.DEFAULT_SEED if seed is None else seed
    repo = _REPOS[model]
    loop = asyncio.get_event_loop()

    width, height = cfg.resolve_resolution(orientation, model)
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


async def generate_audio(
    model: str,
    text: str,
    voice: str | None = None,
    language: str | None = None,
    seed: int | None = None,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
) -> dict[str, Any]:
    """Schedule a TTS (audio) generation. Blocks until the GPU is free.

    tts (kokoro, qwen3tts, chatterbox) -> _run_tts_job (WAV).
    """
    used_seed = cfg.DEFAULT_SEED if seed is None else seed
    repo = _REPOS[model]
    loop = asyncio.get_event_loop()

    out_path = await loop.run_in_executor(
        _executor,
        _run_tts_job,
        model,
        repo,
        text,
        voice,
        language,
        used_seed,
        exaggeration,
        cfg_weight,
    )
    _unload_all()  # libera VRAM apos o job (diretriz de economia de recursos)
    return {
        "model": model,
        "repo": repo,
        "task": "tts",
        "media_type": cfg.MEDIA_TYPE.get(model, "audio/wav"),
        "text": text,
        "voice": voice,
        "language": language,
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
