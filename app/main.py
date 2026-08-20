"""FastAPI application exposing GPU model inference (one request at a time).

Endpoints (separados por tipo de midia)
----------------------------------------
GET  /health            -> service + which models are loaded
GET  /models            -> available model ids, repos and tasks
POST /generate-image    -> image models (flux, model2) -> PNG
POST /generate-audio    -> TTS models (kokoro, qwen3tts) -> WAV
POST /transcribe-audio  -> ASR (whisper_turbo): upload audio -> JSON (lingua original)
GET  /file/{filename}   -> serve a previously generated file (no GPU)

All generation/transcription runs through a single-worker executor + global
lock, so the GPU never processes more than one request concurrently.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import app.config as cfg
import app.pipelines as pipelines
from app.schemas import (
    AudioRequest,
    AudioResponse,
    HealthResponse,
    ImageRequest,
    ImageResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpu-model-api")

app = FastAPI(
    title="GPU Model API",
    version="2.1.0",
    description=(
        "API de inferencia de modelos generativos com GPU (imagem, TTS, ASR). "
        "Endpoints separados por tipo de midia. Processa 1 request por vez para "
        "nao sobrecarregar a GPU."
    ),
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=pipelines.models_loaded())


@app.get("/models")
async def models() -> dict[str, dict[str, str]]:
    out = {}
    for mid in cfg.MODEL_TASK:
        out[mid] = {
            "repo": _repo_for(mid),
            "task": cfg.MODEL_TASK.get(mid, "image"),
            "media_type": cfg.MEDIA_TYPE.get(mid, "image/png"),
        }
    return out


def _repo_for(mid: str) -> str:
    """Repo do modelo, com o mapa de pipelines como fonte unica de verdade.

    Antes este mapa era duplicado aqui, e cada modelo novo precisava ser
    adicionado nos dois lugares -- os 5 modelos T2I novos apareciam com
    repo="" em /models justamente por isso.
    """
    return pipelines._REPOS.get(mid, "")


@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(req: ImageRequest = Body(...)) -> FileResponse:
    """Gera IMAGEM (flux, model2) a partir de um prompt -> PNG."""
    logger.info("generate-image %s", req.model)
    result = await pipelines.generate_image(
        model=req.model,
        prompt=req.prompt,
        orientation=req.orientation,
        steps=req.num_inference_steps,
        seed=req.seed,
    )
    return FileResponse(
        result["image_path"],
        media_type=result["media_type"],
        filename=result["filename"],
    )


@app.post("/generate-image-edit")
async def generate_image_edit(
    image: UploadFile = File(...),
    model: str = Form("flux2_klein"),
    prompt: str = Form(...),
    orientation: str = Form("w"),
    num_inference_steps: int = Form(4),
    seed: int | None = Form(None),
) -> FileResponse:
    """Edita IMAGEM (texto + imagem -> PNG) via FLUX.2-klein I2I.

    Entrada: arquivo de imagem + prompt de edicao. Modelos suportados
    (FLUX.2-klein unifica T2I e I2I num pipeline): flux2_klein,
    flux2_klein_fp8, flux2_klein_base.
    """
    I2I_MODELS = {"flux2_klein", "flux2_klein_fp8", "flux2_klein_base"}
    if model not in cfg.MODEL_TASK or model not in I2I_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"modelo '{model}' nao suporta edicao. Use flux2_klein/flux2_klein_fp8/flux2_klein_base.",
        )
    logger.info("generate-image-edit %s | model=%s", image.filename, model)
    data = await image.read()
    result = await pipelines.generate_image_edit(
        model=model,
        prompt=prompt,
        image_data=data,
        orientation=orientation,
        steps=num_inference_steps,
        seed=seed,
    )
    return FileResponse(
        result["image_path"],
        media_type=result["media_type"],
        filename=result["filename"],
    )


@app.post("/generate-audio", response_model=AudioResponse)
async def generate_audio(req: AudioRequest = Body(...)) -> FileResponse:
    """Sintetiza AUDIO/TTS (kokoro, qwen3tts, chatterbox) a partir de texto -> WAV."""
    logger.info("generate-audio %s", req.model)
    result = await pipelines.generate_audio(
        model=req.model,
        text=req.text,
        voice=req.voice,
        language=req.language,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
    )
    return FileResponse(
        result["image_path"],
        media_type=result["media_type"],
        filename=result["filename"],
    )


@app.post("/transcribe-audio")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper_turbo"),
) -> JSONResponse:
    """ASR: upload de audio -> JSON com transcricao na LINGUA ORIGINAL.

    O idioma e sempre detectado automaticamente pelo modelo (o parametro
    'language' e ignorado de proposito). O JSON de resposta traz o campo
    'language' com o idioma detectado.
    """
    if model not in cfg.MODEL_TASK or cfg.MODEL_TASK.get(model) != "asr":
        raise HTTPException(
            status_code=400,
            detail=f"modelo '{model}' nao e ASR. Use whisper_turbo.",
        )
    logger.info("transcribe-audio %s | model=%s", file.filename, model)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        result = await pipelines.transcribe(
            model=model,
            audio_path=tmp_path,
            language=None,  # sempre na lingua original
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    # Return the JSON transcription content directly.
    from pathlib import Path as _P

    content = _P(result["image_path"]).read_text(encoding="utf-8")
    return JSONResponse({"model": model, "transcription": content})


@app.get("/file/{filename}")
async def get_file(filename: str):
    """Serve a previously generated file by filename (read-only, no GPU)."""
    path = Path(cfg.OUTPUT_DIR) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="arquivo nao encontrado")
    # Guess media type from extension.
    ext = path.suffix.lower()
    mt = {
        ".png": "image/png",
        ".wav": "audio/wav",
        ".json": "application/json",
    }.get(ext, "application/octet-stream")
    return FileResponse(str(path), media_type=mt)
