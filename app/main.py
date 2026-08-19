"""FastAPI application exposing GPU model inference (one request at a time).

Endpoints
---------
GET  /health      -> service + which models are loaded
GET  /models      -> available model ids, repos and tasks
POST /generate    -> generate from a prompt; returns the file directly
                     (image models -> PNG, TTS models -> WAV)
POST /transcribe  -> ASR: upload an audio file, returns JSON transcription
GET  /file/{filename} -> serve a previously generated file (no GPU)

All generation/transcription runs through a single-worker executor + global
lock, so the GPU never processes more than one request concurrently.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import app.config as cfg
import app.pipelines as pipelines
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpu-model-api")

app = FastAPI(
    title="GPU Model API",
    version="2.0.0",
    description=(
        "API de inferencia de modelos generativos com GPU (imagem, TTS, ASR). "
        "Processa 1 request por vez para nao sobrecarregar a GPU."
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
    return {
        "flux": cfg.MODEL_A_REPO,
        "model2": cfg.MODEL_B_REPO,
        "kokoro": cfg.MODEL_C_REPO,
        "qwen3tts": cfg.MODEL_D_REPO,
        "whisper_turbo": cfg.MODEL_E_REPO,
    }.get(mid, "")


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> FileResponse | JSONResponse:
    """Generate from a prompt and return the file directly.

    * image models (flux, model2) -> PNG
    * TTS models (kokoro, qwen3tts) -> WAV
    """
    logger.info("generate %s", req.model)
    result = await pipelines.generate(
        model=req.model,
        prompt=req.prompt,
        orientation=req.orientation,
        steps=req.num_inference_steps,
        seed=req.seed,
        text=req.text,
        voice=req.voice,
        language=req.language,
    )
    media = result["media_type"]
    if media == "application/json":
        return JSONResponse(result)
    return FileResponse(
        result["image_path"],
        media_type=media,
        filename=result["filename"],
    )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = "whisper_turbo",
    language: str | None = None,
) -> JSONResponse:
    """ASR: upload an audio file, returns JSON transcription (word-level)."""
    logger.info("transcribe %s | model=%s", file.filename, model)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        result = await pipelines.transcribe(
            model=model,
            audio_path=tmp_path,
            language=language,
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
