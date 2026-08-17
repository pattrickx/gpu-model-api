"""FastAPI application exposing GPU image generation (one request at a time).

Endpoints
---------
GET  /health     -> service + which models are loaded
GET  /models     -> available model ids and repos
POST /generate   -> generate an image; returns the image file directly
                    (body: {model, prompt, orientation, num_inference_steps, seed})

All generation is serialized through a single-worker executor + global lock, so
the GPU never processes more than one request concurrently.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse

import app.config as cfg
import app.pipelines as pipelines
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpu-model-api")

app = FastAPI(
    title="GPU Model API",
    version="1.0.0",
    description=(
        "API de inferencia de modelos de imagem com GPU. "
        "Processa 1 request por vez para nao sobrecarregar a GPU."
    ),
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=pipelines.models_loaded())


@app.get("/models")
async def models() -> dict[str, dict[str, str]]:
    return {
        "flux": {"repo": cfg.MODEL_A_REPO, "default_orientation": cfg.MODEL_A_DEFAULT_ORIENTATION},
        "model2": {"repo": cfg.MODEL_B_REPO, "default_orientation": cfg.MODEL_B_DEFAULT_ORIENTATION},
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> FileResponse:
    """Generate an image and return it directly as a PNG file response."""
    logger.info(
        "generate %s | prompt=%r | orient=%s | steps=%s",
        req.model, req.prompt, req.orientation, req.num_inference_steps,
    )
    result = await pipelines.generate(
        model=req.model,
        prompt=req.prompt,
        orientation=req.orientation,
        steps=req.num_inference_steps,
        seed=req.seed,
    )
    return FileResponse(
        result["image_path"],
        media_type="image/png",
        filename=result["filename"],
    )


@app.get("/image/{filename}")
async def get_image(filename: str):
    """Serve a previously generated image by filename (read-only, no GPU)."""
    from pathlib import Path

    from fastapi import HTTPException

    path = Path(cfg.OUTPUT_DIR) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="imagem nao encontrada")
    return FileResponse(str(path), media_type="image/png")
