"""FastAPI application exposing GPU image generation (one request at a time).

Endpoints
---------
GET  /health            -> service + which models are loaded
GET  /models            -> available model ids and repos
POST /generate/flux     -> generate with FLUX.1-schnell (model A)
POST /generate/model2   -> generate with the secondary model (model B)

All generation is serialized through a single-worker executor + global lock, so
the GPU never processes more than one request concurrently.
"""

from __future__ import annotations

import logging

from fastapi import BackgroundTasks, FastAPI
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


@app.post("/generate/flux", response_model=GenerateResponse)
async def generate_flux(req: GenerateRequest) -> GenerateResponse:
    return await _generate("flux", req)


@app.post("/generate/model2", response_model=GenerateResponse)
async def generate_model2(req: GenerateRequest) -> GenerateResponse:
    return await _generate("model2", req)


async def _generate(model: str, req: GenerateRequest) -> GenerateResponse:
    logger.info("generate %s | prompt=%r | orient=%s | steps=%s", model, req.prompt, req.orientation, req.num_inference_steps)
    result = await pipelines.generate(
        model=model,
        prompt=req.prompt,
        orientation=req.orientation,
        steps=req.num_inference_steps,
        seed=req.seed,
    )
    return GenerateResponse(**result)


@app.get("/image/{filename}")
async def get_image(filename: str, background_tasks: BackgroundTasks):
    """Serve a previously generated image by filename (read-only, no GPU)."""
    from pathlib import Path

    path = Path(cfg.OUTPUT_DIR) / filename
    if not path.exists() or not path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="imagem nao encontrada")
    return FileResponse(str(path), media_type="image/png")
