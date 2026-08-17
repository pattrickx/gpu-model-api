"""Pydantic request/response schemas for the generation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Orientation = Literal["w", "t"]  # w = wide (16:9), t = vertical (9:16)
ModelId = Literal["flux", "model2"]  # flux = FLUX.1-schnell, model2 = SDXL base


class GenerateRequest(BaseModel):
    """Request to generate one image from a prompt.

    The model is chosen via the ``model`` field ("flux" or "model2").
    The endpoint returns the generated image file directly.
    """

    model: ModelId = Field(
        "flux",
        description="Modelo a usar: 'flux' (FLUX.1-schnell) ou 'model2' (SDXL base 1.0).",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Texto do prompt (em Portugues ou Ingles).",
        examples=["Um lago sereno ao por do sol, luz volumetrica, fotografia 35mm"],
    )
    orientation: Orientation = Field(
        "w",
        description="w = wide 16:9 (1024x576), t = vertical 9:16 (576x1024).",
    )
    num_inference_steps: int = Field(
        4, ge=1, le=50, description="Quantidade de passos de inferencia (loops)."
    )
    seed: int | None = Field(
        None,
        description="Seed para reprodutibilidade. Se omitido, usa 42 por padrao.",
    )


class GenerateResponse(BaseModel):
    model: str = Field(..., description="Identificador do modelo usado ('flux'|'model2').")
    repo: str = Field(..., description="Repositorio HF do modelo.")
    prompt: str
    orientation: str
    width: int
    height: int
    num_inference_steps: int
    seed: int
    image_path: str = Field(..., description="Caminho local do arquivo gerado.")
    filename: str = Field(..., description="Nome do arquivo gerado.")


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
