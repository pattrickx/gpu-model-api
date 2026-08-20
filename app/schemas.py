"""Pydantic request/response schemas for the generation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Orientation = Literal["w", "t"]  # w = wide (16:9), t = vertical (9:16)
# flux/model2 = imagem; kokoro/qwen3tts = TTS (texto->audio); whisper_turbo = ASR (audio->texto)
ModelId = Literal["flux", "model2", "kokoro", "qwen3tts", "whisper_turbo"]
# Modelos separados por tipo de midia (endpoints distintos).
ImageModelId = Literal[
    "flux",
    "model2",
    "sdxl_turbo",
    "sd_turbo",
    "sdxl_lightning",
    "flux2_klein",
    "flux2_klein_fp8",
    "flux2_klein_base",
]
# Modelos de edicao de imagem (texto+imagem -> imagem), endpoint /generate-image-edit
ImageEditModelId = Literal[
    "flux2_klein",
    "flux2_klein_fp8",
    "flux2_klein_base",
]


class ImageRequest(BaseModel):
    """Requisicao de geracao de IMAGEM (flux, model2) -> PNG."""

    model: ImageModelId = Field(
        "flux",
        description=(
            "Modelo de imagem: 'flux' (FLUX.1-schnell), 'model2' (SDXL base), "
            "'sdxl_turbo' e 'sd_turbo' (512x512 fixo, 1-4 steps), "
            "'sdxl_lightning' (SDXL destilado 4 steps), "
            "'flux2_klein' / 'flux2_klein_fp8' / 'flux2_klein_base' (FLUX.2-klein 4B). "
            "Para EDICAO de imagem (texto+imagem) use o endpoint /generate-image-edit "
            "com flux2_klein / flux2_klein_fp8 / flux2_klein_base."
        ),
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Texto do prompt (imagem).",
        examples=["Um lago sereno ao por do sol, luz volumetrica, fotografia 35mm"],
    )
    orientation: Orientation = Field(
        "w",
        description="w = wide 16:9 (1024x576), t = vertical 9:16 (576x1024).",
    )
    num_inference_steps: int = Field(
        4, ge=1, le=50, description="Quantidade de passos de inferencia (imagem)."
    )
    seed: int | None = Field(
        None,
        description="Seed para reprodutibilidade (imagem). Se omitido, usa 42.",
    )

    model_config = {"extra": "ignore"}


class AudioRequest(BaseModel):
    """Requisicao de sintese de AUDIO/TTS (kokoro, qwen3tts) -> WAV."""

    model: AudioModelId = Field(
        "kokoro",
        description="Modelo TTS: 'kokoro' (Kokoro-82M) ou 'qwen3tts' (Qwen3-TTS).",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Texto a sintetizar (TTS).",
    )
    voice: str | None = Field(
        None,
        description="Voz/estilo (TTS). kokoro: 'af_heart' etc; qwen3tts: 'Ryan'/'Aiden'.",
    )
    language: str | None = Field(
        None,
        description="Idioma do TTS (qwen3tts: 'English'/'Portuguese'; kokoro detecta).",
    )

    model_config = {"extra": "ignore"}


class ImageResponse(BaseModel):
    """Resposta de /generate-image (PNG)."""

    model: str = Field(..., description="Identificador do modelo de imagem usado.")
    repo: str = Field(..., description="Repositorio HF do modelo.")
    task: str = Field("image", description="Tipo de saida: 'image'.")
    media_type: str = Field("image/png", description="MIME type do arquivo retornado.")
    prompt: str | None = None
    orientation: str | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    seed: int | None = None
    image_path: str = Field(..., description="Caminho local do arquivo gerado.")
    filename: str = Field(..., description="Nome do arquivo gerado.")


class AudioResponse(BaseModel):
    """Resposta de /generate-audio (WAV TTS)."""

    model: str = Field(..., description="Identificador do modelo TTS usado.")
    repo: str = Field(..., description="Repositorio HF do modelo.")
    task: str = Field("tts", description="Tipo de saida: 'tts'.")
    media_type: str = Field("audio/wav", description="MIME type do arquivo retornado.")
    text: str | None = None
    voice: str | None = None
    language: str | None = None
    image_path: str = Field(..., description="Caminho local do arquivo gerado (WAV).")
    filename: str = Field(..., description="Nome do arquivo gerado.")


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
