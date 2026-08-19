"""Pydantic request/response schemas for the generation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Orientation = Literal["w", "t"]  # w = wide (16:9), t = vertical (9:16)
# flux/model2 = imagem; kokoro/qwen3tts = TTS (texto->audio); whisper_turbo = ASR (audio->texto)
ModelId = Literal["flux", "model2", "kokoro", "qwen3tts", "whisper_turbo"]


class GenerateRequest(BaseModel):
    """Request to generate from a prompt.

    The model is chosen via the ``model`` field. The endpoint returns the
    generated file directly:
      * image models (flux, model2)  -> PNG
      * TTS models (kokoro, qwen3tts) -> WAV
      * ASR model (whisper_turbo)     -> JSON (see /transcribe for file upload)

    Image fields (orientation, num_inference_steps, seed) are only used by
    image models; TTS fields (text, voice, language) are only used by TTS.
    """

    model: ModelId = Field(
        "flux",
        description=(
            "Modelo a usar: 'flux'/'model2' (imagem), 'kokoro'/'qwen3tts' (TTS), "
            "'whisper_turbo' (ASR)."
        ),
    )
    # --- imagem ---
    prompt: str | None = Field(
        None,
        min_length=1,
        max_length=2000,
        description="Texto do prompt (imagem e TTS).",
        examples=["Um lago sereno ao por do sol, luz volumetrica, fotografia 35mm"],
    )
    orientation: Orientation = Field(
        "w",
        description="w = wide 16:9 (1024x576), t = vertical 9:16 (576x1024).",
    )
    num_inference_steps: int = Field(
        4, ge=1, le=50, description="Quantidade de passos de inferencia (loops, imagem)."
    )
    seed: int | None = Field(
        None,
        description="Seed para reprodutibilidade (imagem). Se omitido, usa 42.",
    )
    # --- TTS ---
    text: str | None = Field(
        None,
        max_length=2000,
        description="Texto a sintetizar (TTS: kokoro/qwen3tts).",
    )
    voice: str | None = Field(
        None,
        description="Voz/estilo (TTS). kokoro: 'af_heart' etc; qwen3tts: 'Ryan'/'Aiden'.",
    )
    language: str | None = Field(
        None,
        description="Idioma (TTS qwen3tts: 'English'/'Portuguese'; ASR: 'portuguese').",
    )

    model_config = {"extra": "ignore"}


class GenerateResponse(BaseModel):
    model: str = Field(..., description="Identificador do modelo usado.")
    repo: str = Field(..., description="Repositorio HF do modelo.")
    task: str = Field(..., description="Tipo de saida: 'image' | 'tts' | 'asr'.")
    media_type: str = Field(..., description="MIME type do arquivo retornado.")
    prompt: str | None = None
    orientation: str | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    seed: int | None = None
    text: str | None = None
    voice: str | None = None
    language: str | None = None
    image_path: str = Field(..., description="Caminho local do arquivo gerado.")
    filename: str = Field(..., description="Nome do arquivo gerado.")


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
