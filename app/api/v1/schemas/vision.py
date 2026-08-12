"""
vision.py — Schemas del análisis de imágenes.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class VisionAnalysisRequest(BaseModel):
    """Metadatos de una petición de análisis (opcional)."""

    conversation_id: str = ""


class VisionAnalysisResult(BaseModel):
    """Resultado del modelo de visión (OCR + LaTeX)."""

    texto: str = ""
    formulas: list[str] = Field(default_factory=list)
    tipo: str = "otro"  # matemáticas | física | química | otro
    descripcion: str = ""
