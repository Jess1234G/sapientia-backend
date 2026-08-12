"""
graph.py — Schemas de generación de gráficos (Dual View).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GraphRequest(BaseModel):
    """Solicitud de generación de gráfico."""

    description: str = Field(..., min_length=1)  # qué graficar
    code: str = ""                                # código Python opcional (producido por la IA)
    conversation_id: str = ""


class GraphArtifactOut(BaseModel):
    """Estado y artefactos de un gráfico."""

    artifact_id: str
    status: Literal["pending", "running", "completed", "failed"]
    png_url: str = ""   # vista 2D (Matplotlib)
    html_url: str = ""  # vista 3D interactiva (Plotly)
    error: str = ""


class GraphAccepted(BaseModel):
    """Respuesta 202 Accepted al encolar la tarea."""

    status: str = "accepted"
    task_id: str
    artifact_id: str
