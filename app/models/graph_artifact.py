"""
graph_artifact.py — Modelo GraphArtifact.

Documento Firestore: graph_artifacts/{artifact_id}.
Generado por la tarea Celery `generate_graph_task`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class GraphArtifact(BaseModel):
    """Artefacto de gráfico Dual View (2D estático + 3D interactivo)."""

    artifact_id: str = ""
    user_id: str
    conversation_id: str = ""
    task_id: str = ""
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    code: str = ""
    png_url: str = ""   # vista 2D (Matplotlib)
    html_url: str = ""  # vista 3D interactiva (Plotly, para WebView)
    error: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
