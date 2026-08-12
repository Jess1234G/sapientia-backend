"""
graphs.py — POST /graphs (encola generación) y GET /graphs/{id}.

Dual View: el código Python generado por DeepSeek R1 se ejecuta en el
sandbox E2B y produce:
  - `figura_2d.png` (Matplotlib, vista estática)
  - `figura_3d.html` (Plotly, vista interactiva para WebView)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import get_current_user

router = APIRouter()


@router.post("", status_code=202)
async def create_graph(
    # uid: str = Depends(get_current_user),
):
    """
    Solicita la generación de un gráfico.

    Recibe el código Python producido por la IA (o una descripción).
    Encola `generate_graph_task` en Celery y devuelve 202 + task_id.
    """
    # TODO: validar código → enqueue generate_graph_task → persistir artifact
    return {"status": "accepted", "task_id": "pending"}


@router.get("/{graph_id}")
async def get_graph(
    graph_id: str,
    # uid: str = Depends(get_current_user),
):
    """
    Consulta el estado y artefactos de un gráfico.

    Cuando termina devuelve: `png_url` (2D) y `html_url` (3D).
    """
    # TODO: leer graph_artifact de Firestore por graph_id + uid
    return {"graph_id": graph_id, "status": "pending"}
