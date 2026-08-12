"""
history.py — GET listado y detalle de conversaciones + artefactos.

Persistencia en Firestore para que el usuario pueda volver a ver
sus conversaciones y los gráficos generados.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import get_current_user

router = APIRouter()


@router.get("/conversations")
async def list_conversations(
    # uid: str = Depends(get_current_user),
):
    """Lista las conversaciones del usuario (resumen, ordenadas por fecha)."""
    # TODO: query Firestore conversations WHERE user_id = uid ORDER BY updated_at DESC
    return {"items": []}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    # uid: str = Depends(get_current_user),
):
    """Detalle de una conversación con mensajes y artefactos asociados."""
    # TODO: leer conversation + graph_artifacts asociados
    return {"conversation_id": conversation_id, "messages": []}
