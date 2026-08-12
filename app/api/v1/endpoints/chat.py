"""
chat.py — POST /chat/message con streaming SSE.

Orquesta: contexto RAG (pensum) → DeepSeek R1 → respuesta en streaming.
Implementación real en `services/deepseek/reasoning_service.py`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import get_current_user

router = APIRouter()


@router.post("/message")
async def send_message(
    # uid: str = Depends(get_current_user),
    # payload: ChatMessageRequest,
):
    """
    Envía un mensaje y responde con Server-Sent Events (SSE).

    El body incluye: mensaje, id de conversación opcional y adjuntos
    (resultados de /vision/analyze). Los chunks SSE tienen el formato:
    `data: {"type": "reasoning"|"answer"|"graph_request"|"done", "content": ...}`
    """
    # TODO: implementar streaming con StreamResponse + media_type="text/event-stream"
    return {"detail": "Endpoint pendiente de implementación (streaming SSE)"}
