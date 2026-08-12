"""
chat.py — Schemas del chat con streaming SSE.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    """Petición de mensaje al tutor."""

    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str = ""
    vision_text: str = ""       # texto extraído por /vision/analyze (opcional)
    attachment_ids: list[str] = Field(default_factory=list)


class ChatChunk(BaseModel):
    """Chunk de streaming SSE: `data: {json}`."""

    type: str  # "reasoning" | "answer" | "graph_request" | "done" | "error"
    content: str = ""


class ChatResponse(BaseModel):
    """Respuesta final no-stream (cuando el cliente no pide SSE)."""

    conversation_id: str
    answer: str
    graph_requested: bool = False
