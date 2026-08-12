"""
history.py — Schemas del historial de conversaciones.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class MessageOut(BaseModel):
    """Mensaje serializado para el historial."""

    role: str
    content: str
    latex: str | None = None
    created_at: str = ""


class ConversationSummary(BaseModel):
    """Resumen de una conversación para el listado."""

    conversation_id: str
    title: str
    updated_at: str = ""
    status: str = "active"


class ConversationDetail(ConversationSummary):
    """Detalle completo de una conversación."""

    messages: list[MessageOut] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
