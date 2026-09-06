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
    is_pinned: bool = False


class ConversationUpdate(BaseModel):
    """Actualización parcial de una conversación (PATCH)."""

    title: str | None = None
    is_pinned: bool | None = None


class ConversationDetail(ConversationSummary):
    """Detalle completo de una conversación."""

    messages: list[MessageOut] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
