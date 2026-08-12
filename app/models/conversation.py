"""
conversation.py — Modelo Conversation.

Documento Firestore: conversations/{conversation_id}.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Mensaje individual de una conversación."""

    role: Literal["user", "assistant", "system"] = "user"
    content: str
    latex: str | None = None
    attachments: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Conversation(BaseModel):
    """Conversación del estudiante con el tutor."""

    conversation_id: str = ""
    user_id: str
    title: str = "Nueva conversación"
    messages: list[Message] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    status: Literal["active", "archived"] = "active"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
