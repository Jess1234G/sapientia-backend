"""Schemas de attachments (subida de archivos)."""
from __future__ import annotations

from pydantic import BaseModel


class AttachmentOut(BaseModel):
    """Respuesta de la subida de un attachment."""

    attachment_id: str
    filename: str
    content_type: str
    size: int


class AttachmentUrlOut(AttachmentOut):
    """Attachment con URL temporal de descarga para edición."""

    url: str
