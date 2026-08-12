"""
user.py — Modelo User.

Documento Firestore: users/{uid}.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class User(BaseModel):
    """Perfil del estudiante."""

    uid: str
    email: str = ""
    display_name: str = ""
    photo_url: str = ""
    carrera: str = ""
    semestre: int = Field(default=0, ge=0, le=12)
    created_at: str = ""
    updated_at: str = ""
