"""
auth.py — Schemas de autenticación.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class GoogleTokenRequest(BaseModel):
    """ID token de Google/Firebase recibido del cliente."""

    id_token: str = Field(..., min_length=20)


class SessionResponse(BaseModel):
    """Respuesta de login con sesión JWT propia."""

    access_token: str
    token_type: str = "bearer"
    user: dict


class UserOut(BaseModel):
    """Perfil público del usuario."""

    uid: str
    email: str
    display_name: str = ""
    photo_url: str = ""
    carrera: str = ""
    semestre: int = 0
