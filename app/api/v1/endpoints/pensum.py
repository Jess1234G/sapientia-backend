"""
pensum.py — GET /pensum: contexto curricular del usuario.

Devuelve el pensum contextual (asignaturas del semestre/carrera del
usuario) para mostrar en el chat y alimentar el RAG.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import get_current_user

router = APIRouter()


@router.get("")
async def get_pensum(
    # uid: str = Depends(get_current_user),
):
    """Devuelve el pensum del usuario según su carrera y semestre."""
    # TODO: leer user.carrera/semestre → rag_service.retrieve_pensum()
    return {"carrera": "", "semestre": 0, "asignaturas": []}
