"""
ids.py — Generación de IDs.

ULID: ordenables por tiempo, ideales para Firestore y tareas.
"""
from __future__ import annotations

import uuid

import ulid


def new_id() -> str:
    """Genera un ID ULID (string de 26 caracteres, ordenable)."""
    return str(ulid.new())


def new_uuid() -> str:
    """Genera un UUID v4 (para archivos y tareas)."""
    return str(uuid.uuid4())
