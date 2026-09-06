"""
history.py — Historial persistente de conversaciones.

GET /history/conversations
    Lista las conversaciones del usuario autenticado.

GET /history/conversations/{conversation_id}
    Devuelve una conversación únicamente si pertenece
    al usuario autenticado.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)
from pydantic import BaseModel

from app.api.v1.schemas.history import (
    ConversationDetail,
    ConversationUpdate,
)
from app.core.security import get_current_user
from app.services.firebase.firestore_service import (
    FirestoreService,
    get_firestore_service,
)

router = APIRouter()


# ============================================================
# SCHEMAS
# ============================================================

class ConversationSummary(BaseModel):
    """Resumen utilizado para mostrar una conversación en Sidebar."""

    conversation_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    is_pinned: bool = False


class ConversationListResponse(BaseModel):
    """Respuesta del listado de conversaciones."""

    items: list[ConversationSummary]


# ============================================================
# LISTADO
# ============================================================

@router.get(
    "/conversations",
    response_model=ConversationListResponse,
)
async def list_conversations(
    uid: str = Depends(get_current_user),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
):
    """
    Devuelve únicamente las conversaciones del usuario autenticado.

    FirestoreService ya aplica el filtro por user_id y ordena
    por updated_at descendente.
    """

    conversations = await firestore.list_conversations(uid)

    items = []

    for conversation in conversations:
        items.append(
            ConversationSummary(
                conversation_id=conversation.get(
                    "conversation_id",
                    "",
                ),
                title=conversation.get(
                    "title",
                    "Nueva conversación",
                ),
                status=conversation.get(
                    "status",
                    "active",
                ),
                created_at=conversation.get(
                    "created_at",
                    "",
                ),
                updated_at=conversation.get(
                    "updated_at",
                    "",
                ),
                is_pinned=conversation.get(
                    "is_pinned",
                    False,
                ),
            )
        )

    return ConversationListResponse(
        items=items,
    )


# ============================================================
# DETALLE
# ============================================================

@router.get(
    "/conversations/{conversation_id}",
)
async def get_conversation(
    conversation_id: str,
    uid: str = Depends(get_current_user),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
):
    """
    Devuelve la conversación completa.

    El usuario solo puede acceder a conversaciones
    pertenecientes a su propio UID.
    """

    conversation = await firestore.get_user_conversation(
        conversation_id=conversation_id,
        user_id=uid,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversación no encontrada.",
        )

    # Las conversaciones antiguas pueden no tener is_pinned.
    conversation.setdefault("is_pinned", False)

    return conversation


# ============================================================
# ACTUALIZACIÓN PARCIAL (RENOMBRAR / FIJAR)
# ============================================================

@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
)
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    uid: str = Depends(get_current_user),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
):
    """
    Renombra o fija/desfija una conversación del usuario.

    Solo el propietario puede modificar la conversación.
    """

    try:
        conversation = await firestore.update_conversation(
            conversation_id=conversation_id,
            user_id=uid,
            title=payload.title,
            is_pinned=payload.is_pinned,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversación no encontrada.",
        )

    # Las conversaciones antiguas pueden no tener is_pinned.
    conversation.setdefault("is_pinned", False)

    return conversation


# ============================================================
# ELIMINACIÓN
# ============================================================

@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: str,
    uid: str = Depends(get_current_user),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
):
    """
    Elimina una conversación del usuario.

    Solo el propietario puede eliminar la conversación.
    """

    deleted = await firestore.delete_conversation(
        conversation_id=conversation_id,
        user_id=uid,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversación no encontrada.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)