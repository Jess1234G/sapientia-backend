"""
Pruebas del historial modular de conversaciones.
"""

from __future__ import annotations

from app.core.security import get_current_user
from app.main import app
from app.services.firebase.firestore_service import (
    get_firestore_service,
)


class FakeHistoryFirestoreService:
    """Firestore falso para probar el router de historial."""

    def __init__(self):
        self.conversations = {
            "conversation-1": {
                "conversation_id": "conversation-1",
                "user_id": "user-123",
                "title": "Derivadas",
                "messages": [
                    {
                        "role": "user",
                        "content": "¿Qué es una derivada?",
                    },
                    {
                        "role": "assistant",
                        "content": "Una derivada mide una tasa de cambio.",
                    },
                ],
                "attachments": [],
                "status": "active",
                "created_at": "2026-08-21T10:00:00+00:00",
                "updated_at": "2026-08-21T10:05:00+00:00",
            },
            "conversation-2": {
                "conversation_id": "conversation-2",
                "user_id": "user-123",
                "title": "Integrales",
                "messages": [],
                "attachments": [],
                "status": "active",
                "created_at": "2026-08-21T09:00:00+00:00",
                "updated_at": "2026-08-21T09:30:00+00:00",
            },
            "conversation-other": {
                "conversation_id": "conversation-other",
                "user_id": "other-user",
                "title": "Privada",
                "messages": [],
                "attachments": [],
                "status": "active",
                "created_at": "2026-08-21T08:00:00+00:00",
                "updated_at": "2026-08-21T08:30:00+00:00",
            },
        }

    async def list_conversations(
        self,
        user_id: str,
    ):
        return [
            conversation
            for conversation in self.conversations.values()
            if conversation["user_id"] == user_id
        ]

    async def get_user_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ):
        conversation = self.conversations.get(
            conversation_id
        )

        if conversation is None:
            return None

        if conversation["user_id"] != user_id:
            return None

        return conversation

    async def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: str | None = None,
        is_pinned: bool | None = None,
    ):
        conversation = self.conversations.get(conversation_id)

        if conversation is None:
            return None

        if conversation["user_id"] != user_id:
            return None

        if title is not None:
            cleaned_title = title.strip()

            if not cleaned_title:
                raise ValueError(
                    "El título no puede estar vacío."
                )

            conversation["title"] = cleaned_title[:80]

        if is_pinned is not None:
            conversation["is_pinned"] = bool(is_pinned)

        return conversation

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ):
        conversation = self.conversations.get(conversation_id)

        if conversation is None:
            return False

        if conversation["user_id"] != user_id:
            return False

        del self.conversations[conversation_id]
        return True


def override_dependencies(firestore):
    """
    Configura las dependencias FastAPI para las pruebas.
    """

    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore

    app.dependency_overrides[
        get_current_user
    ] = lambda: "user-123"


def clear_dependencies():
    """Limpia los overrides globales de FastAPI."""

    app.dependency_overrides.clear()


# ============================================================
# LISTADO
# ============================================================

def test_list_conversations_returns_user_history(
    client,
):
    """El usuario recibe únicamente sus conversaciones."""

    firestore = FakeHistoryFirestoreService()

    override_dependencies(firestore)

    try:
        response = client.get(
            "/api/v1/history/conversations"
        )

        assert response.status_code == 200

        data = response.json()

        assert "items" in data
        assert len(data["items"]) == 2

        conversation_ids = {
            item["conversation_id"]
            for item in data["items"]
        }

        assert conversation_ids == {
            "conversation-1",
            "conversation-2",
        }

    finally:
        clear_dependencies()


def test_list_conversations_does_not_expose_other_users(
    client,
):
    """Nunca deben aparecer chats pertenecientes a otro usuario."""

    firestore = FakeHistoryFirestoreService()

    override_dependencies(firestore)

    try:
        response = client.get(
            "/api/v1/history/conversations"
        )

        assert response.status_code == 200

        data = response.json()

        titles = [
            item["title"]
            for item in data["items"]
        ]

        assert "Privada" not in titles

    finally:
        clear_dependencies()


# ============================================================
# DETALLE
# ============================================================

def test_get_conversation_returns_full_conversation(
    client,
):
    """El usuario puede abrir uno de sus chats."""

    firestore = FakeHistoryFirestoreService()

    override_dependencies(firestore)

    try:
        response = client.get(
            "/api/v1/history/conversations/conversation-1"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["conversation_id"] == (
            "conversation-1"
        )

        assert data["title"] == "Derivadas"

        assert len(data["messages"]) == 2

        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    finally:
        clear_dependencies()


def test_get_other_user_conversation_returns_404(
    client,
):
    """Un usuario no puede abrir una conversación ajena."""

    firestore = FakeHistoryFirestoreService()

    override_dependencies(firestore)

    try:
        response = client.get(
            "/api/v1/history/conversations/conversation-other"
        )

        assert response.status_code == 404

        data = response.json()

        assert data["detail"] == (
            "Conversación no encontrada."
        )

    finally:
        clear_dependencies()


def test_get_missing_conversation_returns_404(
    client,
):
    """Una conversación inexistente devuelve 404."""

    firestore = FakeHistoryFirestoreService()

    override_dependencies(firestore)

    try:
        response = client.get(
            "/api/v1/history/conversations/does-not-exist"
        )

        assert response.status_code == 404

        data = response.json()

        assert data["detail"] == (
            "Conversación no encontrada."
        )

    finally:
        clear_dependencies()


# ============================================================
# PATCH — RENOMBRAR / FIJAR
# ============================================================

def test_patch_conversation_renames_title(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.patch(
            "/api/v1/history/conversations/conversation-1",
            json={"title": "Nuevo título"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["title"] == "Nuevo título"
        assert data["conversation_id"] == "conversation-1"
        # Conversación antigua sin is_pinned → False.
        assert data["is_pinned"] is False

    finally:
        clear_dependencies()


def test_patch_conversation_pins(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.patch(
            "/api/v1/history/conversations/conversation-1",
            json={"is_pinned": True},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["is_pinned"] is True
        assert data["title"] == "Derivadas"

    finally:
        clear_dependencies()


def test_patch_conversation_title_and_pin(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.patch(
            "/api/v1/history/conversations/conversation-2",
            json={
                "title": "Integrales avanzadas",
                "is_pinned": False,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["title"] == "Integrales avanzadas"
        assert data["is_pinned"] is False

    finally:
        clear_dependencies()


def test_patch_missing_conversation_returns_404(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.patch(
            "/api/v1/history/conversations/does-not-exist",
            json={"title": "Título"},
        )

        assert response.status_code == 404

        data = response.json()

        assert data["detail"] == "Conversación no encontrada."

    finally:
        clear_dependencies()


def test_patch_other_user_conversation_returns_404(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.patch(
            "/api/v1/history/conversations/conversation-other",
            json={"title": "Hackeado"},
        )

        assert response.status_code == 404

    finally:
        clear_dependencies()


# ============================================================
# DELETE
# ============================================================

def test_delete_conversation_returns_204(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.delete(
            "/api/v1/history/conversations/conversation-1"
        )

        assert response.status_code == 204
        assert response.content == b""

        assert "conversation-1" not in firestore.conversations

    finally:
        clear_dependencies()


def test_delete_missing_conversation_returns_404(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.delete(
            "/api/v1/history/conversations/does-not-exist"
        )

        assert response.status_code == 404

        data = response.json()

        assert data["detail"] == "Conversación no encontrada."

    finally:
        clear_dependencies()


def test_delete_other_user_conversation_returns_404(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.delete(
            "/api/v1/history/conversations/conversation-other"
        )

        assert response.status_code == 404

        assert "conversation-other" in firestore.conversations

    finally:
        clear_dependencies()


# ============================================================
# COMPATIBILIDAD CON CONVERSACIONES ANTIGUAS
# ============================================================

def test_list_old_conversations_return_is_pinned_false(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.get(
            "/api/v1/history/conversations"
        )

        assert response.status_code == 200

        data = response.json()

        for item in data["items"]:
            assert item["is_pinned"] is False

    finally:
        clear_dependencies()


def test_get_old_conversation_returns_is_pinned_false(
    client,
):
    firestore = FakeHistoryFirestoreService()
    override_dependencies(firestore)

    try:
        response = client.get(
            "/api/v1/history/conversations/conversation-1"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["is_pinned"] is False

    finally:
        clear_dependencies()
