"""
Pruebas de resolución de attachments en el historial (F6-C1).

Verifica que GET /history/conversations/{id} convierta los
attachment_ids de cada mensaje en metadata pública (sin exponer
storage_key ni user_id) y omita adjuntos inexistentes o ajenos.
"""

from __future__ import annotations

from app.core.security import get_current_user
from app.main import app
from app.services.firebase.firestore_service import (
    get_firestore_service,
)


class FakeFirestore:
    """Firestore falso con conversaciones y colección de adjuntos."""

    def __init__(self, conversations, attachments):
        self.conversations = conversations
        self.attachments = attachments

    async def get_user_conversation(self, conversation_id, user_id):
        conversation = self.conversations.get(conversation_id)

        if conversation is None:
            return None

        if conversation.get("user_id") != user_id:
            return None

        return conversation

    async def get_user_attachments(self, attachment_ids, user_id):
        result = []

        for attachment_id in attachment_ids:
            if not attachment_id:
                continue

            data = self.attachments.get(attachment_id)

            if data is not None and data.get("user_id") == user_id:
                result.append(data)

        return result


def _attachments():
    return {
        "att-1": {
            "attachment_id": "att-1",
            "user_id": "user-123",
            "filename": "foto.png",
            "content_type": "image/png",
            "size": 111,
            "storage_key": "attachments/user-123/foto.png",
        },
        "att-2": {
            "attachment_id": "att-2",
            "user_id": "user-123",
            "filename": "ejercicio.pdf",
            "content_type": "application/pdf",
            "size": 222,
            "storage_key": "attachments/user-123/ejercicio.pdf",
        },
        "att-other": {
            "attachment_id": "att-other",
            "user_id": "other-user",
            "filename": "secreto.txt",
            "content_type": "text/plain",
            "size": 333,
            "storage_key": "attachments/other-user/secreto.txt",
        },
    }


def _conversation(messages):
    return {
        "conv-1": {
            "conversation_id": "conv-1",
            "user_id": "user-123",
            "title": "Título",
            "messages": messages,
            "attachments": [],
            "status": "active",
            "is_pinned": False,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    }


def _override(firestore):
    app.dependency_overrides[get_firestore_service] = lambda: firestore
    app.dependency_overrides[get_current_user] = lambda: "user-123"


def _clear():
    app.dependency_overrides.clear()


# ============================================================
# A. MessageOut con attachments
# ============================================================

def test_message_out_supports_attachments():
    from app.api.v1.schemas.history import MessageOut

    message = MessageOut(
        role="user",
        content="hola",
        attachments=[
            {
                "attachment_id": "a",
                "filename": "f.txt",
                "content_type": "text/plain",
                "size": 10,
            }
        ],
    )

    assert message.attachments[0].attachment_id == "a"
    assert message.attachments[0].filename == "f.txt"
    assert message.attachments[0].content_type == "text/plain"
    assert message.attachments[0].size == 10


# ============================================================
# B. Mensaje sin attachments -> []
# ============================================================

def test_message_without_attachments_returns_empty_list(client):
    firestore = FakeFirestore(
        _conversation(
            [
                {
                    "role": "user",
                    "content": "hola",
                    "attachments": [],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        _attachments(),
    )
    _override(firestore)

    try:
        response = client.get("/api/v1/history/conversations/conv-1")

        assert response.status_code == 200

        data = response.json()

        assert data["messages"][0]["attachments"] == []

    finally:
        _clear()


# ============================================================
# C. Conversación antigua sin campo attachments -> []
# ============================================================

def test_old_message_without_attachments_field_returns_empty(client):
    firestore = FakeFirestore(
        _conversation(
            [
                {
                    "role": "user",
                    "content": "hola",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        _attachments(),
    )
    _override(firestore)

    try:
        response = client.get("/api/v1/history/conversations/conv-1")

        assert response.status_code == 200

        data = response.json()

        assert data["messages"][0]["attachments"] == []

    finally:
        _clear()


# ============================================================
# D. Attachment válido aparece en historial + K/L
# ============================================================

def test_valid_attachment_appears_in_history(client):
    firestore = FakeFirestore(
        _conversation(
            [
                {
                    "role": "user",
                    "content": "mira",
                    "attachments": ["att-1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        _attachments(),
    )
    _override(firestore)

    try:
        response = client.get("/api/v1/history/conversations/conv-1")

        assert response.status_code == 200

        data = response.json()

        attachments = data["messages"][0]["attachments"]

        assert len(attachments) == 1
        assert attachments[0]["attachment_id"] == "att-1"
        assert attachments[0]["filename"] == "foto.png"
        assert attachments[0]["content_type"] == "image/png"
        assert attachments[0]["size"] == 111

        assert "storage_key" not in attachments[0]
        assert "user_id" not in attachments[0]
        assert "created_at" not in attachments[0]

    finally:
        _clear()


# ============================================================
# E. Attachment inexistente se omite
# ============================================================

def test_missing_attachment_is_omitted(client):
    firestore = FakeFirestore(
        _conversation(
            [
                {
                    "role": "user",
                    "content": "x",
                    "attachments": ["att-1", "missing"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        _attachments(),
    )
    _override(firestore)

    try:
        response = client.get("/api/v1/history/conversations/conv-1")

        assert response.status_code == 200

        data = response.json()

        ids = [
            attachment["attachment_id"]
            for attachment in data["messages"][0]["attachments"]
        ]

        assert ids == ["att-1"]

    finally:
        _clear()


# ============================================================
# F. Attachment de otro usuario se omite
# ============================================================

def test_other_user_attachment_is_omitted(client):
    firestore = FakeFirestore(
        _conversation(
            [
                {
                    "role": "user",
                    "content": "x",
                    "attachments": ["att-1", "att-other"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        _attachments(),
    )
    _override(firestore)

    try:
        response = client.get("/api/v1/history/conversations/conv-1")

        assert response.status_code == 200

        data = response.json()

        ids = [
            attachment["attachment_id"]
            for attachment in data["messages"][0]["attachments"]
        ]

        assert ids == ["att-1"]

    finally:
        _clear()


# ============================================================
# G. Orden de attachment_ids preservado
# ============================================================

def test_attachment_order_is_preserved(client):
    firestore = FakeFirestore(
        _conversation(
            [
                {
                    "role": "user",
                    "content": "x",
                    "attachments": ["att-2", "att-1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        _attachments(),
    )
    _override(firestore)

    try:
        response = client.get("/api/v1/history/conversations/conv-1")

        assert response.status_code == 200

        data = response.json()

        ids = [
            attachment["attachment_id"]
            for attachment in data["messages"][0]["attachments"]
        ]

        assert ids == ["att-2", "att-1"]

    finally:
        _clear()
