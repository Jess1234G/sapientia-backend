"""
Pruebas unitarias de FirestoreService.

Estas pruebas usan un Firestore falso para no tocar
la base de datos real de Firebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import app.services.firebase.firestore_service as firestore_module
from app.services.firebase.firestore_service import FirestoreService


# ============================================================
# FIRESTORE FALSO
# ============================================================

@dataclass
class FakeSnapshot:
    data: dict[str, Any] | None

    @property
    def exists(self) -> bool:
        return self.data is not None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data or {})


@dataclass
class FakeDocumentReference:
    store: dict[str, dict[str, dict[str, Any]]]
    collection_name: str
    document_id: str

    @property
    def id(self) -> str:
        """Identificador del documento simulado."""
        return self.document_id

    def get(self) -> FakeSnapshot:
        collection = self.store.setdefault(
            self.collection_name,
            {},
        )

        data = collection.get(self.document_id)

        return FakeSnapshot(
            dict(data) if data is not None else None
        )

    def set(
        self,
        data: dict[str, Any],
        merge: bool = False,
    ) -> None:
        collection = self.store.setdefault(
            self.collection_name,
            {},
        )

        if merge and self.document_id in collection:
            collection[self.document_id].update(data)
        else:
            collection[self.document_id] = dict(data)

    def update(self, data: dict[str, Any]) -> None:
        collection = self.store.setdefault(
            self.collection_name,
            {},
        )

        if self.document_id not in collection:
            raise ValueError(
                f"Documento '{self.document_id}' no existe."
            )

        for key, value in data.items():
            if key == "messages" and isinstance(value, list):
                existing = collection[self.document_id].get(
                    "messages",
                    [],
                )

                collection[self.document_id]["messages"] = [
                    *existing,
                    *value,
                ]
            else:
                collection[self.document_id][key] = value


@dataclass
class FakeCollectionReference:
    store: dict[str, dict[str, dict[str, Any]]]
    collection_name: str
    counter: int = 0

    def document(
        self,
        document_id: str | None = None,
    ) -> FakeDocumentReference:
        if document_id is None:
            self.counter += 1
            document_id = f"fake-conversation-{self.counter}"

        return FakeDocumentReference(
            store=self.store,
            collection_name=self.collection_name,
            document_id=document_id,
        )


@dataclass
class FakeFirestoreClient:
    store: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )

    def collection(
        self,
        collection_name: str,
    ) -> FakeCollectionReference:
        return FakeCollectionReference(
            store=self.store,
            collection_name=collection_name,
        )


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def firestore_service(monkeypatch):
    """
    FirestoreService conectado a un Firestore completamente falso.

    Además, ArrayUnion se simplifica a una lista para que podamos
    comprobar el comportamiento de acumulación sin Firebase real.
    """

    monkeypatch.setattr(
        firestore_module.firestore_admin,
        "ArrayUnion",
        lambda values: values,
    )

    client = FakeFirestoreClient()

    service = FirestoreService(client)

    return service, client


# ============================================================
# CREATE CONVERSATION
# ============================================================

@pytest.mark.asyncio
async def test_create_conversation(
    firestore_service,
):
    """
    Crear una conversación debe generar un ID y guardar
    una conversación inicialmente vacía.
    """

    service, client = firestore_service

    conversation_id = await service.create_conversation(
        user_id="user-123",
        title="Primera conversación",
    )

    assert conversation_id.startswith(
        "fake-conversation-"
    )

    stored = client.store["conversations"][
        conversation_id
    ]

    assert stored["conversation_id"] == conversation_id
    assert stored["user_id"] == "user-123"
    assert stored["title"] == "Primera conversación"
    assert stored["messages"] == []
    assert stored["attachments"] == []
    assert stored["status"] == "active"
    assert "created_at" in stored
    assert "updated_at" in stored


# ============================================================
# ADD MESSAGE — ACUMULACIÓN
# ============================================================

@pytest.mark.asyncio
async def test_add_message_preserves_previous_messages(
    firestore_service,
):
    """
    add_message() debe conservar los mensajes anteriores
    y añadir el nuevo al historial.
    """

    service, client = firestore_service

    conversation_id = await service.create_conversation(
        user_id="user-123",
        title="Historial",
    )

    first_message = {
        "role": "user",
        "content": "Hola Sapientia",
    }

    second_message = {
        "role": "assistant",
        "content": "Hola. ¿En qué puedo ayudarte?",
    }

    await service.add_message(
        conversation_id,
        first_message,
    )

    await service.add_message(
        conversation_id,
        second_message,
    )

    stored = client.store["conversations"][
        conversation_id
    ]

    assert stored["messages"] == [
        first_message,
        second_message,
    ]


# ============================================================
# GET USER CONVERSATION
# ============================================================

@pytest.mark.asyncio
async def test_get_user_conversation_respects_owner(
    firestore_service,
):
    """
    Una conversación solo debe devolverse al usuario
    al que realmente pertenece.
    """

    service, client = firestore_service

    conversation_id = await service.create_conversation(
        user_id="owner-123",
        title="Conversación privada",
    )

    owner_result = (
        await service.get_user_conversation(
            conversation_id=conversation_id,
            user_id="owner-123",
        )
    )

    other_user_result = (
        await service.get_user_conversation(
            conversation_id=conversation_id,
            user_id="other-user",
        )
    )

    assert owner_result is not None
    assert owner_result["conversation_id"] == (
        conversation_id
    )
    assert owner_result["user_id"] == "owner-123"

    assert other_user_result is None


# ============================================================
# GET USER CONVERSATION — CONVERSACIÓN INEXISTENTE
# ============================================================

@pytest.mark.asyncio
async def test_get_user_conversation_missing_returns_none(
    firestore_service,
):
    """
    Una conversación inexistente debe devolver None.
    """

    service, _ = firestore_service

    result = await service.get_user_conversation(
        conversation_id="does-not-exist",
        user_id="user-123",
    )

    assert result is None


# ============================================================
# GRAPH ARTIFACTS
# ============================================================

@pytest.mark.asyncio
async def test_create_graph_artifact(
    firestore_service,
):
    """
    Crear un artefacto debe generar un ID y guardar
    los campos proporcionados.
    """

    service, client = firestore_service

    artifact_id = (
        await service.create_graph_artifact(
            user_id="user-123",
            conversation_id="conversation-123",
            task_id="task-123",
        )
    )

    assert artifact_id

    stored = client.store["graph_artifacts"][
        artifact_id
    ]

    assert stored["artifact_id"] == artifact_id
    assert stored["user_id"] == "user-123"
    assert stored["conversation_id"] == (
        "conversation-123"
    )
    assert stored["task_id"] == "task-123"
    assert stored["status"] == "pending"
    assert "created_at" in stored


@pytest.mark.asyncio
async def test_get_graph_artifact(
    firestore_service,
):
    """
    Un artefacto existente debe poder recuperarse por ID.
    """

    service, _ = firestore_service

    artifact_id = (
        await service.create_graph_artifact(
            user_id="user-123",
        )
    )

    result = await service.get_graph_artifact(
        artifact_id
    )

    assert result is not None
    assert result["artifact_id"] == artifact_id
    assert result["user_id"] == "user-123"
    assert result["status"] == "pending"


@pytest.mark.asyncio
async def test_get_graph_artifact_missing_returns_none(
    firestore_service,
):
    """
    Un artefacto inexistente debe devolver None.
    """

    service, _ = firestore_service

    result = await service.get_graph_artifact(
        "does-not-exist"
    )

    assert result is None


@pytest.mark.asyncio
async def test_update_graph_artifact(
    firestore_service,
):
    """
    La actualización debe modificar únicamente los campos
    indicados y conservar el resto del documento.
    """

    service, client = firestore_service

    artifact_id = (
        await service.create_graph_artifact(
            user_id="user-123",
            conversation_id="conversation-123",
            status="pending",
        )
    )

    await service.update_graph_artifact(
        artifact_id,
        status="completed",
        html_url="https://example.com/graph.html",
    )

    stored = client.store["graph_artifacts"][
        artifact_id
    ]

    assert stored["artifact_id"] == artifact_id
    assert stored["user_id"] == "user-123"
    assert stored["conversation_id"] == (
        "conversation-123"
    )
    assert stored["status"] == "completed"
    assert stored["html_url"] == (
        "https://example.com/graph.html"
    )
    assert "created_at" in stored


@pytest.mark.asyncio
async def test_update_graph_artifact_missing_raises_error(
    firestore_service,
):
    """
    Actualizar un artefacto inexistente debe fallar explícitamente.
    """

    service, _ = firestore_service

    with pytest.raises(ValueError):
        await service.update_graph_artifact(
            "does-not-exist",
            status="completed",
        )


@pytest.mark.asyncio
async def test_update_graph_artifact_without_fields_does_nothing(
    firestore_service,
):
    """
    Una actualización sin campos no debe modificar el documento.
    """

    service, client = firestore_service

    artifact_id = (
        await service.create_graph_artifact(
            user_id="user-123",
        )
    )

    before = dict(
        client.store["graph_artifacts"][
            artifact_id
        ]
    )

    await service.update_graph_artifact(
        artifact_id,
    )

    after = client.store["graph_artifacts"][
        artifact_id
    ]

    assert after == before