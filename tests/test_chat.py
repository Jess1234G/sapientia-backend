"""
Pruebas del endpoint modular /api/v1/chat/message.
"""

from __future__ import annotations

import json

from app.core.security import get_current_user
from app.main import app
from app.services.deepseek.reasoning_service import (
    get_reasoning_service,
)
from app.services.firebase.firestore_service import (
    get_firestore_service,
)


class FakeChatFirestore:
    """Firestore falso para probar el chat."""

    def __init__(self):
        self.conversations = {}
        self.counter = 0

        self.users = {
            "user-123": {
                "uid": "user-123",
                "display_name": "Usuario de Prueba",
                "email": "test@example.com",
                "carrera": "",
                "semestre": 0,
            }
        }

    async def get_user(self, user_id: str):
        return self.users.get(user_id)

    async def get_user_memory(self, user_id: str):
        user = self.users.get(user_id)

        if not user:
            return {
                "preferences": [],
                "facts": [],
                "goals": [],
                "projects": [],
            }

        memory = user.get("memory", {})

        return {
            "preferences": list(
                memory.get("preferences", [])
            ),
            "facts": list(
                memory.get("facts", [])
            ),
            "goals": list(
                memory.get("goals", [])
            ),
            "projects": list(
                memory.get("projects", [])
            ),
        }

    async def list_conversations(self, user_id: str):
        return [
            conversation
            for conversation in self.conversations.values()
            if conversation["user_id"] == user_id
        ]

    async def create_conversation(
        self,
        user_id: str,
        title: str,
    ):
        self.counter += 1

        conversation_id = (
            f"conversation-{self.counter}"
        )

        self.conversations[conversation_id] = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": title,
            "messages": [],
            "attachments": [],
            "status": "active",
            "created_at": "2026-08-21T00:00:00+00:00",
            "updated_at": "2026-08-21T00:00:00+00:00",
        }

        return conversation_id

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

    async def add_message(
        self,
        conversation_id: str,
        message: dict,
    ):
        self.conversations[
            conversation_id
        ]["messages"].append(message)

    async def create_graph_artifact(
        self,
        **fields,
    ):
        self.counter += 1
        artifact_id = (
            f"artifact-{self.counter}"
        )

        self.graph_artifacts = getattr(
            self,
            "graph_artifacts",
            {},
        )

        self.graph_artifacts[
            artifact_id
        ] = {
            "artifact_id": artifact_id,
            **fields,
        }

        return artifact_id

    async def update_graph_artifact(
        self,
        artifact_id: str,
        **fields,
    ):
        self.graph_artifacts[
            artifact_id
        ].update(fields)


class FakeReasoningService:
    """ReasoningService falso con streaming determinista."""

    def __init__(self):
        self.received_message = None
        self.received_vision = None
        self.received_memory = None
        self.graph_code = None

    async def stream_reasoning(
        self,
        user_message: str,
        rag_context: str = "",
        vision_text: str = "",
        memory_context: str = "",
        metrics=None,
    ):
        self.received_message = user_message
        self.received_vision = vision_text
        self.received_memory = memory_context
        self.received_metrics = metrics

        yield {
            "type": "delta",
            "content": "Hola. ",
        }

        yield {
            "type": "delta",
            "content": "Soy Sapientia.",
        }

    def extract_graph_code(
        self,
        answer: str,
    ):
        return self.graph_code


def override_dependencies(
    firestore,
    reasoning,
):
    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore

    app.dependency_overrides[
        get_reasoning_service
    ] = lambda: reasoning

    app.dependency_overrides[
        get_current_user
    ] = lambda: "user-123"


def clear_dependencies():
    app.dependency_overrides.clear()


# ============================================================
# CREAR NUEVA CONVERSACIÓN
# ============================================================

def test_chat_creates_conversation_and_saves_messages(
    client,
):
    firestore = FakeChatFirestore()
    reasoning = FakeReasoningService()

    override_dependencies(
        firestore,
        reasoning,
    )

    try:
        response = client.post(
            "/api/v1/chat/message",
            json={
                "message": "Hola Sapientia.",
            },
        )

        assert response.status_code == 200

        body = response.text

        assert "Hola. " in body
        assert "Soy Sapientia." in body
        assert '"type": "done"' in body

        assert len(
            firestore.conversations
        ) == 1

        conversation = next(
            iter(
                firestore.conversations.values()
            )
        )

        assert conversation["user_id"] == "user-123"

        assert len(
            conversation["messages"]
        ) == 2

        assert conversation["messages"][0][
            "role"
        ] == "user"

        assert conversation["messages"][0][
            "content"
        ] == "Hola Sapientia."

        assert conversation["messages"][1][
            "role"
        ] == "assistant"

        assert conversation["messages"][1][
            "content"
        ] == "Hola. Soy Sapientia."

    finally:
        clear_dependencies()


# ============================================================
# CONTINUAR CONVERSACIÓN EXISTENTE
# ============================================================

def test_chat_continues_existing_conversation(
    client,
):
    firestore = FakeChatFirestore()

    firestore.conversations[
        "conversation-existing"
    ] = {
        "conversation_id": "conversation-existing",
        "user_id": "user-123",
        "title": "Conversación existente",
        "messages": [
            {
                "role": "user",
                "content": "Mensaje anterior",
            },
        ],
        "attachments": [],
        "status": "active",
        "created_at": "2026-08-21T00:00:00+00:00",
        "updated_at": "2026-08-21T00:00:00+00:00",
    }

    reasoning = FakeReasoningService()

    override_dependencies(
        firestore,
        reasoning,
    )

    try:
        response = client.post(
            "/api/v1/chat/message",
            json={
                "message": "Segundo mensaje",
                "conversation_id": (
                    "conversation-existing"
                ),
            },
        )

        assert response.status_code == 200

        conversation = firestore.conversations[
            "conversation-existing"
        ]

        assert len(
            conversation["messages"]
        ) == 3

    finally:
        clear_dependencies()


# ============================================================
# CONVERSACIÓN AJENA
# ============================================================

def test_chat_rejects_other_users_conversation(
    client,
):
    firestore = FakeChatFirestore()

    firestore.conversations[
        "conversation-private"
    ] = {
        "conversation_id": "conversation-private",
        "user_id": "another-user",
        "title": "Privada",
        "messages": [],
        "attachments": [],
        "status": "active",
        "created_at": "2026-08-21T00:00:00+00:00",
        "updated_at": "2026-08-21T00:00:00+00:00",
    }

    reasoning = FakeReasoningService()

    override_dependencies(
        firestore,
        reasoning,
    )

    try:
        response = client.post(
            "/api/v1/chat/message",
            json={
                "message": "No debo entrar.",
                "conversation_id": (
                    "conversation-private"
                ),
            },
        )

        assert response.status_code == 404

        assert response.json()["detail"] == (
            "Conversación no encontrada."
        )

    finally:
        clear_dependencies()


# ============================================================
# VISION TEXT
# ============================================================

def test_chat_passes_vision_text_to_reasoning(
    client,
):
    firestore = FakeChatFirestore()
    reasoning = FakeReasoningService()

    override_dependencies(
        firestore,
        reasoning,
    )

    try:
        response = client.post(
            "/api/v1/chat/message",
            json={
                "message": "Resuelve el ejercicio.",
                "vision_text": (
                    "x^2 + 2x + 1 = 0"
                ),
            },
        )

        assert response.status_code == 200

        assert reasoning.received_message == (
            "Resuelve el ejercicio."
        )

        assert reasoning.received_vision == (
            "x^2 + 2x + 1 = 0"
        )

    finally:
        clear_dependencies()


# ============================================================
# MEMORIA
# ============================================================

def test_chat_passes_memory_to_reasoning(
    client,
):
    firestore = FakeChatFirestore()

    firestore.conversations[
        "previous-chat"
    ] = {
        "conversation_id": "previous-chat",
        "user_id": "user-123",
        "title": "Conversación anterior",
        "messages": [
            {
                "role": "user",
                "content": "Estoy estudiando cálculo.",
            },
            {
                "role": "assistant",
                "content": "Podemos trabajar derivadas e integrales.",
            },
        ],
        "attachments": [],
        "status": "active",
        "created_at": "2026-08-20T00:00:00+00:00",
        "updated_at": "2026-08-20T01:00:00+00:00",
    }

    reasoning = FakeReasoningService()

    override_dependencies(
        firestore,
        reasoning,
    )

    try:
        response = client.post(
            "/api/v1/chat/message",
            json={
                "message": "Ahora ayúdame con una integral.",
            },
        )

        assert response.status_code == 200

        assert (
            reasoning.received_memory
            is not None
        )

        assert (
            "Conversación anterior"
            in reasoning.received_memory
        )

        assert (
            "Estoy estudiando cálculo."
            in reasoning.received_memory
        )

    finally:
        clear_dependencies()


def test_chat_memory_excludes_current_conversation(
    client,
):
    firestore = FakeChatFirestore()

    firestore.conversations[
        "conversation-existing"
    ] = {
        "conversation_id": "conversation-existing",
        "user_id": "user-123",
        "title": "Chat actual",
        "messages": [
            {
                "role": "user",
                "content": "Este mensaje pertenece al chat actual.",
            }
        ],
        "attachments": [],
        "status": "active",
        "created_at": "2026-08-21T00:00:00+00:00",
        "updated_at": "2026-08-21T00:00:00+00:00",
    }

    reasoning = FakeReasoningService()

    override_dependencies(
        firestore,
        reasoning,
    )

    try:
        response = client.post(
            "/api/v1/chat/message",
            json={
                "message": "Continúa esta conversación.",
                "conversation_id": "conversation-existing",
            },
        )

        assert response.status_code == 200

        assert reasoning.received_memory is not None

        assert (
            "Este mensaje pertenece al chat actual."
            not in reasoning.received_memory
        )

    finally:
        clear_dependencies()


# ============================================================
# INTEGRACIÓN CHAT → GRÁFICO
# ============================================================

def test_chat_emits_graph_created_when_code_present(
    client,
    monkeypatch,
):
    import app.services.graphs.graph_service as graph_service_module

    class FakeTask:
        id = "task-999"

    class FakeGenerateGraphTask:
        @classmethod
        def delay(cls, artifact_id, code):
            return FakeTask()

    monkeypatch.setattr(
        graph_service_module,
        "generate_graph_task",
        FakeGenerateGraphTask,
    )

    firestore = FakeChatFirestore()
    reasoning = FakeReasoningService()
    reasoning.graph_code = (
        "import plotly.graph_objects as go\n"
        "fig = go.Figure()\n"
        "fig.write_html('figura_3d.html')"
    )

    override_dependencies(
        firestore,
        reasoning,
    )

    try:
        response = client.post(
            "/api/v1/chat/message",
            json={
                "message": "Grafica la función.",
            },
        )

        assert response.status_code == 200

        body = response.text

        assert '"type": "graph_request"' in body
        assert '"type": "graph_created"' in body
        assert "artifact-2" in body
        assert "task-999" in body

        assert (
            "artifact-2"
            in firestore.graph_artifacts
        )

        artifact = firestore.graph_artifacts[
            "artifact-2"
        ]

        assert artifact["status"] == "pending"
        assert artifact["task_id"] == "task-999"
        assert artifact["code"] == reasoning.graph_code

    finally:
        clear_dependencies()