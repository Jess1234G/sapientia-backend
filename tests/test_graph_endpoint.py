from __future__ import annotations

from fastapi.testclient import TestClient

import app.services.graphs.graph_service as graph_service_module
from app.main import app


class FakeDocument:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


class FakeFirestore:
    def __init__(self):
        self.created = []
        self.updated = []

        self.artifacts = {
            "artifact-123": {
                "artifact_id": "artifact-123",
                "user_id": "user-123",
                "conversation_id": "chat-123",
                "status": "completed",
                "task_id": "task-123",
                "html_url": "https://gcs.test/graph.html",
                "png_url": "",
                "error": "",
            },
            "private-123": {
                "artifact_id": "private-123",
                "user_id": "other-user",
                "status": "completed",
                "task_id": "task-private",
                "html_url": "https://gcs.test/private.html",
                "png_url": "",
                "error": "",
            },
        }

    async def create_graph_artifact(self, **fields):
        artifact_id = "new-artifact"
        self.created.append(
            {
                "artifact_id": artifact_id,
                **fields,
            }
        )
        self.artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            **fields,
        }
        return artifact_id

    async def update_graph_artifact(
        self,
        artifact_id: str,
        **fields,
    ):
        self.updated.append(
            (
                artifact_id,
                fields,
            )
        )
        self.artifacts[artifact_id].update(fields)

    async def get_graph_artifact(
        self,
        artifact_id: str,
    ):
        artifact = self.artifacts.get(artifact_id)
        return (
            dict(artifact)
            if artifact is not None
            else None
        )


class FakeTask:
    id = "task-456"


class FakeGenerateGraphTask:
    calls = []

    @classmethod
    def delay(cls, artifact_id, code):
        cls.calls.append(
            (
                artifact_id,
                code,
            )
        )
        return FakeTask()


def fake_current_user():
    return "user-123"


def override_dependencies(
    firestore: FakeFirestore,
):
    from app.api.v1.endpoints.graphs import (
        get_current_user,
        get_firestore_service,
    )

    app.dependency_overrides[
        get_current_user
    ] = fake_current_user

    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore


def clear_dependencies():
    app.dependency_overrides.clear()


def test_create_graph_requires_code(monkeypatch):
    firestore = FakeFirestore()

    override_dependencies(
        firestore
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/graphs",
                json={
                    "description": "Parábola",
                    "code": "",
                    "conversation_id": "chat-123",
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "El código Python del gráfico "
            "es obligatorio."
        )

    finally:
        clear_dependencies()


def test_create_graph_enqueues_task(
    monkeypatch,
):
    firestore = FakeFirestore()

    FakeGenerateGraphTask.calls.clear()

    override_dependencies(
        firestore
    )

    monkeypatch.setattr(
        graph_service_module,
        "generate_graph_task",
        FakeGenerateGraphTask,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/graphs",
                json={
                    "description": "Gráfico de prueba",
                    "code": "print('graph')",
                    "conversation_id": "chat-123",
                },
            )

        assert response.status_code == 202

        data = response.json()

        assert data["status"] == "accepted"
        assert data["artifact_id"] == (
            "new-artifact"
        )
        assert data["task_id"] == (
            "task-456"
        )

        assert FakeGenerateGraphTask.calls == [
            (
                "new-artifact",
                "print('graph')",
            )
        ]

        assert firestore.updated == [
            (
                "new-artifact",
                {
                    "task_id": "task-456"
                },
            )
        ]

    finally:
        clear_dependencies()


def test_get_graph_returns_owned_artifact():
    firestore = FakeFirestore()

    override_dependencies(
        firestore
    )

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/graphs/artifact-123"
            )

        assert response.status_code == 200

        data = response.json()

        assert data["artifact_id"] == (
            "artifact-123"
        )
        assert data["status"] == (
            "completed"
        )
        assert data["html_url"] == (
            "https://gcs.test/graph.html"
        )
        assert data["png_url"] == ""

    finally:
        clear_dependencies()


def test_get_graph_returns_404_when_missing():
    firestore = FakeFirestore()

    override_dependencies(
        firestore
    )

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/graphs/missing"
            )

        assert response.status_code == 404
        assert response.json()["detail"] == (
            "Gráfico no encontrado."
        )

    finally:
        clear_dependencies()


def test_get_graph_hides_other_users_artifact():
    firestore = FakeFirestore()

    override_dependencies(
        firestore
    )

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/graphs/private-123"
            )

        assert response.status_code == 404
        assert response.json()["detail"] == (
            "Gráfico no encontrado."
        )

    finally:
        clear_dependencies()

