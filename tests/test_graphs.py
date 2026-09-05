from __future__ import annotations

import asyncio

import app.worker.tasks as tasks_module
from app.worker.tasks import generate_graph_task


class FakeFirestore:
    def __init__(self):
        self.artifact = {
            "artifact_id": "artifact-123",
            "user_id": "user-123",
            "conversation_id": "chat-123",
            "status": "pending",
            "code": "print('graph')",
        }
        self.updates = []

    async def get_graph_artifact(self, artifact_id: str):
        if artifact_id != self.artifact["artifact_id"]:
            return None
        return dict(self.artifact)

    async def update_graph_artifact(self, artifact_id: str, **fields):
        self.updates.append((artifact_id, fields))
        self.artifact.update(fields)


class FakeRunner:
    async def execute(self, code: str, timeout_s: int = 60):
        return type(
            "Artifacts",
            (),
            {
                "files": {
                    "figura_3d.html": b"<html>3D</html>",
                }
            },
        )()


class FakeGCS:
    def __init__(self):
        self.calls = []

    def upload_bytes(
        self,
        content: bytes,
        prefix: str,
        ext: str,
        content_type: str,
    ) -> str:
        self.calls.append(
            {
                "content": content,
                "prefix": prefix,
                "ext": ext,
                "content_type": content_type,
            }
        )
        return "https://gcs.test/graph.html"


class FakeTaskRequest:
    id = "task-123"


class FakeTaskSelf:
    request = FakeTaskRequest()


def test_generate_graph_task_success(monkeypatch):
    firestore = FakeFirestore()
    runner = FakeRunner()
    gcs = FakeGCS()

    monkeypatch.setattr(
        tasks_module,
        "get_firestore_service",
        lambda: firestore,
    )
    monkeypatch.setattr(
        tasks_module,
        "get_e2b_service",
        lambda: runner,
    )
    monkeypatch.setattr(
        tasks_module,
        "get_r2_service",
        lambda: gcs,
    )

    result = generate_graph_task.run(
        "artifact-123",
        "print('graph')",
    )

    assert result["artifact_id"] == "artifact-123"
    assert result["status"] == "completed"
    assert result["html_url"] == "https://gcs.test/graph.html"

    assert gcs.calls[0]["content"] == b"<html>3D</html>"
    assert gcs.calls[0]["ext"] == "html"
    assert gcs.calls[0]["content_type"] == "text/html"

    assert firestore.artifact["status"] == "completed"
    assert firestore.artifact["html_url"] == (
        "https://gcs.test/graph.html"
    )


def test_generate_graph_task_artifact_not_found(monkeypatch):
    firestore = FakeFirestore()

    monkeypatch.setattr(
        tasks_module,
        "get_firestore_service",
        lambda: firestore,
    )
    monkeypatch.setattr(
        tasks_module,
        "get_e2b_service",
        lambda: FakeRunner(),
    )
    monkeypatch.setattr(
        tasks_module,
        "get_r2_service",
        lambda: FakeGCS(),
    )

    try:
        generate_graph_task.run(
            "missing",
            "print('graph')",
        )
    except Exception as exc:
        assert "artefacto" in str(exc).lower()
    else:
        raise AssertionError(
            "Debía fallar si el artefacto no existe."
        )


def test_generate_graph_task_requires_3d_html(monkeypatch):
    firestore = FakeFirestore()

    class NoHtmlRunner:
        async def execute(
            self,
            code: str,
            timeout_s: int = 60,
        ):
            return type(
                "Artifacts",
                (),
                {"files": {}},
            )()

    monkeypatch.setattr(
        tasks_module,
        "get_firestore_service",
        lambda: firestore,
    )
    monkeypatch.setattr(
        tasks_module,
        "get_e2b_service",
        lambda: NoHtmlRunner(),
    )
    monkeypatch.setattr(
        tasks_module,
        "get_r2_service",
        lambda: FakeGCS(),
    )

    try:
        generate_graph_task.run(
            "artifact-123",
            "print('graph')",
        )
    except Exception as exc:
        assert "figura_3d.html" in str(exc)
    else:
        raise AssertionError(
            "Debía fallar si falta figura_3d.html."
        )

