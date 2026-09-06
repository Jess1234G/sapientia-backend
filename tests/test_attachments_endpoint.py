"""Pruebas del endpoint POST /api/v1/attachments."""
from __future__ import annotations

from app.api.v1.endpoints.attachments import get_attachment_service
from app.config import settings
from app.core.security import get_current_user
from app.main import app
from app.services.storage.attachment_service import AttachmentService


class FakeR2:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def upload_bytes_with_key(self, storage_key, content, content_type):
        self.uploaded.append((storage_key, content, content_type))

    def generate_presigned_url(self, storage_key, expires_in=3600):
        return f"https://fake.example/{storage_key}"

    def delete_object(self, storage_key):
        self.deleted.append(storage_key)


class FakeFirestore:
    def __init__(self):
        self.attachments = {}

    async def create_attachment_metadata(self, **kwargs):
        data = dict(kwargs)
        data["created_at"] = "2026-01-01T00:00:00+00:00"
        self.attachments[kwargs["attachment_id"]] = data
        return data

    async def get_user_attachment(self, attachment_id, user_id):
        data = self.attachments.get(attachment_id)
        if data is None or data["user_id"] != user_id:
            return None
        return data


def _override(r2=None, firestore=None):
    r2 = r2 or FakeR2()
    firestore = firestore or FakeFirestore()
    service = AttachmentService(r2=r2, firestore=firestore)
    app.dependency_overrides[get_current_user] = lambda: "user-123"
    app.dependency_overrides[get_attachment_service] = lambda: service
    return service, r2, firestore


def _clear():
    app.dependency_overrides.clear()


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake" * 10


def test_upload_valid(client):
    _override()
    try:
        response = client.post(
            "/api/v1/attachments",
            files={"file": ("foto.png", PNG_BYTES, "image/png")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["attachment_id"]
        assert data["filename"] == "foto.png"
        assert data["content_type"] == "image/png"
        assert data["size"] == len(PNG_BYTES)
    finally:
        _clear()


def test_upload_without_auth(client):
    response = client.post(
        "/api/v1/attachments",
        files={"file": ("foto.png", PNG_BYTES, "image/png")},
    )
    assert response.status_code == 401


def test_upload_invalid_type(client):
    _override()
    try:
        response = client.post(
            "/api/v1/attachments",
            files={
                "file": (
                    "doc.docx",
                    b"fake",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert response.status_code == 400
    finally:
        _clear()


def test_upload_empty_file(client):
    _override()
    try:
        response = client.post(
            "/api/v1/attachments",
            files={"file": ("foto.png", b"", "image/png")},
        )
        assert response.status_code == 400
    finally:
        _clear()


def test_upload_size_exceeded(client, monkeypatch):
    monkeypatch.setattr(settings, "max_attachment_size_mb", 1)
    _override()
    try:
        content = b"x" * (1024 * 1024 + 1)
        response = client.post(
            "/api/v1/attachments",
            files={"file": ("file.txt", content, "text/plain")},
        )
        assert response.status_code == 413
    finally:
        _clear()


def test_response_contains_attachment_id(client):
    _override()
    try:
        response = client.post(
            "/api/v1/attachments",
            files={"file": ("foto.png", PNG_BYTES, "image/png")},
        )
        data = response.json()
        assert data["attachment_id"]
    finally:
        _clear()


def test_response_contains_metadata(client):
    _override()
    try:
        response = client.post(
            "/api/v1/attachments",
            files={"file": ("foto.png", PNG_BYTES, "image/png")},
        )
        data = response.json()
        assert data["filename"] == "foto.png"
        assert data["content_type"] == "image/png"
        assert data["size"] == len(PNG_BYTES)
    finally:
        _clear()


def test_response_does_not_expose_storage_key(client):
    _override()
    try:
        response = client.post(
            "/api/v1/attachments",
            files={"file": ("foto.png", PNG_BYTES, "image/png")},
        )
        data = response.json()
        assert "storage_key" not in data
        assert "user_id" not in data
        assert "created_at" not in data
    finally:
        _clear()
