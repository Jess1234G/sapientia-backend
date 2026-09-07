"""
Pruebas del endpoint GET /api/v1/attachments/{attachment_id}/url (F6-C1).

Verifica autenticación, ownership, omisión de storage_key y que la
URL prefirmada generada sea temporal.
"""

from __future__ import annotations

from app.api.v1.endpoints.attachments import get_attachment_service
from app.core.security import get_current_user
from app.main import app
from app.services.storage.attachment_service import AttachmentService


class FakeR2:
    def __init__(self):
        self.last_storage_key = None
        self.last_expires_in = None

    def generate_presigned_url(self, storage_key, expires_in=3600):
        self.last_storage_key = storage_key
        self.last_expires_in = expires_in
        return f"https://fake.example/{storage_key}?expires={expires_in}"


class FakeFirestore:
    def __init__(self):
        self.attachments = {
            "att-1": {
                "attachment_id": "att-1",
                "user_id": "user-123",
                "filename": "foto.png",
                "content_type": "image/png",
                "size": 100,
                "storage_key": "attachments/user-123/foto.png",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            "att-other": {
                "attachment_id": "att-other",
                "user_id": "other-user",
                "filename": "secreto.pdf",
                "content_type": "application/pdf",
                "size": 200,
                "storage_key": "attachments/other-user/secreto.pdf",
            },
        }

    async def get_user_attachment(self, attachment_id, user_id):
        data = self.attachments.get(attachment_id)

        if data is None or data.get("user_id") != user_id:
            return None

        return data


def _override():
    r2 = FakeR2()
    firestore = FakeFirestore()
    service = AttachmentService(r2=r2, firestore=firestore)
    app.dependency_overrides[get_current_user] = lambda: "user-123"
    app.dependency_overrides[get_attachment_service] = lambda: service
    return service, r2, firestore


def _clear():
    app.dependency_overrides.clear()


# ============================================================
# H. GET autenticado -> 200
# ============================================================

def test_get_url_authenticated_returns_200(client):
    _override()

    try:
        response = client.get("/api/v1/attachments/att-1/url")

        assert response.status_code == 200

        data = response.json()

        assert data["attachment_id"] == "att-1"
        assert data["url"].startswith("https://fake.example/")

    finally:
        _clear()


# ============================================================
# 401 sin token
# ============================================================

def test_get_url_without_auth_returns_401(client):
    response = client.get("/api/v1/attachments/att-1/url")

    assert response.status_code == 401


# ============================================================
# I. URL de otro usuario -> 404
# ============================================================

def test_get_url_other_user_returns_404(client):
    _override()

    try:
        response = client.get("/api/v1/attachments/att-other/url")

        assert response.status_code == 404

    finally:
        _clear()


# ============================================================
# J. Attachment inexistente -> 404
# ============================================================

def test_get_url_missing_returns_404(client):
    _override()

    try:
        response = client.get("/api/v1/attachments/does-not-exist/url")

        assert response.status_code == 404

    finally:
        _clear()


# ============================================================
# K. Response NO contiene storage_key
# ============================================================

def test_get_url_does_not_expose_storage_key(client):
    _override()

    try:
        response = client.get("/api/v1/attachments/att-1/url")

        data = response.json()

        assert "storage_key" not in data
        assert "user_id" not in data
        assert "created_at" not in data

    finally:
        _clear()


# ============================================================
# L. Response contiene filename/content_type/size
# ============================================================

def test_get_url_contains_metadata(client):
    _override()

    try:
        response = client.get("/api/v1/attachments/att-1/url")

        data = response.json()

        assert data["filename"] == "foto.png"
        assert data["content_type"] == "image/png"
        assert data["size"] == 100

    finally:
        _clear()


# ============================================================
# M. Presigned URL es temporal
# ============================================================

def test_get_url_is_temporary(client):
    _, r2, _ = _override()

    try:
        response = client.get("/api/v1/attachments/att-1/url")

        assert response.status_code == 200

        assert r2.last_storage_key == "attachments/user-123/foto.png"
        assert r2.last_expires_in is not None
        assert r2.last_expires_in < 3600

    finally:
        _clear()
