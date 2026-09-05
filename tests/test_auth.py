"""
Pruebas del endpoint de autenticación modular.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.core.security import get_current_user
from app.services.firebase.auth_service import get_auth_service
from app.services.firebase.firestore_service import (
    get_firestore_service,
)
from app.main import app


# ============================================================
# HELPERS
# ============================================================

class FakeAuthService:
    """AuthService simulado para las pruebas."""

    def __init__(self, claims=None):
        self.claims = claims

    def verify_google_id_token(self, id_token: str):
        return self.claims

    def create_session_token(self, uid: str) -> str:
        return "test-session-token"


class FakeFirestoreService:
    """Firestore simulado para las pruebas."""

    def __init__(self):
        self.users = {}

    async def upsert_user(
        self,
        uid: str,
        email: str,
        name: str,
        picture: str,
    ):
        user = {
            "uid": uid,
            "email": email,
            "display_name": name,
            "photo_url": picture,
            "carrera": "",
            "semestre": 0,
            "created_at": "2026-08-21T00:00:00+00:00",
            "updated_at": "2026-08-21T00:00:00+00:00",
        }

        self.users[uid] = user

        return user

    async def get_user(self, uid: str):
        return self.users.get(uid)


# ============================================================
# GOOGLE LOGIN
# ============================================================

def test_auth_google_valid_token(client):
    """Un token válido crea sesión y perfil."""

    auth_service = FakeAuthService(
        claims={
            "uid": "test-user",
            "email": "test@example.com",
            "name": "Usuario de Prueba",
            "picture": "https://example.com/avatar.png",
        }
    )

    firestore = FakeFirestoreService()

    app.dependency_overrides[
        get_auth_service
    ] = lambda: auth_service

    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore

    try:
        response = client.post(
            "/api/v1/auth/google",
            json={
                "id_token": "valid-google-token",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["access_token"] == (
            "test-session-token"
        )

        assert data["token_type"] == "bearer"

        assert data["user"]["uid"] == "test-user"
        assert data["user"]["email"] == (
            "test@example.com"
        )
        assert data["user"]["display_name"] == (
            "Usuario de Prueba"
        )

    finally:
        app.dependency_overrides.clear()


def test_auth_google_invalid_token(client):
    """Un token inválido debe devolver 401."""

    auth_service = FakeAuthService(
        claims=None
    )

    firestore = FakeFirestoreService()

    app.dependency_overrides[
        get_auth_service
    ] = lambda: auth_service

    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore

    try:
        response = client.post(
            "/api/v1/auth/google",
            json={
                "id_token": "invalid-google-token",
            },
        )

        assert response.status_code == 401

        data = response.json()

        assert data["detail"] == (
            "Token de Google/Firebase inválido o expirado."
        )

    finally:
        app.dependency_overrides.clear()


# ============================================================
# /AUTH/ME
# ============================================================

def test_auth_me_requires_token(client):
    """/auth/me debe exigir autenticación."""

    firestore = FakeFirestoreService()

    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore

    try:
        response = client.get(
            "/api/v1/auth/me"
        )

        assert response.status_code == 401

    finally:
        app.dependency_overrides.clear()


def test_auth_me_returns_user(client):
    """Un usuario autenticado puede recuperar su perfil."""

    firestore = FakeFirestoreService()

    firestore.users["test-user"] = {
        "uid": "test-user",
        "email": "test@example.com",
        "display_name": "Usuario de Prueba",
        "photo_url": "",
        "carrera": "Ingeniería Agroindustrial",
        "semestre": 5,
        "created_at": "2026-08-21T00:00:00+00:00",
        "updated_at": "2026-08-21T00:00:00+00:00",
    }

    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore

    app.dependency_overrides[
        get_current_user
    ] = lambda: "test-user"

    try:
        response = client.get(
            "/api/v1/auth/me"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["uid"] == "test-user"
        assert data["email"] == (
            "test@example.com"
        )
        assert data["display_name"] == (
            "Usuario de Prueba"
        )
        assert data["carrera"] == (
            "Ingeniería Agroindustrial"
        )
        assert data["semestre"] == 5

    finally:
        app.dependency_overrides.clear()


def test_auth_me_user_not_found(client):
    """Si no existe el perfil, /auth/me devuelve 404."""

    firestore = FakeFirestoreService()

    app.dependency_overrides[
        get_firestore_service
    ] = lambda: firestore

    app.dependency_overrides[
        get_current_user
    ] = lambda: "unknown-user"

    try:
        response = client.get(
            "/api/v1/auth/me"
        )

        assert response.status_code == 404

        data = response.json()

        assert data["detail"] == (
            "Perfil de usuario no encontrado."
        )

    finally:
        app.dependency_overrides.clear()