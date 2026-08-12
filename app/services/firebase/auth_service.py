"""
auth_service.py — Verificación de identidad.

- `exchange_google_token(id_token)`: valida el ID token de Google/Firebase
  y devuelve las claims del usuario.
- `validate_token(jwt)`: decide entre un ID token de Firebase o el JWT
  propio del backend (HS256), y devuelve el `uid`.
"""
from __future__ import annotations

import logging

from firebase_admin import auth as firebase_auth
from jose import JWTError, jwt

from app.config import settings
from app.services.firebase.client import init_firebase

logger = logging.getLogger(__name__)


class AuthService:
    """Servicio de autenticación y autorización."""

    def __init__(self) -> None:
        self.secret = settings.secret_key

    def verify_google_id_token(self, id_token: str) -> dict | None:
        """
        Valida un ID token de Google/Firebase con Firebase Admin.

        Devuelve las claims (`uid`, `email`, `name`, `picture`) o None.
        """
        try:
            init_firebase()
            decoded = firebase_auth.verify_id_token(id_token)
            return {
                "uid": decoded["uid"],
                "email": decoded.get("email"),
                "name": decoded.get("name"),
                "picture": decoded.get("picture"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("ID token inválido: %s", exc)
            return None

    def create_session_token(self, uid: str) -> str:
        """Emite un JWT propio (HS256) con claims mínimas para la sesión."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        payload = {
            "sub": uid,
            "iat": now,
            "exp": now + timedelta(days=1),
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def validate_token(self, token: str) -> str | None:
        """
        Devuelve el `uid` si el token es un JWT propio válido.
        (El ID token de Google se valida solo en /auth/google al hacer login.)
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            return payload.get("sub")
        except JWTError:
            logger.warning("Sesión JWT inválida o expirada")
            return None


def get_auth_service() -> AuthService:
    """Dependencia FastAPI: AuthService singleton."""
    return AuthService()
