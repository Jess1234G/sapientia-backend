"""
firebase/client.py — Singleton de firebase_admin.

Inicializa Firebase Admin SDK en el arranque con credenciales desde
variables de entorno o desde el archivo de cuenta de servicio.
"""
from __future__ import annotations

import json
import logging

import firebase_admin
from firebase_admin import credentials

from app.config import settings

logger = logging.getLogger(__name__)

# Caché del default app de firebase_admin (thread-safe por el SDK)
_default_app: firebase_admin.App | None = None


def init_firebase() -> firebase_admin.App:
    """Inicializa (una sola vez) Firebase Admin y devuelve la app."""
    global _default_app
    if _default_app is not None:
        return _default_app

    # 1) Preferir archivo de cuenta de servicio
    service_account = settings.firebase_service_account_path
    try:
        if settings.firebase_credentials_available:
            # El SDK de Firebase Admin exige, además de `client_email`,
            # el campo `token_uri` (y el resto de campos estándar de una
            # service account). Reconstruimos el JSON completo: los valores
            # sensibles vienen de variables de entorno y los valores públicos
            # (URIs de OAuth y certificados) son constantes de Google.
            private_key = settings.firebase_private_key.replace(
                "\\n", "\n"
            )
            client_email = settings.firebase_client_email

            cred = credentials.Certificate(
                {
                    "type": "service_account",
                    "project_id": settings.firebase_project_id,
                    "private_key_id": settings.firebase_private_key_id,
                    "private_key": private_key,
                    "client_email": client_email,
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_uri": (
                        "https://accounts.google.com/o/oauth2/auth"
                    ),
                    "auth_provider_x509_cert_url": (
                        "https://www.googleapis.com/oauth2/v1/certs"
                    ),
                    "client_x509_cert_url": (
                        "https://www.googleapis.com/robot/v1/metadata/x509/"
                        + client_email
                    ),
                    "universe_domain": "googleapis.com",
                }
            )
            _default_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin inicializado desde variables de entorno")
            return _default_app
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudieron usar credenciales por env: %s", exc)

    # 2) Fallback: archivo JSON de cuenta de servicio
    try:
        with open(service_account, encoding="utf-8") as fh:
            service_account_info = json.load(fh)
        cred = credentials.Certificate(service_account_info)
        _default_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin inicializado desde %s", service_account)
        return _default_app
    except FileNotFoundError:
        logger.error(
            "Credenciales de Firebase no configuradas. "
            "Revisa backend/.env o backend/secrets/service-account.json"
        )
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inicializando Firebase Admin")
        raise


def get_firebase_app() -> firebase_admin.App:
    """Dependencia FastAPI: app de Firebase inicializada."""
    return init_firebase()
