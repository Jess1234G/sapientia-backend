"""
simpletex_service.py — OCR matemático mediante SimpleTex.

Responsabilidades:
- validar configuración;
- firmar peticiones;
- enviar la imagen a SimpleTex;
- validar la respuesta;
- devolver texto y LaTeX.

No construye prompts.
No llama a DeepSeek.
No ejecuta E2B.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import string
from typing import Any

import requests

from app.config import settings


logger = logging.getLogger(__name__)


class SimpleTexError(RuntimeError):
    """Error controlado del servicio SimpleTex."""


class SimpleTexService:
    """Cliente asíncrono de SimpleTex basado en requests."""

    def __init__(self) -> None:
        self.app_id = settings.simpletex_app_id
        self.app_secret = settings.simpletex_app_secret
        self.api_url = settings.simpletex_api_url

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str = "imagen.png",
        content_type: str = "application/octet-stream",
        connect_timeout_s: int = 10,
        read_timeout_s: int = 60,
    ) -> dict[str, str]:
        """
        Envía una imagen a SimpleTex y devuelve:

            {
                "text": "...",
                "latex": "..."
            }
        """

        self._validate_configuration()

        if not image_bytes:
            raise SimpleTexError(
                "La imagen está vacía."
            )

        timestamp = str(int(__import__("time").time()))
        random_str = "".join(
            secrets.choice(
                string.ascii_letters + string.digits
            )
            for _ in range(16)
        )

        sign_params = {
            "app-id": self.app_id,
            "random-str": random_str,
            "timestamp": timestamp,
        }

        sign_string = "&".join(
            f"{key}={sign_params[key]}"
            for key in sorted(sign_params)
        )

        signature = hashlib.md5(
            f"{sign_string}&secret={self.app_secret}".encode(
                "utf-8"
            )
        ).hexdigest()

        headers = {
            "app-id": self.app_id,
            "random-str": random_str,
            "timestamp": timestamp,
            "sign": signature,
            "Accept": "application/json",
            "User-Agent": "Sapientia/1.0",
        }

        files = {
            "file": (
                filename,
                image_bytes,
                content_type,
            )
        }

        try:
            response = await asyncio.to_thread(
                requests.post,
                self.api_url,
                headers=headers,
                files=files,
                timeout=(
                    connect_timeout_s,
                    read_timeout_s,
                ),
            )
        except requests.exceptions.Timeout as exc:
            raise SimpleTexError(
                "Tiempo de espera agotado al conectar con SimpleTex."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise SimpleTexError(
                f"No fue posible conectar con SimpleTex: {exc}"
            ) from exc

        if response.status_code != 200:
            cf_ray = response.headers.get(
                "CF-Ray",
                "no disponible",
            )
            body = response.text[:2000]

            logger.error(
                "SimpleTex HTTP error | status=%s | CF-Ray=%s",
                response.status_code,
                cf_ray,
            )

            raise SimpleTexError(
                f"SimpleTex respondió HTTP "
                f"{response.status_code}: {body}"
            )

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise SimpleTexError(
                "SimpleTex respondió HTTP 200, "
                "pero no devolvió JSON válido."
            ) from exc

        if not data.get("status"):
            raise SimpleTexError(
                f"SimpleTex rechazó la solicitud: {data}"
            )

        result = data.get("res") or {}

        latex = str(
            result.get("latex") or ""
        ).strip()

        text = str(
            result.get("text") or ""
        ).strip()

        if not latex and not text:
            raise SimpleTexError(
                "SimpleTex no devolvió texto ni LaTeX."
            )

        return {
            "text": text,
            "latex": latex,
        }

    def _validate_configuration(self) -> None:
        """Valida la configuración mínima requerida."""

        if not self.app_id:
            raise SimpleTexError(
                "SIMPLETEX_APP_ID no configurado."
            )

        if not self.app_secret:
            raise SimpleTexError(
                "SIMPLETEX_APP_SECRET no configurado."
            )

        if not self.api_url:
            raise SimpleTexError(
                "SIMPLETEX_API_URL no configurado."
            )


def get_simpletex_service() -> SimpleTexService:
    """Dependencia para FastAPI."""

    return SimpleTexService()
