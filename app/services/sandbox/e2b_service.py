"""
e2b_service.py — Ejecución de código Python en sandbox seguro (E2B).

Ejecuta el código generado por DeepSeek R1 de forma aislada y recupera
los artefactos `figura_2d.png` (Matplotlib) y `figura_3d.html` (Plotly).
"""
from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """Error al ejecutar código en el sandbox."""


class E2BService:
    """Runner de código Python sobre E2B."""

    def __init__(self) -> None:
        self.api_key = settings.e2b_api_key
        self.template = settings.e2b_template
        # El SDK real se inicializa aquí (import perezoso para no romper tests).
        self._client = None

    async def run(self, code: str, timeout_s: int = 60) -> dict:
        """
        Ejecuta `code` y devuelve los artefactos generados.

        Output esperado: {"files": [{"name": "...", "content": bytes}]}
        El worker filtra `figura_2d.png` y `figura_3d.html`.
        """
        if not self.api_key:
            raise SandboxError("E2B_API_KEY no configurada")

        # TODO: implementar con el SDK de E2B
        #   sandbox = await Sandbox.create(self.template, api_key=self.api_key)
        #   result = await sandbox.run_python(code, timeout=timeout_s)
        #   files = await sandbox.list_files()
        #   await sandbox.kill()
        logger.info("E2B sandbox solicitado con template=%s", self.template)
        raise SandboxError("Integración E2B pendiente de implementación")

    def _fetch_file(self, path: str) -> bytes:
        """Descarga un archivo del sandbox como bytes."""
        # TODO: sandbox.download(path)
        raise NotImplementedError


def get_e2b_service() -> E2BService:
    """Dependencia FastAPI: E2BService."""
    return E2BService()
