"""
e2b_service.py — Ejecución segura de código Python mediante E2B.

Responsabilidades:

- validar la configuración de E2B;
- crear un sandbox aislado;
- ejecutar código Python;
- detectar errores de ejecución;
- recuperar figura_3d.html;
- cerrar siempre el sandbox;
- devolver artefactos en el contrato esperado por Sapientia.

No realiza uploads a GCS.
No escribe Firestore.
No modifica Celery.
"""

from __future__ import annotations

import logging
from typing import Any

from e2b_code_interpreter import AsyncSandbox

from app.config import settings
from app.services.sandbox.code_runner import ExecutionArtifacts


logger = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """Error controlado durante la ejecución del sandbox."""


class E2BService:
    """Runner de código Python sobre E2B Code Interpreter."""

    GRAPH_FILE = "figura_3d.html"

    def __init__(self) -> None:
        self.api_key = settings.e2b_api_key
        self.template = settings.e2b_template
        self._client = None

    async def execute(
        self,
        code: str,
        timeout_s: int = 60,
    ) -> ExecutionArtifacts:
        """
        Implementa el contrato CodeRunner.

        Reutiliza la ejecución E2B existente y transforma
        su salida al contrato estándar ExecutionArtifacts.
        """

        result = await self.run(
            code=code,
            timeout_s=timeout_s,
        )

        files = {
            item["name"]: item["content"]
            for item in result.get("files", [])
        }

        return ExecutionArtifacts(
            files=files,
        )

    async def run(
        self,
        code: str,
        timeout_s: int = 60,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Ejecuta código Python en E2B y recupera figura_3d.html.

        Contrato de salida:

            {
                "files": [
                    {
                        "name": "figura_3d.html",
                        "content": b"...",
                    }
                ]
            }

        Raises:
            SandboxError:
                Si falta la API key, falla la ejecución o no se
                encuentra el artefacto esperado.
        """

        if not self.api_key:
            raise SandboxError(
                "E2B_API_KEY no configurada"
            )

        if not code or not code.strip():
            raise SandboxError(
                "El código Python está vacío"
            )

        if timeout_s <= 0:
            raise SandboxError(
                "timeout_s debe ser mayor que 0"
            )

        sandbox = None

        try:
            logger.info(
                "Creando sandbox E2B | template=%s | timeout=%ss",
                self.template,
                timeout_s,
            )

            sandbox = await AsyncSandbox.create(
                template=self.template,
                timeout=timeout_s,
                api_key=self.api_key,
            )

            execution = await sandbox.run_code(
                code,
                language="python",
                timeout=timeout_s,
            )

            if execution.error is not None:
                raise SandboxError(
                    "Error ejecutando código en E2B: "
                    f"{execution.error}"
                )

            try:
                html_bytes = await sandbox.files.read(
                    self.GRAPH_FILE,
                    format="bytes",
                )
            except Exception as exc:
                raise SandboxError(
                    "No se encontró el artefacto esperado "
                    f"'{self.GRAPH_FILE}' en el sandbox."
                ) from exc

            if not html_bytes:
                raise SandboxError(
                    f"El archivo '{self.GRAPH_FILE}' "
                    "está vacío."
                )

            logger.info(
                "E2B completado correctamente | archivo=%s | bytes=%s",
                self.GRAPH_FILE,
                len(html_bytes),
            )

            return {
                "files": [
                    {
                        "name": self.GRAPH_FILE,
                        "content": html_bytes,
                    }
                ]
            }

        except SandboxError:
            raise

        except Exception as exc:
            logger.exception(
                "Error inesperado durante ejecución E2B"
            )
            raise SandboxError(
                "Error durante la ejecución del sandbox E2B."
            ) from exc

        finally:
            if sandbox is not None:
                try:
                    await sandbox.kill()

                    logger.info(
                        "Sandbox E2B cerrado correctamente"
                    )

                except Exception:
                    logger.exception(
                        "No fue posible cerrar correctamente "
                        "el sandbox E2B"
                    )

    def _fetch_file(self, path: str) -> bytes:
        """
        Compatibilidad con el contrato anterior.

        La recuperación actual se realiza directamente mediante
        sandbox.files.read() dentro de run().
        """

        raise NotImplementedError(
            "_fetch_file() no se utiliza en la implementación async actual."
        )


def get_e2b_service() -> E2BService:
    """Dependencia FastAPI: E2BService."""

    return E2BService()

