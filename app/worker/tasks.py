"""
tasks.py — Tareas pesadas de Celery.

- generate_graph_task:
    ejecuta código Python mediante CodeRunner/E2B,
    recupera figura_3d.html, lo sube a R2 y actualiza
    el GraphArtifact en Firestore.

- analyze_image_task:
    reservado para el procesamiento asíncrono de visión.
"""

from __future__ import annotations

import asyncio
import logging

from app.services.firebase.firestore_service import (
    get_firestore_service,
)
from app.services.sandbox.e2b_service import (
    get_e2b_service,
)
from app.services.storage.r2_service import (
    get_r2_service,
)
from app.worker.celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(
    name="sapientia.generate_graph",
    bind=True,
    max_retries=3,
)
def generate_graph_task(
    self,
    artifact_id: str,
    code: str,
) -> dict:
    """
    Ejecuta código Python para generar figura_3d.html.

    Flujo:
        Firestore
        -> estado running
        -> CodeRunner / E2B
        -> figura_3d.html
        -> R2
        -> Firestore completed

    En caso de error:
        Firestore -> status=failed + error
        -> propaga la excepción.
    """

    logger.info(
        "generate_graph_task iniciado | artifact_id=%s",
        artifact_id,
    )

    async def _run() -> dict:
        firestore = get_firestore_service()
        runner = get_e2b_service()
        r2 = get_r2_service()

        artifact = await firestore.get_graph_artifact(
            artifact_id
        )

        if artifact is None:
            raise RuntimeError(
                f"El artefacto '{artifact_id}' no existe."
            )

        user_id = str(
            artifact.get("user_id") or ""
        )

        if not user_id:
            raise RuntimeError(
                f"El artefacto '{artifact_id}' "
                "no tiene user_id."
            )

        task_id = getattr(
            getattr(self, "request", None),
            "id",
            "",
        ) or ""

        await firestore.update_graph_artifact(
            artifact_id,
            status="running",
            task_id=task_id,
            code=code,
        )

        try:
            execution = await runner.execute(
                code=code,
                timeout_s=60,
            )

            html_bytes = execution.files.get(
                "figura_3d.html"
            )

            if not html_bytes:
                raise RuntimeError(
                    "E2B no produjo "
                    "'figura_3d.html'."
                )

            html_url = r2.upload_bytes(
                content=html_bytes,
                prefix=(
                    f"artifacts/"
                    f"{user_id}/"
                    f"{artifact_id}"
                ),
                ext="html",
                content_type="text/html",
            )

            await firestore.update_graph_artifact(
                artifact_id,
                status="completed",
                html_url=html_url,
                error="",
            )

            result = {
                "artifact_id": artifact_id,
                "task_id": task_id,
                "status": "completed",
                "html_url": html_url,
            }

            logger.info(
                "generate_graph_task completado | "
                "artifact_id=%s",
                artifact_id,
            )

            return result

        except Exception as exc:
            error_message = str(exc)

            logger.exception(
                "generate_graph_task falló | "
                "artifact_id=%s",
                artifact_id,
            )

            try:
                await firestore.update_graph_artifact(
                    artifact_id,
                    status="failed",
                    error=error_message,
                )
            except Exception:
                logger.exception(
                    "No fue posible actualizar el "
                    "artefacto como failed | artifact_id=%s",
                    artifact_id,
                )

            raise

    return asyncio.run(_run())


@celery_app.task(
    name="sapientia.analyze_image",
    bind=True,
    max_retries=2,
)
def analyze_image_task(
    self,
    attachment_id: str,
    image_path: str,
) -> dict:
    """
    Analiza una imagen de forma asíncrona.

    Pendiente de integración con VisionService.
    """
    logger.info(
        "analyze_image_task iniciado: %s",
        attachment_id,
    )

    return {
        "attachment_id": attachment_id,
        "status": "completed",
    }

