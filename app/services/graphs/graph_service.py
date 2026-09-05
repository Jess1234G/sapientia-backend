"""
graph_service.py — Orquestación de generación de gráficos.

Responsabilidades:
- validar el código;
- crear GraphArtifact;
- encolar generate_graph_task;
- guardar task_id;
- marcar failed si el encolado falla.

No ejecuta E2B.
No sube GCS.
No genera gráficos.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.firebase.firestore_service import FirestoreService
from app.worker.tasks import generate_graph_task


class GraphServiceError(RuntimeError):
    """Error controlado durante la creación de un gráfico."""


@dataclass(frozen=True)
class GraphCreationResult:
    artifact_id: str
    task_id: str


class GraphService:
    """Orquesta la creación y encolado de un gráfico."""

    def __init__(
        self,
        firestore: FirestoreService,
    ) -> None:
        self.firestore = firestore

    async def create_graph(
        self,
        *,
        user_id: str,
        conversation_id: str,
        code: str,
    ) -> GraphCreationResult:
        code = code.strip()

        if not code:
            raise GraphServiceError(
                "El código Python del gráfico es obligatorio."
            )

        artifact_id = (
            await self.firestore.create_graph_artifact(
                user_id=user_id,
                conversation_id=conversation_id,
                code=code,
                status="pending",
            )
        )

        try:
            task = generate_graph_task.delay(
                artifact_id,
                code,
            )
        except Exception as exc:
            await self.firestore.update_graph_artifact(
                artifact_id,
                status="failed",
                error=str(exc),
            )

            raise GraphServiceError(
                "No fue posible encolar la generación del gráfico."
            ) from exc

        await self.firestore.update_graph_artifact(
            artifact_id,
            task_id=task.id,
        )

        return GraphCreationResult(
            artifact_id=artifact_id,
            task_id=task.id,
        )
