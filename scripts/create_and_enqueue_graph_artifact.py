"""
create_and_enqueue_graph_artifact.py

Prueba unificada para diagnosticar el incidente del worker.

En un único proceso:

1. Inicializa Firestore.
2. Muestra el project_id efectivo (para descartar que worker y
   creador vean proyectos distintos).
3. Crea el GraphArtifact y CONSERVA el artifact_id en una variable.
4. Verifica inmediatamente la lectura (get_graph_artifact).
5. Encola generate_graph_task usando ESA MISMA variable (sin
   copiar/pegar manualmente el ID).
6. Imprime artifact_id + task_id.

No ejecuta E2B, R2 ni GCS. Solo diagnostica la capa Firestore/Celery.
"""

from __future__ import annotations

import asyncio

from app.services.firebase.firestore_service import (
    get_firestore_service,
)
from app.worker.tasks import generate_graph_task


PLOTLY_CODE = (
    "import plotly.graph_objects as go\n"
    "fig = go.Figure(\n"
    "    data=[go.Scatter3d(\n"
    "        x=[1, 2, 3],\n"
    "        y=[1, 4, 9],\n"
    "        z=[1, 8, 27],\n"
    "    )]\n"
    ")\n"
    "fig.write_html(\n"
    "    'figura_3d.html',\n"
    "    include_plotlyjs='cdn'\n"
    ")\n"
)


async def main() -> None:
    firestore = get_firestore_service()

    # Diagnóstico: ¿qué proyecto ve este proceso?
    project_id = getattr(
        firestore.client,
        "project",
        None,
    )
    print(f"PROJECT_ID={project_id}")

    # 1. Crear el artefacto.
    artifact_id = await firestore.create_graph_artifact(
        user_id="test-real-user",
        conversation_id="test-real-conversation",
        code=PLOTLY_CODE,
        status="pending",
    )
    print(f"ARTIFACT_ID={artifact_id}")

    # 2. Verificar lectura inmediata con la MISMA variable.
    artifact = await firestore.get_graph_artifact(
        artifact_id
    )
    print(f"READBACK_FOUND={artifact is not None}")
    if artifact is not None:
        print(f"READBACK_STATUS={artifact.get('status')}")
        print(f"READBACK_USER_ID={artifact.get('user_id')}")

    # 3. Encolar usando ESA MISMA variable (sin intervención manual).
    task = generate_graph_task.delay(
        artifact_id,
        PLOTLY_CODE,
    )
    print(f"TASK_ID={task.id}")

    print("ENQUEUED_OK=True")


if __name__ == "__main__":
    asyncio.run(main())
