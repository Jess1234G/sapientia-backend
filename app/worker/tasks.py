"""
tasks.py — Tareas pesadas de Celery.

- `generate_graph_task`: ejecuta el código Python de la IA en E2B,
  genera PNG (Matplotlib) + HTML (Plotly) y los sube a GCS.
- `analyze_image_task`: análisis OCR/LaTeX de imágenes con el modelo de visión.
"""
from __future__ import annotations

import logging

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="sapientia.generate_graph", bind=True, max_retries=3)
def generate_graph_task(self, artifact_id: str, code: str) -> dict:
    """
    Genera gráfico Dual View en sandbox seguro.

    Contrato:
      - input:  artifact_id (Firestore) + código Python (matplotlib/plotly)
      - output: {png_url, html_url} persistidos en el artefacto.
    """
    logger.info("generate_graph_task iniciado: %s", artifact_id)
    # TODO: implementar con e2b_service + gcs_service
    return {"artifact_id": artifact_id, "status": "completed"}


@celery_app.task(name="sapientia.analyze_image", bind=True, max_retries=2)
def analyze_image_task(self, attachment_id: str, image_path: str) -> dict:
    """Analiza una imagen (OCR + LaTeX) de forma asíncrona."""
    logger.info("analyze_image_task iniciado: %s", attachment_id)
    # TODO: implementar con vision_service
    return {"attachment_id": attachment_id, "status": "completed"}
