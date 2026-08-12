"""
celery_app.py — Instancia Celery conectada a Redis.

Se usa tanto para la API (encolar) como para el worker (consumir).
"""
from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "sapientia",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue=settings.celery_task_queue,
    broker_connection_retry_on_startup=True,
    task_track_started=True,
)
