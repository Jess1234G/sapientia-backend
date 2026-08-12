"""
logging.py — Configuración global de logging estructurado.

- Desarrollo: salida legible en consola con timestamps.
- Producción: salida JSON para facilitar ingesta (Cloud Logging, ELK, etc.).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.config import settings


class JsonFormatter(logging.Formatter):
    """Formateador que emite cada registro como una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Configura el root logger según el entorno."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)

    if settings.is_production:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Silenciar logs ruidosos de librerías en dev
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
