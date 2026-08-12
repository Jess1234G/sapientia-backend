"""
gcs_service.py — Subida de artefactos a Google Cloud Storage.

Sube PNG (vista 2D) y HTML (vista 3D) y devuelve URLs accesibles.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from google.cloud import storage

from app.config import settings
from app.utils.ids import new_uuid

logger = logging.getLogger(__name__)


class GcsService:
    """Almacenamiento de artefactos en GCS."""

    def __init__(self) -> None:
        self.client = storage.Client()
        self.bucket = self.client.bucket(settings.gcs_bucket_name)

    def upload_bytes(self, content: bytes, prefix: str, ext: str, content_type: str) -> str:
        """Sube bytes a GCS y devuelve la URL pública (o firmada)."""
        blob_name = f"{prefix}/{new_uuid()}.{ext}"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(content, content_type=content_type)

        if settings.gcs_bucket_public:
            return blob.public_url
        # URL firmada de 1 hora para buckets privados
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET",
        )

    def upload_artifact(
        self,
        user_id: str,
        artifact_id: str,
        png_bytes: bytes,
        html_bytes: bytes,
    ) -> dict[str, str]:
        """Sube el par PNG + HTML y devuelve {png_url, html_url}."""
        prefix = f"artifacts/{user_id}/{artifact_id}"
        now = datetime.now(timezone.utc).strftime("%Y%m%d")
        png_url = self.upload_bytes(png_bytes, f"{prefix}/{now}", "png", "image/png")
        html_url = self.upload_bytes(html_bytes, f"{prefix}/{now}", "html", "text/html")
        return {"png_url": png_url, "html_url": html_url}


def get_gcs_service() -> GcsService:
    """Dependencia FastAPI: GcsService."""
    return GcsService()
