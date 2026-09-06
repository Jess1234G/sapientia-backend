"""
r2_service.py — Almacenamiento de artefactos en Cloudflare R2.

Usa la API S3-compatible de Cloudflare R2.
"""

from __future__ import annotations

import logging

import boto3
from botocore.config import Config

from app.config import settings
from app.utils.ids import new_uuid

logger = logging.getLogger(__name__)


class R2Service:
    """Almacenamiento de artefactos en Cloudflare R2."""

    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region,
            config=Config(signature_version="s3v4"),
        )

        self.bucket_name = settings.r2_bucket_name

    def upload_bytes(
        self,
        content: bytes | bytearray,
        prefix: str,
        ext: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Sube un objeto a R2 y devuelve una URL prefirmada de lectura.

        Args:
            content: Contenido binario del archivo.
            prefix: Prefijo/ruta dentro del bucket.
            ext: Extensión del archivo sin punto.
            content_type: MIME type del objeto.
            expires_in: Duración de la URL en segundos.

        Returns:
            URL prefirmada para descargar el objeto.
        """
        if not content:
            raise ValueError("El contenido del artefacto no puede estar vacío.")

        if not isinstance(content, (bytes, bytearray)):
            raise TypeError(
                f"content debe ser bytes o bytearray, no {type(content).__name__}"
            )

        clean_prefix = prefix.strip("/")

        blob_name = (
            f"{clean_prefix}/{new_uuid()}.{ext.lstrip('.')}"
            if clean_prefix
            else f"{new_uuid()}.{ext.lstrip('.')}"
        )

        body = bytes(content)

        self.client.put_object(
            Bucket=self.bucket_name,
            Key=blob_name,
            Body=body,
            ContentType=content_type,
        )

        logger.info(
            "Artefacto subido a R2: bucket=%s key=%s size=%d",
            self.bucket_name,
            blob_name,
            len(body),
        )

        return self.client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": blob_name,
            },
            ExpiresIn=expires_in,
        )

    def upload_bytes_with_key(
        self,
        storage_key: str,
        content: bytes | bytearray,
        content_type: str,
    ) -> None:
        """
        Sube bytes a R2 bajo una storage_key explícita.

        A diferencia de upload_bytes(), no genera una key aleatoria
        ni devuelve una URL prefirmada: el caller controla la key.
        """

        if not content:
            raise ValueError(
                "El contenido del archivo no puede estar vacío."
            )

        if not isinstance(content, (bytes, bytearray)):
            raise TypeError(
                "content debe ser bytes o bytearray, "
                f"no {type(content).__name__}"
            )

        if not storage_key:
            raise ValueError(
                "storage_key no puede estar vacía."
            )

        body = bytes(content)

        self.client.put_object(
            Bucket=self.bucket_name,
            Key=storage_key,
            Body=body,
            ContentType=content_type,
        )

        logger.info(
            "Objeto subido a R2: bucket=%s key=%s size=%d",
            self.bucket_name,
            storage_key,
            len(body),
        )

    def generate_presigned_url(
        self,
        storage_key: str,
        expires_in: int = 3600,
    ) -> str:
        """Genera una URL prefirmada de lectura para una storage_key."""

        return self.client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": storage_key,
            },
            ExpiresIn=expires_in,
        )

    def delete_object(
        self,
        storage_key: str,
    ) -> None:
        """Elimina un objeto de R2 por su storage_key."""

        self.client.delete_object(
            Bucket=self.bucket_name,
            Key=storage_key,
        )

        logger.info(
            "Objeto eliminado de R2: bucket=%s key=%s",
            self.bucket_name,
            storage_key,
        )


def get_r2_service() -> R2Service:
    """Dependencia FastAPI: R2Service."""
    return R2Service()
