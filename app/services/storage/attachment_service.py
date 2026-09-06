"""attachment_service.py — Subida y metadata de adjuntos (R2 + Firestore)."""
from __future__ import annotations

import logging

from app.config import settings
from app.services.firebase.firestore_service import FirestoreService
from app.services.storage.r2_service import R2Service
from app.utils.ids import new_id, new_uuid

logger = logging.getLogger(__name__)


ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "application/pdf",
    "text/plain",
}

MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "application/pdf": "pdf",
    "text/plain": "txt",
}

EXTENSION_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}

IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}


class AttachmentServiceError(RuntimeError):
    """Error controlado del servicio de adjuntos."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class AttachmentService:
    """Sube adjuntos a R2 y guarda su metadata en Firestore."""

    def __init__(
        self,
        r2: R2Service,
        firestore: FirestoreService,
    ) -> None:
        self.r2 = r2
        self.firestore = firestore

    def _max_size_bytes(self, content_type: str) -> int:
        if content_type in IMAGE_CONTENT_TYPES:
            return min(
                settings.max_image_size_mb,
                settings.max_attachment_size_mb,
            ) * 1024 * 1024

        return settings.max_attachment_size_mb * 1024 * 1024

    def _validate(
        self,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> None:
        if not content:
            raise AttachmentServiceError(
                "El archivo no puede estar vacío."
            )

        if content_type not in ALLOWED_CONTENT_TYPES:
            raise AttachmentServiceError(
                "Tipo de archivo no permitido."
            )

        # Si el archivo tiene extensión, no debe contradecir el MIME.
        dot = filename.rfind(".")
        if dot != -1:
            ext = filename[dot:].lower()
            expected = EXTENSION_TO_MIME.get(ext)
            if expected is not None and expected != content_type:
                raise AttachmentServiceError(
                    "La extensión no coincide con el tipo de archivo."
                )

        if len(content) > self._max_size_bytes(content_type):
            raise AttachmentServiceError(
                "El archivo supera el tamaño máximo permitido.",
                status_code=413,
            )

    async def create_attachment(
        self,
        user_id: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> dict:
        """Valida, sube a R2 y guarda metadata. Devuelve la metadata."""

        cleaned_filename = filename.strip()

        if not cleaned_filename:
            raise AttachmentServiceError(
                "El nombre del archivo no puede estar vacío."
            )

        content_type = (content_type or "").strip().lower()

        self._validate(cleaned_filename, content_type, content)

        attachment_id = new_id()
        ext = MIME_TO_EXT[content_type]
        storage_key = f"attachments/{user_id}/{new_uuid()}.{ext}"

        # 1) Subir a R2.
        try:
            self.r2.upload_bytes_with_key(
                storage_key=storage_key,
                content=content,
                content_type=content_type,
            )
        except Exception as exc:
            logger.error(
                "No se pudo subir el attachment a R2: %s",
                exc,
            )
            raise

        # 2) Guardar metadata en Firestore (con rollback de R2 si falla).
        try:
            return await self.firestore.create_attachment_metadata(
                attachment_id=attachment_id,
                user_id=user_id,
                filename=cleaned_filename,
                content_type=content_type,
                size=len(content),
                storage_key=storage_key,
            )
        except Exception as exc:
            logger.error(
                "No se pudo guardar la metadata del attachment: %s",
                exc,
            )

            try:
                self.r2.delete_object(storage_key)
            except Exception as rollback_exc:
                logger.error(
                    "Rollback R2 falló para %s: %s",
                    storage_key,
                    rollback_exc,
                )

            raise

    async def get_attachment(
        self,
        attachment_id: str,
        user_id: str,
    ) -> dict | None:
        """Devuelve la metadata de un adjunto (con verificación de ownership)."""

        return await self.firestore.get_user_attachment(
            attachment_id,
            user_id,
        )
