"""
vision_service.py — Servicio modular de análisis de imágenes.

Responsabilidades:
- validar la imagen;
- delegar OCR/LaTeX a SimpleTexService;
- transformar el resultado al contrato VisionAnalysisResult.

No llama a DeepSeek.
No ejecuta E2B.
No guarda Firestore.
No genera gráficos.
"""

from __future__ import annotations

from app.api.v1.schemas.vision import VisionAnalysisResult
from app.config import settings
from app.services.vision.simpletex_service import (
    SimpleTexError,
    SimpleTexService,
    get_simpletex_service,
)


class VisionServiceError(RuntimeError):
    """Error controlado durante el análisis de visión."""


class VisionService:
    """Servicio de análisis inicial de imágenes."""

    def __init__(
        self,
        simpletex: SimpleTexService | None = None,
    ) -> None:
        self.simpletex = (
            simpletex or get_simpletex_service()
        )

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str = "imagen.png",
        content_type: str = "application/octet-stream",
    ) -> VisionAnalysisResult:
        """
        Valida la imagen, ejecuta OCR/LaTeX mediante SimpleTex
        y devuelve un VisionAnalysisResult.
        """

        self._validate_image(
            image_bytes=image_bytes,
            content_type=content_type,
        )

        try:
            result = await self.simpletex.analyze(
                image_bytes=image_bytes,
                filename=filename,
                content_type=content_type,
            )
        except SimpleTexError as exc:
            raise VisionServiceError(
                f"Error en el análisis OCR: {exc}"
            ) from exc
        except Exception as exc:
            raise VisionServiceError(
                "Error inesperado durante el análisis de imagen."
            ) from exc

        text = (
            result.get("text") or ""
        ).strip()

        latex = (
            result.get("latex") or ""
        ).strip()

        formulas: list[str] = []

        if latex:
            formulas.append(latex)

        return VisionAnalysisResult(
            texto=text,
            formulas=formulas,
            tipo="otro",
            descripcion="",
        )

    @staticmethod
    def _validate_image(
        *,
        image_bytes: bytes,
        content_type: str,
    ) -> None:
        """Valida tamaño y tipo MIME de la imagen."""

        if not image_bytes:
            raise VisionServiceError(
                "La imagen está vacía."
            )

        normalized_content_type = (
            content_type or ""
        ).lower()

        if not normalized_content_type.startswith(
            "image/"
        ):
            raise VisionServiceError(
                "El archivo debe ser una imagen."
            )

        max_bytes = (
            settings.max_image_size_mb
            * 1024
            * 1024
        )

        if len(image_bytes) > max_bytes:
            raise VisionServiceError(
                "La imagen supera el tamaño máximo permitido "
                f"de {settings.max_image_size_mb} MB."
            )


def get_vision_service() -> VisionService:
    """Dependencia para FastAPI."""

    return VisionService()
