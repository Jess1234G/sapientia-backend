"""
vision.py — POST /vision/analyze

Recibe una imagen autenticada y delega el análisis a VisionService.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.api.v1.schemas.vision import VisionAnalysisResult
from app.core.security import get_current_user
from app.services.vision.vision_service import (
    VisionService,
    VisionServiceError,
    get_vision_service,
)


router = APIRouter()


@router.post(
    "/analyze",
    response_model=VisionAnalysisResult,
)
async def analyze_image(
    image: UploadFile = File(...),
    uid: str = Depends(get_current_user),
    vision: VisionService = Depends(get_vision_service),
) -> VisionAnalysisResult:
    """
    Analiza una imagen mediante VisionService.
    """

    del uid

    try:
        image_bytes = await image.read()

        return await vision.analyze(
            image_bytes=image_bytes,
            filename=image.filename or "imagen.png",
            content_type=(
                image.content_type
                or "application/octet-stream"
            ),
        )

    except VisionServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

