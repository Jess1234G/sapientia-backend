"""
vision.py — POST /vision/analyze (multipart imagen → OCR/LaTeX).

Recibe la imagen del usuario, la envía al modelo de visión
(DeepSeek-VL / Janus-Pro) y devuelve texto + fórmulas LaTeX
+ estructura descrita para el pipeline de razonamiento.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.security import get_current_user

router = APIRouter()


@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    # uid: str = Depends(get_current_user),
):
    """
    Analiza una imagen subida (multipart).

    - Valida tipo MIME y tamaño (límite configurable).
    - Encola tarea asíncrona `analyze_image_task` (Celery).
    - Devuelve `task_id` o el resultado si se procesa en línea.
    """
    # TODO: validar MIME/size → vision_service.analyze() → guardar adjunto
    return {
        "filename": image.filename,
        "detail": "Endpoint pendiente de implementación (OCR/LaTeX)",
    }
