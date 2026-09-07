"""attachments.py — Subida de archivos adjuntos."""
from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.api.v1.schemas.attachments import (
    AttachmentOut,
    AttachmentUrlOut,
)
from app.core.security import get_current_user
from app.services.firebase.firestore_service import (
    FirestoreService,
    get_firestore_service,
)
from app.services.storage.attachment_service import (
    AttachmentService,
    AttachmentServiceError,
)
from app.services.storage.r2_service import (
    R2Service,
    get_r2_service,
)

router = APIRouter()


def get_attachment_service(
    r2: R2Service = Depends(get_r2_service),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
) -> AttachmentService:
    return AttachmentService(r2=r2, firestore=firestore)


@router.post(
    "",
    response_model=AttachmentOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachment(
    file: UploadFile = File(...),
    uid: str = Depends(get_current_user),
    attachment_service: AttachmentService = Depends(
        get_attachment_service
    ),
):
    """Sube un archivo adjunto y devuelve su metadata."""

    content = await file.read()

    try:
        metadata = await attachment_service.create_attachment(
            user_id=uid,
            filename=file.filename or "",
            content_type=file.content_type or "",
            content=content,
        )
    except AttachmentServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    return metadata


@router.get(
    "/{attachment_id}/url",
    response_model=AttachmentUrlOut,
)
async def get_attachment_download_url(
    attachment_id: str,
    uid: str = Depends(get_current_user),
    attachment_service: AttachmentService = Depends(
        get_attachment_service
    ),
):
    """Devuelve una URL temporal de descarga de un adjunto autenticado."""

    data = await attachment_service.get_attachment_url(
        attachment_id,
        uid,
    )

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Adjunto no encontrado.",
        )

    return data
