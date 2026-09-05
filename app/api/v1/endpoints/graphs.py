"""
graphs.py — API V1 para generación y consulta de gráficos.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.api.v1.schemas.graph import (
    GraphAccepted,
    GraphArtifactOut,
    GraphRequest,
)
from app.core.security import get_current_user
from app.services.firebase.firestore_service import (
    FirestoreService,
    get_firestore_service,
)
from app.services.graphs.graph_service import (
    GraphService,
    GraphServiceError,
)


router = APIRouter()


def get_graph_service(
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
) -> GraphService:
    return GraphService(
        firestore=firestore,
    )


@router.post(
    "",
    response_model=GraphAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_graph(
    request: GraphRequest,
    uid: str = Depends(get_current_user),
    graph_service: GraphService = Depends(
        get_graph_service
    ),
) -> GraphAccepted:
    """
    Crea un GraphArtifact y encola su generación.

    El worker requiere código Python, por lo que en esta etapa
    code debe estar presente.
    """

    try:
        result = await graph_service.create_graph(
            user_id=uid,
            conversation_id=request.conversation_id,
            code=request.code,
        )

    except GraphServiceError as exc:
        if (
            "código Python"
            in str(exc)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return GraphAccepted(
        status="accepted",
        task_id=result.task_id,
        artifact_id=result.artifact_id,
    )


@router.get(
    "/{graph_id}",
    response_model=GraphArtifactOut,
)
async def get_graph(
    graph_id: str,
    uid: str = Depends(get_current_user),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
) -> GraphArtifactOut:
    """
    Consulta un GraphArtifact perteneciente al usuario autenticado.
    """

    artifact = await firestore.get_graph_artifact(
        graph_id
    )

    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gráfico no encontrado.",
        )

    if artifact.get("user_id") != uid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gráfico no encontrado.",
        )

    return GraphArtifactOut(
        artifact_id=str(
            artifact.get("artifact_id") or graph_id
        ),
        status=artifact.get(
            "status",
            "pending",
        ),
        png_url=artifact.get(
            "png_url",
            "",
        ),
        html_url=artifact.get(
            "html_url",
            "",
        ),
        error=artifact.get(
            "error",
            "",
        ),
    )

