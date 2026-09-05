"""
chat.py — Chat principal de Sapientia con streaming SSE.

POST /api/v1/chat/message

Flujo:

1. Autenticación del usuario.
2. Crear o recuperar conversación.
3. Guardar mensaje del usuario.
4. Construir memoria contextual.
5. Ejecutar ReasoningService.
6. Emitir respuesta mediante SSE.
7. Guardar la respuesta completa en Firestore.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.v1.schemas.chat import ChatMessageRequest
from app.core.metrics import RequestMetrics
from app.core.security import get_current_user
from app.services.deepseek.reasoning_service import (
    ReasoningService,
    get_reasoning_service,
)
from app.services.firebase.firestore_service import (
    FirestoreService,
    get_firestore_service,
)
from app.services.graphs.graph_service import (
    GraphService,
    GraphServiceError,
)
from app.services.memory.memory_service import MemoryService


router = APIRouter()

logger = logging.getLogger(__name__)


# ============================================================
# DEPENDENCY
# ============================================================

def get_memory_service(
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
) -> MemoryService:
    """
    Construye MemoryService utilizando el mismo FirestoreService
    de la petición.
    """

    return MemoryService(
        firestore=firestore,
    )


def get_graph_service(
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
) -> GraphService:
    return GraphService(
        firestore=firestore,
    )


# ============================================================
# SSE
# ============================================================

def _sse_event(
    event_type: str,
    content: str = "",
) -> str:
    """Serializa un evento SSE."""

    payload = {
        "type": event_type,
        "content": content,
    }

    return (
        f"data: "
        f"{json.dumps(payload, ensure_ascii=False)}"
        f"\n\n"
    )


# ============================================================
# CHAT
# ============================================================

@router.post("/message")
async def send_message(
    payload: ChatMessageRequest,
    uid: str = Depends(get_current_user),
    firestore: FirestoreService = Depends(
        get_firestore_service
    ),
    memory: MemoryService = Depends(
        get_memory_service
    ),
    reasoning: ReasoningService = Depends(
        get_reasoning_service
    ),
    graph_service: GraphService = Depends(
        get_graph_service
    ),
):
    """
    Envía un mensaje y devuelve Server-Sent Events.

    La conversación se crea automáticamente cuando
    `conversation_id` está vacío.
    """

    message = payload.message.strip()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El mensaje no puede estar vacío.",
        )

    metrics = RequestMetrics()

    # ========================================================
    # 1. CREAR / RECUPERAR CONVERSACIÓN
    # ========================================================

    conversation_started_at = perf_counter()

    conversation_id = payload.conversation_id.strip()

    if conversation_id:
        conversation = (
            await firestore.get_user_conversation(
                conversation_id=conversation_id,
                user_id=uid,
            )
        )

        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversación no encontrada.",
            )

    else:
        title = message[:80]

        conversation_id = (
            await firestore.create_conversation(
                user_id=uid,
                title=title,
            )
        )

    metrics.mark_phase(
        "conversation",
        conversation_started_at,
    )

    # ========================================================
    # 2. GUARDAR MENSAJE DEL USUARIO
    # ========================================================

    user_message = {
        "role": "user",
        "content": message,
        "attachments": payload.attachment_ids,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    message_persist_started_at = perf_counter()

    await firestore.add_message(
        conversation_id=conversation_id,
        message=user_message,
    )

    metrics.mark_phase(
        "user_message_persist",
        message_persist_started_at,
    )

    # ========================================================
    # 3. CONSTRUIR MEMORIA
    # ========================================================

    memory_started_at = perf_counter()

    memory_context = await memory.build_context(
        user_id=uid,
        current_conversation_id=conversation_id,
    )

    metrics.mark_phase(
        "memory",
        memory_started_at,
    )

    # ========================================================
    # 4. STREAMING
    # ========================================================

    async def event_stream():
        assistant_parts: list[str] = []

        try:
            async for chunk in reasoning.stream_reasoning(
                user_message=message,
                vision_text=payload.vision_text,
                memory_context=memory_context,
                metrics=metrics,
            ):
                if not chunk:
                    continue

                content = chunk.get(
                    "content",
                    "",
                )

                if not content:
                    continue

                metrics.mark_first_token()

                assistant_parts.append(content)

                yield _sse_event(
                    event_type="answer",
                    content=content,
                )

            # ------------------------------------------------
            # 5. RESPUESTA COMPLETA
            # ------------------------------------------------

            answer = "".join(
                assistant_parts
            ).strip()

            if not answer:
                yield _sse_event(
                    event_type="error",
                    content=(
                        "Sapientia no devolvió "
                        "contenido."
                    ),
                )
                return

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "attachments": [],
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            assistant_persist_started_at = perf_counter()

            await firestore.add_message(
                conversation_id=conversation_id,
                message=assistant_message,
            )

            metrics.mark_phase(
                "assistant_message_persist",
                assistant_persist_started_at,
            )

            # ------------------------------------------------
            # 6. GRAPH REQUEST
            # ------------------------------------------------

            graph_code = reasoning.extract_graph_code(
                answer
            )

            graph_requested = graph_code is not None
            graph_artifact_id = None
            graph_task_id = None

            if graph_code is not None:
                try:
                    graph_result = (
                        await graph_service.create_graph(
                            user_id=uid,
                            conversation_id=conversation_id,
                            code=graph_code,
                        )
                    )

                    graph_artifact_id = (
                        graph_result.artifact_id
                    )
                    graph_task_id = (
                        graph_result.task_id
                    )

                    # Conservamos el evento existente.
                    yield _sse_event(
                        event_type="graph_request",
                        content="true",
                    )

                    # Nuevo evento con los identificadores
                    # necesarios para consultar el resultado.
                    yield _sse_event(
                        event_type="graph_created",
                        content=json.dumps(
                            {
                                "artifact_id": graph_artifact_id,
                                "task_id": graph_task_id,
                            },
                            ensure_ascii=False,
                        ),
                    )

                except GraphServiceError as exc:
                    yield _sse_event(
                        event_type="error",
                        content=str(exc),
                    )
                    return

            # ------------------------------------------------
            # 7. FIN
            # ------------------------------------------------

            metrics.finish()

            logger.info(
                "Sapientia chat metrics | conversation_id=%s | metrics=%s",
                conversation_id,
                metrics.summary(),
            )

            yield _sse_event(
                event_type="done",
                content=json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "graph_requested": graph_requested,
                        "graph_artifact_id": graph_artifact_id,
                        "graph_task_id": graph_task_id,
                    },
                    ensure_ascii=False,
                ),
            )

        except Exception as exc:
            yield _sse_event(
                event_type="error",
                content=str(exc),
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )