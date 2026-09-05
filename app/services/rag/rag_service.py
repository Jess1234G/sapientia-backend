"""
rag_service.py — Recuperación de contexto RAG.
"""

from __future__ import annotations

import logging

from app.services.rag.embeddings import (
    EmbeddingService,
    get_embedding_service,
)
from app.services.rag.vector_store import (
    VectorStore,
    get_vector_store,
)

logger = logging.getLogger(__name__)


class RagService:
    """Recupera contexto de pensum desde el vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embeddings: EmbeddingService | None = None,
    ) -> None:
        self.vector_store = (
            vector_store or get_vector_store()
        )
        self.embeddings = (
            embeddings or get_embedding_service()
        )

    def query_context(
        self,
        pregunta: str,
        carrera: str,
        semestre: int,
        top_k: int = 5,
    ) -> str:
        """Devuelve texto con los fragmentos relevantes del pensum."""

        vector = self.embeddings.embed_text(pregunta)

        results = self.vector_store.query(
            vector=vector,
            top_k=top_k,
            filter_={
                "carrera": carrera,
                "semestre": semestre,
            },
        )

        if not results:
            return ""

        chunks = []

        for idx, hit in enumerate(
            results,
            start=1,
        ):
            meta = hit.get(
                "metadata",
                {},
            )

            chunks.append(
                f"[{idx}] "
                f"{meta.get('asignatura', 'Asignatura')} "
                f"(semestre "
                f"{meta.get('semestre', '?')}):\n"
                f"{meta.get('contenido', '')}"
            )

        return "\n\n".join(chunks)


def get_rag_service() -> RagService:
    """Dependencia FastAPI: RagService."""

    return RagService()

