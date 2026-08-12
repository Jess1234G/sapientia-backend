"""
embeddings.py — Generación de embeddings (dimensión 384).

Usa sentence-transformers con un modelo ligero (p.ej. all-MiniLM-L6-v2,
384 dimensiones) para vectorizar fragmentos de pensum.
"""
from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Servicio de embeddings de textos."""

    def __init__(self) -> None:
        # Import perezoso: la descarga del modelo es pesada y solo ocurre
        # cuando el servicio se usa por primera vez.
        from sentence_transformers import SentenceTransformer

        self._model: SentenceTransformer = SentenceTransformer(settings.embedding_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Devuelve lista de vectores (list[float]) para cada texto."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_text(self, text: str) -> list[float]:
        """Devuelve el vector de un único texto."""
        return self.embed_texts([text])[0]


def get_embedding_service() -> EmbeddingService:
    """Dependencia FastAPI: EmbeddingService singleton."""
    return EmbeddingService()
