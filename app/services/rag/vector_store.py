"""
vector_store.py — Abstracción sobre Pinecone/Qdrant.

`get_vector_store()` devuelve la implementación según
`settings.vector_store_backend` (pinecone | qdrant).
"""
from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Interfaz común para upsert y query de vectores."""

    def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        """Inserta/actualiza vectores con su metadata."""
        raise NotImplementedError

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_: dict | None = None,
    ) -> list[dict]:
        """Busca los top_k vecinos más cercanos con filtro opcional."""
        raise NotImplementedError


class PineconeStore(VectorStore):
    """Implementación con Pinecone (índice sapientia-pensums)."""

    def __init__(self) -> None:
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)

    def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        records = [
            {"id": id_, "values": vec, "metadata": meta}
            for id_, vec, meta in zip(ids, vectors, metadata, strict=True)
        ]
        self.index.upsert(vectors=records)

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_: dict | None = None,
    ) -> list[dict]:
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            filter_=filter_,
            include_metadata=True,
        )
        return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in response.matches]


class QdrantStore(VectorStore):
    """Implementación con Qdrant (local vía Docker o cloud)."""

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._models = models
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.collection = settings.pinecone_index_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self._models.VectorParams(
                    size=settings.embedding_dim, distance=self._models.Distance.COSINE
                ),
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        points = [
            self._models.PointStruct(
                id=id_, vector=vec, payload={**meta, "_id": id_}
            )
            for id_, vec, meta in zip(ids, vectors, metadata, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_: dict | None = None,
    ) -> list[dict]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            query_filter=self._payload_filter(filter_) if filter_ else None,
            with_payload=True,
        )
        return [
            {"id": p.id, "score": p.score, "metadata": p.payload}
            for p in result.points
        ]

    @staticmethod
    def _payload_filter(filter_: dict):
        from qdrant_client.http import models

        conditions = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filter_.items()
        ]
        return models.Filter(must=conditions)


def get_vector_store() -> VectorStore:
    """Fábrica según configuración."""
    if settings.vector_store_backend == "qdrant":
        return QdrantStore()
    return PineconeStore()
