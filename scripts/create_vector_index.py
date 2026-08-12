"""
create_vector_index.py — Crea el índice vectorial si no existe.

Para Pinecone: crea el índice `sapientia-pensums` (dimensión 384).
Para Qdrant: la colección se crea automáticamente en vector_store.py.
"""
from __future__ import annotations

import argparse
import logging
import sys

from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Crea el índice vectorial de Sapientia.")
    args = parser.parse_args()

    if settings.vector_store_backend == "qdrant":
        logger.info("Qdrant crea la colección automáticamente (dimensión %s). Nada que hacer.", settings.embedding_dim)
        return 0

    if settings.vector_store_backend != "pinecone":
        logger.error("Backend desconocido: %s", settings.vector_store_backend)
        return 1

    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = {i.name for i in pc.list_indexes()}

    if settings.pinecone_index_name in existing:
        logger.info("El índice %s ya existe.", settings.pinecone_index_name)
        return 0

    logger.info("Creando índice %s (dimensión %s)...", settings.pinecone_index_name, settings.embedding_dim)
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dim,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": settings.pinecone_environment}},
    )
    logger.info("Índice creado correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
