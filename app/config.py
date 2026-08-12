"""
config.py — Configuración central con pydantic-settings.

Carga variables desde el entorno o desde backend/.env.
Un único acceso `from app.config import settings`.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración global del backend Sapientia."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Entorno
    environment: str = Field(default="development")
    secret_key: str = Field(default="change-me")
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(default="http://localhost:3000")

    # Firebase
    firebase_project_id: str = ""
    firebase_private_key_id: str = ""
    firebase_private_key: str = ""
    firebase_client_email: str = ""
    google_client_id: str = ""
    firebase_service_account_path: str = Field(default="secrets/service-account.json")

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-reasoner"

    # Visión
    vision_model_endpoint: str = "http://localhost:8888/v1/chat/completions"
    vision_model_name: str = "janus-pro"
    vision_api_key: str = ""

    # RAG
    vector_store_backend: str = Field(default="pinecone")  # pinecone | qdrant
    pinecone_api_key: str = ""
    pinecone_index_name: str = "sapientia-pensums"
    pinecone_environment: str = "gcp-starter"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Sandbox
    e2b_api_key: str = ""
    e2b_template: str = "sapientia-python"

    # Storage
    gcs_bucket_name: str = "sapientia-artifacts"
    gcs_bucket_public: bool = True

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_task_queue: str = "sapientia"

    # Límites
    rate_limit_vision_per_min: int = 5
    rate_limit_graph_per_min: int = 3
    max_image_size_mb: int = 10

    # Propiedades derivadas
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def firebase_credentials_available(self) -> bool:
        """True si hay credenciales (env vars o archivo) para Firebase Admin."""
        return bool(
            self.firebase_project_id
            and self.firebase_client_email
            and self.firebase_private_key
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Acceso global de un solo import
settings = get_settings()
