"""
conftest.py — Fixtures de pytest para la suite del backend.

Mockea los servicios externos (Firebase, DeepSeek, E2B, Pinecone/Qdrant)
para que las pruebas sean rápidas y deterministas.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Cliente de test sobre la app FastAPI."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_auth_service(monkeypatch):
    """Hace que `validate_token` acepte cualquier token como 'test-user'."""
    from app.core import security

    async def fake_validate(self, token: str) -> str:
        return "test-user"

    monkeypatch.setattr(security, "get_current_user", fake_validate)
    return "test-user"


@pytest.fixture
def mock_deepseek(monkeypatch):
    """Mockea el streaming de DeepSeek R1."""
    from app.services.deepseek import reasoning_service

    async def fake_stream(self, messages, max_tokens=4096):
        yield {"type": "delta", "content": "Respuesta simulada de prueba"}

    monkeypatch.setattr(reasoning_service.ReasoningService, "stream_reasoning", fake_stream)
