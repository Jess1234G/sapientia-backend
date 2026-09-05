"""
conftest.py — Fixtures de pytest para la suite del backend.

Los servicios externos se mockean para que las pruebas sean:
- rápidas
- deterministas
- independientes de APIs externas
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


# ============================================================
# CLIENTE FASTAPI
# ============================================================

@pytest.fixture(scope="session")
def client() -> TestClient:
    """Cliente de test sobre la aplicación FastAPI."""

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


# ============================================================
# AUTENTICACIÓN — ARQUITECTURA MODULAR
# ============================================================

@pytest.fixture
def mock_auth_service(monkeypatch):
    """
    Mock para el sistema de autenticación modular.

    Hace que la validación de token devuelva el UID de prueba.
    """

    from app.core import security

    async def fake_validate(self, token: str) -> str:
        return "test-user"

    monkeypatch.setattr(
        security,
        "get_current_user",
        fake_validate,
    )

    return "test-user"


# ============================================================
# DEEPSEEK — ARQUITECTURA MODULAR
# ============================================================

@pytest.fixture
def mock_deepseek(monkeypatch):
    """
    Mockea el ReasoningService de la arquitectura modular.

    Se conserva para las pruebas futuras de /chat/message.
    """

    from app.services.deepseek import reasoning_service

    async def fake_stream(
        self,
        messages,
        max_tokens=4096,
    ):
        yield {
            "type": "delta",
            "content": "Respuesta simulada de prueba",
        }

    monkeypatch.setattr(
        reasoning_service.ReasoningService,
        "stream_reasoning",
        fake_stream,
    )


# ============================================================
# DEEPSEEK — BACKEND MONOLÍTICO ACTUAL
# ============================================================

@pytest.fixture
def mock_legacy_deepseek(monkeypatch):
    """
    Mockea el cliente DeepSeek utilizado actualmente por app.main.

    El backend monolítico llama directamente a:

        deepseek_client.chat.completions.create(...)

    y posteriormente obtiene:

        response.choices[0].message.content
    """

    import app.main as main_module

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Respuesta simulada de Sapientia para pruebas."
                ),
                finish_reason="stop",
            )
        ]
    )

    def fake_create(*args, **kwargs):
        return fake_response

    monkeypatch.setattr(
        main_module.deepseek_client.chat.completions,
        "create",
        fake_create,
    )

    # Evita que la prueba dependa de una API key real.
    monkeypatch.setattr(
        main_module,
        "DEEPSEEK_API_KEY",
        "test-deepseek-key",
    )

    # Evita llamadas reales a Pinecone/RAG.
    monkeypatch.setattr(
        main_module,
        "index",
        None,
    )

    # Evita llamadas reales a Tavily.
    monkeypatch.setattr(
        main_module,
        "tavily_client",
        None,
    )

    return fake_response


# ============================================================
# VISION — PREPARACIÓN PARA PRUEBAS FUTURAS
# ============================================================

@pytest.fixture
def mock_vision_environment(monkeypatch):
    """
    Desactiva dependencias externas de visión para pruebas.

    Se utilizará cuando implementemos las pruebas reales de
    /preguntar-vision.
    """

    import app.main as main_module

    monkeypatch.setattr(
        main_module,
        "DEEPSEEK_API_KEY",
        "test-deepseek-key",
    )

    return True