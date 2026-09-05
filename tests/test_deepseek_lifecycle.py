"""
Pruebas de reutilización del cliente DeepSeek.
"""

import pytest

from app.main import app
from app.services.deepseek.client import (
    DeepSeekClient,
    get_deepseek_client,
)
from app.services.deepseek.reasoning_service import (
    get_reasoning_service,
)


def test_deepseek_client_is_reused():
    """El dependency provider devuelve la misma instancia por proceso."""

    first = get_deepseek_client()
    second = get_deepseek_client()

    assert first is second


def test_reasoning_service_uses_shared_client():
    """ReasoningService debe utilizar el DeepSeekClient reutilizable."""

    reasoning = get_reasoning_service()
    client = get_deepseek_client()

    assert reasoning.client is client


@pytest.mark.asyncio
async def test_deepseek_client_close_delegates_to_async_client(
    monkeypatch,
):
    """close() delega en el cliente asíncrono subyacente."""

    client = DeepSeekClient()

    closed = False

    async def fake_close():
        nonlocal closed
        closed = True

    monkeypatch.setattr(
        client.client,
        "close",
        fake_close,
    )

    await client.close()

    assert closed is True


def test_deepseek_client_cache_can_be_cleared():
    """Tras limpiar la cache, se crea una instancia nueva."""

    first = get_deepseek_client()

    get_deepseek_client.cache_clear()

    second = get_deepseek_client()

    assert first is not second

    get_deepseek_client.cache_clear()


@pytest.mark.asyncio
async def test_fastapi_lifespan_clears_deepseek_client_cache(
    monkeypatch,
):
    client = get_deepseek_client()

    closed = False

    async def fake_close():
        nonlocal closed
        closed = True

    monkeypatch.setattr(
        client,
        "close",
        fake_close,
    )

    async with app.router.lifespan_context(app):
        assert (
            get_deepseek_client()
            is client
        )

    assert closed is True

    # El shutdown debe limpiar el cache.
    replacement = get_deepseek_client()

    assert replacement is not client

    # Limpieza para no contaminar otros tests.
    get_deepseek_client.cache_clear()
