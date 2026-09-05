"""
Pruebas unitarias del DeepSeekClient.

No realiza llamadas reales a DeepSeek.
"""

from __future__ import annotations

import pytest

import app.services.deepseek.client as client_module
from app.services.deepseek.client import DeepSeekClient


class FakeCompletionDetails:
    def __init__(self, reasoning_tokens=None):
        self.reasoning_tokens = reasoning_tokens


class FakeUsage:
    def __init__(
        self,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        prompt_cache_hit_tokens=None,
        prompt_cache_miss_tokens=None,
        reasoning_tokens=None,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.prompt_cache_hit_tokens = (
            prompt_cache_hit_tokens
        )
        self.prompt_cache_miss_tokens = (
            prompt_cache_miss_tokens
        )
        self.completion_tokens_details = (
            FakeCompletionDetails(
                reasoning_tokens=reasoning_tokens
            )
        )


class FakeDelta:
    def __init__(
        self,
        content=None,
        reasoning_content=None,
    ):
        self.content = content
        self.reasoning_content = reasoning_content


class FakeChoice:
    def __init__(
        self,
        content=None,
        reasoning_content=None,
        finish_reason=None,
    ):
        self.delta = FakeDelta(
            content=content,
            reasoning_content=reasoning_content,
        )

        self.finish_reason = finish_reason


class FakeChunk:
    def __init__(
        self,
        content=None,
        reasoning_content=None,
        usage=None,
        choices=True,
        finish_reason=None,
    ):
        self.choices = (
            [
                FakeChoice(
                    content=content,
                    reasoning_content=reasoning_content,
                    finish_reason=finish_reason,
                )
            ]
            if choices
            else []
        )

        self.usage = usage


class FakeCompletions:
    def __init__(self):
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs

        return self._stream()

    async def _stream(self):
        yield FakeChunk("Hola ")
        yield FakeChunk(None)
        yield FakeChunk("Sapientia.")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def create_fake_deepseek_client():
    client = DeepSeekClient()

    fake_client = FakeClient()

    client.client = fake_client

    return client, fake_client


@pytest.mark.asyncio
async def test_chat_stream_uses_expected_defaults():
    """Los valores por defecto deben ser high + 4096."""

    client, fake_client = (
        create_fake_deepseek_client()
    )

    chunks = []

    async for content in client.chat_stream(
        messages=[
            {
                "role": "user",
                "content": "Hola",
            }
        ]
    ):
        chunks.append(content)

    assert chunks == [
        "Hola ",
        "Sapientia.",
    ]

    request = (
        fake_client
        .chat
        .completions
        .request
    )

    assert request["max_tokens"] == 4096
    assert request["reasoning_effort"] == "high"
    assert request["stream"] is True

    assert request["extra_body"] == {
        "thinking": {
            "type": "enabled",
        }
    }


@pytest.mark.asyncio
async def test_chat_stream_accepts_max_reasoning():
    """Debe permitir elevar el razonamiento a max."""

    client, fake_client = (
        create_fake_deepseek_client()
    )

    chunks = []

    async for content in client.chat_stream(
        messages=[
            {
                "role": "user",
                "content": "Resuelve este problema.",
            }
        ],
        max_tokens=8192,
        reasoning_effort="max",
    ):
        chunks.append(content)

    assert chunks == [
        "Hola ",
        "Sapientia.",
    ]

    request = (
        fake_client
        .chat
        .completions
        .request
    )

    assert request["max_tokens"] == 8192
    assert request["reasoning_effort"] == "max"


@pytest.mark.asyncio
async def test_chat_stream_rejects_invalid_max_tokens():
    """No se permiten límites de tokens inválidos."""

    client, _ = create_fake_deepseek_client()

    with pytest.raises(ValueError):
        async for _ in client.chat_stream(
            messages=[],
            max_tokens=0,
        ):
            pass


@pytest.mark.asyncio
async def test_chat_stream_rejects_invalid_reasoning_effort():
    """Solo se admiten low, high y max."""

    client, _ = create_fake_deepseek_client()

    with pytest.raises(ValueError):
        async for _ in client.chat_stream(
            messages=[],
            reasoning_effort="invalid",
        ):
            pass



@pytest.mark.asyncio
async def test_chat_stream_accepts_low_reasoning_effort():
    """Flash debe poder utilizar reasoning_effort=low."""

    client, fake_client = (
        create_fake_deepseek_client()
    )

    chunks = []

    async for content in client.chat_stream(
        messages=[
            {
                "role": "user",
                "content": "Explica qué es una derivada.",
            }
        ],
        max_tokens=1536,
        reasoning_effort="low",
        model="deepseek-v4-flash",
    ):
        chunks.append(content)

    assert chunks == [
        "Hola ",
        "Sapientia.",
    ]

    request = (
        fake_client
        .chat
        .completions
        .request
    )

    assert request["reasoning_effort"] == "low"
    assert request["model"] == "deepseek-v4-flash"

@pytest.mark.asyncio
async def test_chat_stream_collects_latency_metrics():
    """Debe medir TTFT y tiempo total sin alterar el streaming."""

    client, _ = create_fake_deepseek_client()

    from app.services.deepseek.client import (
        StreamMetrics,
    )

    metrics = StreamMetrics()

    chunks = []

    async for content in client.chat_stream(
        messages=[
            {
                "role": "user",
                "content": "Hola",
            }
        ],
        metrics=metrics,
    ):
        chunks.append(content)

    assert chunks == [
        "Hola ",
        "Sapientia.",
    ]

    assert metrics.request_started_at is not None
    assert metrics.first_token_at is not None
    assert metrics.completed_at is not None

    assert metrics.ttft_seconds is not None
    assert metrics.total_seconds is not None
    assert metrics.stream_seconds is not None

    assert metrics.ttft_seconds >= 0
    assert metrics.stream_seconds >= 0
    assert metrics.total_seconds >= (
        metrics.ttft_seconds
    )


@pytest.mark.asyncio
async def test_each_stream_uses_independent_metrics():
    """
    Dos solicitudes no deben compartir el objeto de métricas.
    """

    client, _ = create_fake_deepseek_client()

    from app.services.deepseek.client import (
        StreamMetrics,
    )

    metrics_a = StreamMetrics()
    metrics_b = StreamMetrics()

    async for _ in client.chat_stream(
        messages=[
            {
                "role": "user",
                "content": "A",
            }
        ],
        metrics=metrics_a,
    ):
        pass

    async for _ in client.chat_stream(
        messages=[
            {
                "role": "user",
                "content": "B",
            }
        ],
        metrics=metrics_b,
    ):
        pass

    assert metrics_a is not metrics_b

    assert metrics_a.total_seconds is not None
    assert metrics_b.total_seconds is not None


@pytest.mark.asyncio
async def test_chat_stream_measures_reasoning_before_content():
    client = DeepSeekClient()
    fake_client = FakeClient()

    async def fake_stream():
        yield FakeChunk(
            reasoning_content="pensando..."
        )
        yield FakeChunk(
            reasoning_content="más razonamiento..."
        )
        yield FakeChunk(
            content="Respuesta final."
        )

    async def fake_create(**kwargs):
        return fake_stream()

    fake_client.chat.completions.create = fake_create

    client.client = fake_client

    from app.services.deepseek.client import (
        StreamMetrics,
    )

    metrics = StreamMetrics()

    chunks = []

    async for content in client.chat_stream(
        messages=[],
        metrics=metrics,
    ):
        chunks.append(content)

    assert chunks == [
        "Respuesta final.",
    ]

    assert (
        metrics.first_reasoning_at
        is not None
    )

    assert (
        metrics.first_token_at
        is not None
    )

    assert (
        metrics.time_to_first_reasoning_ms
        is not None
    )

    assert (
        metrics.reasoning_to_content_ms
        is not None
    )


@pytest.mark.asyncio
async def test_reasoning_content_is_not_yielded():
    client = DeepSeekClient()
    fake_client = FakeClient()

    async def fake_stream():
        yield FakeChunk(
            reasoning_content="contenido interno"
        )
        yield FakeChunk(
            content="Respuesta pública."
        )

    async def fake_create(**kwargs):
        return fake_stream()

    fake_client.chat.completions.create = fake_create

    client.client = fake_client

    output = []

    async for content in client.chat_stream(
        messages=[],
    ):
        output.append(content)

    assert output == [
        "Respuesta pública.",
    ]

    assert "contenido interno" not in output


@pytest.mark.asyncio
async def test_chat_stream_accepts_model_override():
    """Debe permitir seleccionar otro modelo por solicitud."""

    client, fake_client = (
        create_fake_deepseek_client()
    )

    chunks = []

    async for content in client.chat_stream(
        messages=[
            {
                "role": "user",
                "content": "Hola",
            }
        ],
        model="deepseek-v4-flash",
    ):
        chunks.append(content)

    assert chunks == [
        "Hola ",
        "Sapientia.",
    ]

    request = (
        fake_client
        .chat
        .completions
        .request
    )

    assert request["model"] == (
        "deepseek-v4-flash"
    )


@pytest.mark.asyncio
async def test_chat_stream_captures_token_usage():
    """
    DeepSeek puede devolver usage en un chunk final con
    choices vacío. El cliente debe conservar esa información.
    """

    from app.services.deepseek.client import (
        StreamMetrics,
    )

    client = DeepSeekClient()
    fake_client = FakeClient()

    async def fake_stream():
        yield FakeChunk(
            content="Respuesta ",
        )

        yield FakeChunk(
            content="final.",
        )

        yield FakeChunk(
            usage=FakeUsage(
                prompt_tokens=120,
                completion_tokens=80,
                total_tokens=200,
                prompt_cache_hit_tokens=90,
                prompt_cache_miss_tokens=30,
                reasoning_tokens=50,
            ),
            choices=False,
        )

    async def fake_create(**kwargs):
        return fake_stream()

    fake_client.chat.completions.create = fake_create

    client.client = fake_client

    metrics = StreamMetrics()

    output = []

    async for content in client.chat_stream(
        messages=[],
        metrics=metrics,
    ):
        output.append(content)

    assert output == [
        "Respuesta ",
        "final.",
    ]

    assert metrics.usage is not None

    assert metrics.usage.prompt_tokens == 120
    assert metrics.usage.completion_tokens == 80
    assert metrics.usage.total_tokens == 200

    assert (
        metrics.usage.prompt_cache_hit_tokens
        == 90
    )

    assert (
        metrics.usage.prompt_cache_miss_tokens
        == 30
    )

    assert (
        metrics.usage.reasoning_tokens
        == 50
    )


@pytest.mark.asyncio
async def test_chat_stream_requests_usage_statistics():
    """
    El cliente debe pedir explícitamente estadísticas de uso
    cuando utiliza streaming.
    """

    client, fake_client = (
        create_fake_deepseek_client()
    )

    async for _ in client.chat_stream(
        messages=[],
    ):
        pass

    request = (
        fake_client
        .chat
        .completions
        .request
    )

    assert request["stream_options"] == {
        "include_usage": True,
    }


@pytest.mark.asyncio
async def test_usage_does_not_affect_stream_output():
    """
    Registrar usage nunca debe alterar el contenido que recibe
    el usuario.
    """

    from app.services.deepseek.client import (
        StreamMetrics,
    )

    client = DeepSeekClient()
    fake_client = FakeClient()

    async def fake_stream():
        yield FakeChunk(
            content="Hola.",
        )

        yield FakeChunk(
            usage=FakeUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                reasoning_tokens=2,
            ),
            choices=False,
        )

    async def fake_create(**kwargs):
        return fake_stream()

    fake_client.chat.completions.create = fake_create

    client.client = fake_client

    metrics = StreamMetrics()

    output = []

    async for content in client.chat_stream(
        messages=[],
        metrics=metrics,
    ):
        output.append(content)

    assert output == ["Hola."]

    assert metrics.usage is not None
    assert metrics.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_chat_stream_can_disable_thinking():
    """Debe permitir desactivar thinking por solicitud."""

    client, fake_client = (
        create_fake_deepseek_client()
    )

    async for _ in client.chat_stream(
        messages=[],
        model="deepseek-v4-flash",
        max_tokens=1536,
        thinking_enabled=False,
    ):
        pass

    request = (
        fake_client
        .chat
        .completions
        .request
    )

    assert request["extra_body"] == {
        "thinking": {
            "type": "disabled",
        }
    }


@pytest.mark.asyncio
async def test_chat_stream_captures_stop_finish_reason():
    """
    Debe registrar finish_reason='stop' cuando DeepSeek
    termina normalmente.
    """

    from app.services.deepseek.client import (
        StreamMetrics,
    )

    client = DeepSeekClient()
    fake_client = FakeClient()

    async def fake_stream():
        yield FakeChunk(
            content="Respuesta.",
        )

        yield FakeChunk(
            content="",
            finish_reason="stop",
        )

        yield FakeChunk(
            usage=FakeUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            choices=False,
        )

    async def fake_create(**kwargs):
        return fake_stream()

    fake_client.chat.completions.create = fake_create
    client.client = fake_client

    metrics = StreamMetrics()

    output = []

    async for content in client.chat_stream(
        messages=[],
        metrics=metrics,
    ):
        output.append(content)

    assert output == [
        "Respuesta.",
    ]

    assert metrics.finish_reason == "stop"
    assert metrics.completed_normally is True
    assert metrics.truncated is False


@pytest.mark.asyncio
async def test_chat_stream_detects_length_finish_reason():
    """
    Debe detectar cuando DeepSeek termina por alcanzar
    el límite máximo de tokens.
    """

    client = DeepSeekClient()
    fake_client = FakeClient()

    async def fake_stream():
        yield FakeChunk(
            content="Respuesta incompleta...",
        )

        yield FakeChunk(
            content="",
            finish_reason="length",
        )

        yield FakeChunk(
            usage=FakeUsage(
                prompt_tokens=20,
                completion_tokens=100,
                total_tokens=120,
            ),
            choices=False,
        )

    async def fake_create(**kwargs):
        return fake_stream()

    fake_client.chat.completions.create = fake_create
    client.client = fake_client

    from app.services.deepseek.client import (
        StreamMetrics,
    )

    metrics = StreamMetrics()

    output = []

    async for content in client.chat_stream(
        messages=[],
        metrics=metrics,
    ):
        output.append(content)

    assert output == [
        "Respuesta incompleta...",
    ]

    assert metrics.finish_reason == "length"
    assert metrics.completed_normally is False
    assert metrics.truncated is True
