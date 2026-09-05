"""
Pruebas del ReasoningService de Sapientia.
"""

from __future__ import annotations

import pytest
from time import perf_counter

from app.services.deepseek.reasoning_service import (
    ReasoningService,
    SYSTEM_PROMPT,
)


class FakeDeepSeekClient:
    """Cliente DeepSeek falso para pruebas."""

    def __init__(self):
        self.messages = None

    async def _stream(self):
        yield "Hola "
        yield "desde Sapientia."

    async def chat_stream(
        self,
        messages,
        max_tokens=4096,
        reasoning_effort="high",
        model=None,
        thinking_enabled=True,
        metrics=None,
    ):
        self.messages = messages
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.model = model
        self.thinking_enabled = thinking_enabled
        self.metrics = metrics

        async for chunk in self._stream():

            if (
                self.metrics is not None
                and self.metrics.first_token_at is None
            ):
                self.metrics.first_token_at = perf_counter()

            yield chunk

        if self.metrics is not None:
            self.metrics.finish_reason = "stop"
            self.metrics.completed_at = perf_counter()


# ============================================================
# SYSTEM PROMPT
# ============================================================

def test_system_prompt_defines_sapientia_as_general_ai():
    """Sapientia ya no debe describirse únicamente como tutor."""

    assert "inteligencia artificial generalista" in (
        SYSTEM_PROMPT
    )

    # El prompt envuelve la frase en varias líneas; normalizamos
    # los espacios para verificar la intención con precisión.
    normalized = " ".join(SYSTEM_PROMPT.split())

    assert (
        "No debes limitarte a presentarte únicamente como "
        "un tutor universitario."
    ) in normalized


def test_system_prompt_requires_human_explanations():
    """La matemática debe acompañarse de explicación humana."""

    assert "LENGUAJE HUMANO" in SYSTEM_PROMPT
    assert "qué representa cada símbolo" in SYSTEM_PROMPT
    assert "LaTeX" in SYSTEM_PROMPT


def test_system_prompt_requires_3d_only_graphs():
    """El nuevo cerebro no debe solicitar gráficos 2D."""

    prompt_lower = SYSTEM_PROMPT.lower()

    assert "figura_3d.html" in SYSTEM_PROMPT
    assert "no generes una figura 2d" in prompt_lower

    # La única referencia a un archivo 2D debe ser para prohibirlo.
    assert "no solicites ni produzcas `figura_2d.png`" in (
        prompt_lower
    )


# ============================================================
# BUILD MESSAGES
# ============================================================

def test_build_messages_without_context():
    """Construye correctamente system + user."""

    service = ReasoningService(
        client=FakeDeepSeekClient()
    )

    messages = service.build_messages(
        user_message="¿Qué es una integral?",
    )

    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert messages[1]["content"] == (
        "¿Qué es una integral?"
    )


def test_build_messages_with_rag_and_vision():
    """Incluye correctamente RAG y visión."""

    service = ReasoningService(
        client=FakeDeepSeekClient()
    )

    messages = service.build_messages(
        user_message="Resuelve el ejercicio.",
        rag_context="Contexto de Ingeniería.",
        vision_text="x^2 + 2x + 1 = 0",
    )

    system = messages[0]["content"]

    assert "Contexto de Ingeniería." in system
    assert "x^2 + 2x + 1 = 0" in system


# ============================================================
# STREAMING
# ============================================================

@pytest.mark.asyncio
async def test_stream_reasoning():
    """El servicio transmite los deltas del cliente."""

    client = FakeDeepSeekClient()

    service = ReasoningService(
        client=client
    )

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message="Hola.",
    ):
        chunks.append(chunk)

    assert chunks == [
        {
            "type": "delta",
            "content": "Hola ",
        },
        {
            "type": "delta",
            "content": "desde Sapientia.",
        },
    ]

    assert client.messages is not None


# ============================================================
# GRAPH EXTRACTION
# ============================================================

def test_extract_graph_code_accepts_plotly_3d():
    """Detecta código de un gráfico 3D."""

    answer = """
Aquí tienes el gráfico:

```python
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=[1, 2],
            y=[2, 3],
            z=[3, 4],
        )
    ]
)

fig.write_html("figura_3d.html")
```
"""

    service = ReasoningService(
        client=FakeDeepSeekClient()
    )

    code = service.extract_graph_code(answer)

    assert code is not None
    assert "plotly" in code.lower()
    assert "figura_3d.html" in code


def test_extract_graph_code_rejects_plain_python():
    """No cualquier bloque Python debe convertirse en gráfico."""

    answer = """
x = 5
y = x + 2
print(y)

"""

    service = ReasoningService(
        client=FakeDeepSeekClient()
    )

    assert (
        service.extract_graph_code(answer)
        is None
    )


def test_extract_graph_code_empty_answer():
    """Una respuesta vacía no produce código."""

    service = ReasoningService(
        client=FakeDeepSeekClient()
    )

    assert (
        service.extract_graph_code("")
        is None
    )


# ============================================================
# MODEL ROUTING
# ============================================================

@pytest.mark.asyncio
async def test_simple_query_routes_to_flash():
    """
    Una consulta sencilla debe utilizar V4 Flash,
    high, un límite de 1536 tokens y thinking deshabilitado.
    """

    client = FakeDeepSeekClient()

    service = ReasoningService(
        client=client,
    )

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message="¿Qué es la fotosíntesis?",
    ):
        chunks.append(chunk)

    assert chunks

    assert client.model == (
        "deepseek-v4-flash"
    )

    assert client.reasoning_effort == "high"

    assert client.max_tokens == 1536

    assert client.thinking_enabled is False


@pytest.mark.asyncio
async def test_normal_query_routes_to_flash():
    """
    Una consulta normal debe utilizar V4 Flash,
    low, un límite de 1536 tokens y thinking habilitado.
    """

    client = FakeDeepSeekClient()

    service = ReasoningService(
        client=client,
    )

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message=(
            "Resuelve la ecuación "
            "2x^2 - 7x + 3 = 0."
        ),
    ):
        chunks.append(chunk)

    assert chunks

    assert client.model == (
        "deepseek-v4-flash"
    )

    assert client.reasoning_effort == "low"

    assert client.max_tokens == 1536

    assert client.thinking_enabled is True


@pytest.mark.asyncio
async def test_complex_query_routes_to_pro():
    """
    Una consulta compleja debe utilizar V4 Pro,
    high, un límite de 2048 tokens y thinking habilitado.
    """

    client = FakeDeepSeekClient()

    service = ReasoningService(
        client=client,
    )

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message=(
            "Resuelve la ecuación diferencial "
            "y demuestra la solución general."
        ),
    ):
        chunks.append(chunk)

    assert chunks

    assert client.model == (
        "deepseek-v4-pro"
    )

    assert client.reasoning_effort == "high"

    assert client.max_tokens == 2048

    assert client.thinking_enabled is True


@pytest.mark.asyncio
async def test_stream_reasoning_records_routing_metrics():
    """
    ReasoningService debe registrar en RequestMetrics
    la decisión de routing que realmente utilizó.
    """

    from app.core.metrics import RequestMetrics

    client = FakeDeepSeekClient()

    service = ReasoningService(
        client=client,
    )

    metrics = RequestMetrics()

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message="¿Qué es la fotosíntesis?",
        metrics=metrics,
    ):
        chunks.append(chunk)

    assert chunks

    assert metrics.difficulty == "simple"
    assert metrics.model == "deepseek-v4-flash"
    assert metrics.reasoning_effort == "high"
    assert metrics.max_tokens == 1536

    assert client.model == "deepseek-v4-flash"
    assert client.max_tokens == 1536
    assert client.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_complex_reasoning_records_pro_routing():
    """
    Las consultas complejas deben registrar Pro en las métricas.
    """

    from app.core.metrics import RequestMetrics

    client = FakeDeepSeekClient()

    service = ReasoningService(
        client=client,
    )

    metrics = RequestMetrics()

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message=(
            "Resuelve la ecuación diferencial "
            "y demuestra la solución general."
        ),
        metrics=metrics,
    ):
        chunks.append(chunk)

    assert chunks

    assert metrics.difficulty == "complex"
    assert metrics.model == "deepseek-v4-pro"
    assert metrics.reasoning_effort == "high"
    assert metrics.max_tokens == 2048

    assert client.model == "deepseek-v4-pro"
    assert client.max_tokens == 2048
    assert client.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_stream_reasoning_records_ok_budget_verdict():
    """
    Una generación normal debe producir un verdict OK
    y registrarlo en RequestMetrics.
    """

    from app.core.metrics import RequestMetrics

    client = FakeDeepSeekClient()

    service = ReasoningService(
        client=client,
    )

    metrics = RequestMetrics()

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message="¿Qué es la fotosíntesis?",
        metrics=metrics,
    ):
        chunks.append(chunk)

    assert chunks

    assert metrics.budget_verdict == "ok"


@pytest.mark.asyncio
async def test_stream_reasoning_records_truncated_budget_verdict():
    """
    Una generación terminada por length debe registrarse
    como truncada.
    """

    from app.core.metrics import RequestMetrics

    class TruncatedFakeClient(
        FakeDeepSeekClient
    ):
        async def chat_stream(
            self,
            messages,
            max_tokens=4096,
            reasoning_effort="high",
            model=None,
            thinking_enabled=True,
            metrics=None,
        ):
            self.messages = messages
            self.max_tokens = max_tokens
            self.reasoning_effort = reasoning_effort
            self.model = model
            self.thinking_enabled = thinking_enabled
            self.metrics = metrics

            if metrics is not None:
                metrics.first_token_at = perf_counter()

            yield "Respuesta incompleta."

            if metrics is not None:
                metrics.finish_reason = "length"
                metrics.completed_at = perf_counter()

    client = TruncatedFakeClient()

    service = ReasoningService(
        client=client,
    )

    metrics = RequestMetrics()

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message="¿Qué es la fotosíntesis?",
        metrics=metrics,
    ):
        chunks.append(chunk)

    assert chunks == [
        {
            "type": "delta",
            "content": "Respuesta incompleta.",
        }
    ]

    assert metrics.budget_verdict == "truncated"


@pytest.mark.asyncio
async def test_stream_reasoning_records_no_content_budget_verdict():
    """
    Una generación que termina normalmente pero no produce
    contenido visible debe registrarse como no_content.
    """

    from time import perf_counter

    from app.core.metrics import RequestMetrics

    class NoContentFakeClient(FakeDeepSeekClient):
        async def chat_stream(
            self,
            messages,
            max_tokens=4096,
            reasoning_effort="high",
            model=None,
            thinking_enabled=True,
            metrics=None,
        ):
            self.messages = messages
            self.max_tokens = max_tokens
            self.reasoning_effort = reasoning_effort
            self.model = model
            self.thinking_enabled = thinking_enabled
            self.metrics = metrics

            # No producimos ningún delta de contenido.
            if metrics is not None:
                metrics.finish_reason = "stop"
                metrics.completed_at = perf_counter()

            return
            yield  # Hace que la función siga siendo async generator.

    client = NoContentFakeClient()

    service = ReasoningService(
        client=client,
    )

    metrics = RequestMetrics()

    chunks = []

    async for chunk in service.stream_reasoning(
        user_message="Genera una respuesta.",
        metrics=metrics,
    ):
        chunks.append(chunk)

    assert chunks == []

    assert metrics.budget_verdict == (
        "no_content"
    )
