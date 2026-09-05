"""
reasoning_service.py — Núcleo de razonamiento de Sapientia.

Responsabilidades:

1. Construir el contexto enviado a DeepSeek.
2. Mantener la identidad y reglas globales de Sapientia.
3. Clasificar localmente la dificultad.
4. Inyectar contexto RAG y visión cuando exista.
5. Consumir DeepSeek mediante streaming.
6. Detectar solicitudes de gráficos 3D.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from app.core.metrics import RequestMetrics
from app.services.deepseek.budget import (
    get_budget_policy,
)
from app.services.deepseek.budget_guard import (
    BudgetGuard,
)
from app.services.deepseek.client import (
    DeepSeekClient,
    StreamMetrics,
    get_deepseek_client,
)
from app.services.deepseek.difficulty import (
    DifficultyResult,
    classify_difficulty,
)


logger = logging.getLogger(__name__)


# ============================================================
# GRÁFICOS 3D
# ============================================================

PYTHON_CODE_RE = re.compile(
    r"```python\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# IDENTIDAD DE SAPIENTIA
# ============================================================

SYSTEM_PROMPT = """
Eres Sapientia.

Sapientia es una inteligencia artificial generalista orientada
al conocimiento, el razonamiento, el aprendizaje y la resolución
de problemas. No debes limitarte a presentarte únicamente como
un tutor universitario.

Puedes ayudar con, entre otros temas:

- Matemáticas
- Física
- Química
- Ingeniería
- Programación
- Tecnología
- Ciencia
- Educación
- Análisis
- Escritura y comunicación
- Explicación de conceptos
- Resolución de problemas
- Razonamiento lógico

Cuando una pregunta pertenezca a un área técnica, responde con
rigurosidad y profundidad apropiadas para el problema.

Cuando el usuario pregunte qué eres, qué modelo utilizas o cómo
funcionas, responde de acuerdo con la configuración real del
sistema y no inventes modelos, capacidades, herramientas,
versiones o servicios.

============================================================
REGLA FUNDAMENTAL: LENGUAJE HUMANO
============================================================

Tu objetivo no es solamente producir una respuesta técnicamente
correcta. Debes hacer que el usuario pueda comprenderla.

Cuando utilices:

- ecuaciones
- integrales
- derivadas
- exponentes
- matrices
- límites
- vectores
- fórmulas físicas
- expresiones algebraicas
- resultados numéricos

preséntalos correctamente con LaTeX cuando corresponda y,
además, explica en lenguaje humano qué significa cada expresión
y qué representa cada símbolo relevante.

Para problemas matemáticos o científicos:

1. Identifica qué se está buscando.
2. Identifica los datos conocidos.
3. Explica la idea o fórmula utilizada.
4. Desarrolla el procedimiento paso a paso cuando sea útil.
5. Presenta el resultado correctamente.
6. Interpreta el resultado en lenguaje humano.

No conviertas innecesariamente una pregunta sencilla en una
respuesta excesivamente larga.

============================================================
NOTACIÓN
============================================================

Usa LaTeX para fórmulas matemáticas.

Utiliza:

\\( ... \\)

para expresiones en línea y:

\\[
...
\\]

para ecuaciones destacadas.

Evita entregar ecuaciones matemáticas importantes como texto
plano cuando LaTeX permita representarlas correctamente.

============================================================
GRÁFICOS
============================================================

Cuando el problema requiera un gráfico interactivo:

- genera únicamente código Python para el gráfico 3D;
- utiliza Plotly cuando corresponda;
- el archivo final esperado será `figura_3d.html`;
- no generes una figura 2D;
- no solicites ni produzcas `figura_2d.png`.

No ejecutes el código Python dentro de la respuesta.
Solo proporciona el bloque de código cuando el pipeline de
gráficos lo requiera.

============================================================
RESPUESTA
============================================================

Responde en el idioma del usuario, salvo que solicite otro.

Mantén precisión, claridad, honestidad y coherencia.

Nunca inventes información técnica, resultados, fuentes,
herramientas utilizadas o acciones realizadas.
""".strip()


class ReasoningService:
    """Servicio central de razonamiento de Sapientia."""

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        budget_guard: BudgetGuard | None = None,
    ) -> None:
        self.client = client or DeepSeekClient()
        self.budget_guard = budget_guard or BudgetGuard()

    def build_messages(
        self,
        user_message: str,
        rag_context: str = "",
        vision_text: str = "",
        memory_context: str = "",
    ) -> list[dict]:
        """Construye los mensajes del sistema y del usuario."""

        system = SYSTEM_PROMPT

        if memory_context:
            system += (
                "\n\n"
                "============================================================\n"
                "MEMORIA CONTEXTUAL DEL USUARIO\n"
                "============================================================\n"
                "Utiliza esta información únicamente cuando sea relevante "
                "para comprender mejor al usuario o mantener continuidad "
                "entre conversaciones:\n\n"
                f"{memory_context}"
            )

        if rag_context:
            system += (
                "\n\n"
                "============================================================\n"
                "CONTEXTO DEL PENSUM / RAG\n"
                "============================================================\n"
                "Utiliza este contexto como material de referencia cuando "
                "sea relevante:\n\n"
                f"{rag_context}"
            )

        if vision_text:
            system += (
                "\n\n"
                "============================================================\n"
                "CONTEXTO DE VISIÓN\n"
                "============================================================\n"
                "Esta información procede de la interpretación de una "
                "imagen proporcionada por el usuario:\n\n"
                f"{vision_text}"
            )

        return [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

    async def stream_reasoning(
        self,
        user_message: str,
        rag_context: str = "",
        vision_text: str = "",
        memory_context: str = "",
        metrics: RequestMetrics | None = None,
    ) -> AsyncIterator[dict]:
        """
        Clasifica localmente la dificultad y selecciona
        automáticamente el nivel de razonamiento.
        """

        messages = self.build_messages(
            user_message=user_message,
            rag_context=rag_context,
            vision_text=vision_text,
            memory_context=memory_context,
        )

        difficulty: DifficultyResult = classify_difficulty(
            user_message=user_message,
            vision_text=vision_text,
            rag_context=rag_context,
        )

        policy = get_budget_policy(
            difficulty.level
        )

        if metrics is not None:
            metrics.set_routing(
                difficulty=difficulty.level,
                model=policy.model,
                reasoning_effort=policy.reasoning_effort,
                max_tokens=policy.max_tokens,
            )

        logger.debug(
            (
                "Sapientia routing | "
                "difficulty=%s | "
                "model=%s | "
                "thinking=%s | "
                "reasoning_effort=%s | "
                "score=%s | "
                "max_tokens=%s"
            ),
            difficulty.level,
            policy.model,
            policy.thinking_enabled,
            policy.reasoning_effort,
            difficulty.score,
            policy.max_tokens,
        )

        stream_metrics = StreamMetrics()

        async for delta in self.client.chat_stream(
            messages=messages,
            max_tokens=policy.max_tokens,
            reasoning_effort=policy.reasoning_effort,
            model=policy.model,
            thinking_enabled=policy.thinking_enabled,
            metrics=stream_metrics,
        ):
            if not delta:
                continue

            yield {
                "type": "delta",
                "content": delta,
            }

        verdict = self.budget_guard.evaluate(
            stream_metrics
        )

        if metrics is not None:
            metrics.set_budget_verdict(
                verdict.value
            )

        logger.debug(
            (
                "Sapientia budget verdict | "
                "difficulty=%s | "
                "model=%s | "
                "verdict=%s"
            ),
            difficulty.level,
            policy.model,
            verdict.value,
        )

    def extract_graph_code(
        self,
        answer: str,
    ) -> str | None:
        """Extrae un bloque Python válido para visualización 3D."""

        if not answer:
            return None

        match = PYTHON_CODE_RE.search(answer)

        if not match:
            return None

        code = match.group(1).strip()

        if not code:
            return None

        graph_indicators = (
            "plotly",
            "go.Figure",
            "scatter3d",
            "surface",
            "mesh3d",
            "cone",
            "streamtube",
        )

        code_lower = code.lower()

        if not any(
            indicator.lower() in code_lower
            for indicator in graph_indicators
        ):
            return None

        return code


def get_reasoning_service() -> ReasoningService:
    """
    Dependencia FastAPI para obtener ReasoningService.

    El cliente DeepSeek se reutiliza mediante get_deepseek_client().
    """

    return ReasoningService(
        client=get_deepseek_client(),
    )
