"""
reasoning_service.py — Orquesta el prompt y el razonamiento de DeepSeek R1.

Pipeline:
  1. Construir mensajes: system prompt + contexto RAG (pensum) + adjuntos.
  2. Llamar a DeepSeek R1 en streaming.
  3. Detectar bloques de código Python (matplotlib/plotly) → solicitud de gráfico.
"""
from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from app.services.deepseek.client import DeepSeekClient

logger = logging.getLogger(__name__)

# Código Python para gráficos (matplotlib/plotly) entre ```python ... ```
PYTHON_CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = (
    "Eres Sapientia, un tutor universitario experto en ciencias duras "
    "(Física, Matemáticas, Química e Ingeniería). "
    "Responde en el idioma del estudiante, paso a paso, con rigor académico. "
    "Usa notación LaTeX para fórmulas. "
    "Cuando el problema lo requiera, genera código Python dentro de "
    "```python``` que produzca gráficos en DOS formatos: "
    "(1) una figura 2D estática con matplotlib y (2) una figura 3D interactiva "
    "con plotly. Guarda la figura 2D como 'figura_2d.png' y la 3D como "
    "'figura_3d.html' usando fig.write_html('figura_3d.html'). "
    "No ejecutes código en tu respuesta; solo propón el bloque de código."
)


class ReasoningService:
    """Servicio de razonamiento con contexto RAG."""

    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient()

    def build_messages(
        self,
        user_message: str,
        rag_context: str = "",
        vision_text: str = "",
    ) -> list[dict]:
        """Construye la lista de mensajes OpenAI con contexto inyectado."""
        system = SYSTEM_PROMPT
        if rag_context:
            system += (
                "\n\nCONTEXTO DEL PENSUM (usa este material de referencia):\n"
                + rag_context
            )
        if vision_text:
            system += (
                "\n\nEXTRACTO DE LA IMAGEN DEL ESTUDIANTE "
                "(interpretado por el modelo de visión):\n"
                + vision_text
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    async def stream_reasoning(
        self,
        user_message: str,
        rag_context: str = "",
        vision_text: str = "",
    ) -> AsyncIterator[dict]:
        """
        Genera y hace stream del razonamiento.

        Emite dicts con forma `{"type": "delta", "content": "..."}`.
        El endpoint SSE se encarga de serializarlos a texto plano.
        """
        messages = self.build_messages(user_message, rag_context, vision_text)
        async for delta in self.client.chat_stream(messages):
            yield {"type": "delta", "content": delta}

    def extract_graph_code(self, answer: str) -> str | None:
        """Extrae el primer bloque ```python ... ``` si existe."""
        match = PYTHON_CODE_RE.search(answer)
        return match.group(1).strip() if match else None


def get_reasoning_service() -> ReasoningService:
    """Dependencia FastAPI: ReasoningService."""
    return ReasoningService()
