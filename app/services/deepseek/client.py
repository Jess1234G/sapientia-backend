"""
client.py — Cliente HTTP async para la API de DeepSeek.

La API de DeepSeek es compatible con el protocolo OpenAI Chat Completions,
por lo que se usa `openai` con base_url apuntando a DeepSeek.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Cliente async de DeepSeek (razonamiento R1)."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Streaming de Chat Completions. Itera sobre los deltas de contenido.

        `messages`: lista de mensajes con formato OpenAI
        ({role: "system"|"user"|"assistant", content: "..."}).
        """
        stream = await self.client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def get_deepseek_client() -> DeepSeekClient:
    """Dependencia FastAPI: DeepSeekClient singleton."""
    return DeepSeekClient()
