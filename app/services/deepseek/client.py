"""
client.py — Cliente HTTP asíncrono para la API de DeepSeek.

Mantiene compatibilidad con el protocolo OpenAI Chat Completions
y permite controlar el esfuerzo de razonamiento de DeepSeek V4.

También ofrece métricas ligeras por solicitud:
    - tiempo hasta el primer token (TTFT)
    - tiempo total de la solicitud
    - tiempo de streaming después del primer token
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

from openai import AsyncOpenAI

from app.config import settings


logger = logging.getLogger(__name__)


ReasoningEffort = Literal["low", "high", "max"]


@dataclass
class TokenUsage:
    """
    Conteo de tokens reportado por DeepSeek en el chunk final
    de usage (que llega con `choices=[]`).
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    completion_tokens_details: Any | None = None

    @property
    def reasoning_tokens(self) -> int | None:
        """Tokens de razonamiento interno (thinking)."""

        if self.completion_tokens_details is None:
            return None

        return getattr(
            self.completion_tokens_details,
            "reasoning_tokens",
            None,
        )

    @classmethod
    def from_usage(cls, usage: Any) -> "TokenUsage":
        """Construye TokenUsage desde el objeto usage del chunk."""

        return cls(
            prompt_tokens=getattr(
                usage,
                "prompt_tokens",
                None,
            ),
            completion_tokens=getattr(
                usage,
                "completion_tokens",
                None,
            ),
            total_tokens=getattr(
                usage,
                "total_tokens",
                None,
            ),
            prompt_cache_hit_tokens=getattr(
                usage,
                "prompt_cache_hit_tokens",
                None,
            ),
            prompt_cache_miss_tokens=getattr(
                usage,
                "prompt_cache_miss_tokens",
                None,
            ),
            completion_tokens_details=getattr(
                usage,
                "completion_tokens_details",
                None,
            ),
        )


@dataclass
class StreamMetrics:
    """
    Métricas de una única solicitud de streaming.

    La instancia pertenece a una sola solicitud y, por tanto,
    es segura frente a concurrencia entre usuarios.
    """

    request_started_at: float | None = None
    first_reasoning_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None

    finish_reason: str | None = None

    usage: TokenUsage | None = None

    @property
    def time_to_first_reasoning_ms(self) -> float | None:
        """Tiempo hasta el primer fragmento de razonamiento."""

        if self.first_reasoning_at is None:
            return None

        return max(
            0.0,
            (
                self.first_reasoning_at
                - self.request_started_at
            ) * 1000.0,
        )

    @property
    def reasoning_to_content_ms(self) -> float | None:
        """
        Tiempo entre el primer razonamiento y el primer contenido
        visible.
        """

        if (
            self.first_reasoning_at is None
            or self.first_token_at is None
        ):
            return None

        return max(
            0.0,
            (
                self.first_token_at
                - self.first_reasoning_at
            ) * 1000.0,
        )

    @property
    def ttft_seconds(self) -> float | None:
        """Tiempo hasta el primer fragmento de contenido."""

        if (
            self.request_started_at is None
            or self.first_token_at is None
        ):
            return None

        return max(
            0.0,
            self.first_token_at
            - self.request_started_at,
        )

    @property
    def total_seconds(self) -> float | None:
        """Tiempo total de la solicitud."""

        if (
            self.request_started_at is None
            or self.completed_at is None
        ):
            return None

        return max(
            0.0,
            self.completed_at
            - self.request_started_at,
        )

    @property
    def stream_seconds(self) -> float | None:
        """Tiempo desde el primer token hasta el final."""

        if (
            self.first_token_at is None
            or self.completed_at is None
        ):
            return None

        return max(
            0.0,
            self.completed_at
            - self.first_token_at,
        )

    @property
    def completed_normally(self) -> bool | None:
        """
        Indica si la generación terminó mediante una parada normal.

        None significa que todavía no conocemos el motivo.
        """

        if self.finish_reason is None:
            return None

        return self.finish_reason == "stop"

    @property
    def truncated(self) -> bool:
        """
        Indica si DeepSeek terminó la generación por límite de longitud.
        """

        return self.finish_reason == "length"


class DeepSeekClient:
    """
    Cliente async reutilizable de DeepSeek.

    Mantiene una única instancia de AsyncOpenAI por proceso y
    mide latencia individual por solicitud cuando el caller
    proporciona un StreamMetrics.
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        reasoning_effort: ReasoningEffort = "high",
        metrics: StreamMetrics | None = None,
        model: str | None = None,
        thinking_enabled: bool = True,
    ) -> AsyncIterator[str]:
        """
        Genera una respuesta mediante streaming.

        Parámetros:
            messages:
                Mensajes compatibles con Chat Completions.

            max_tokens:
                Límite de tokens de salida.

            reasoning_effort:
                `high` para razonamiento normal.
                `max` para problemas especialmente complejos.

            metrics:
                Objeto opcional para registrar TTFT y latencia total.
        """

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens debe ser mayor que 0."
            )

        if reasoning_effort not in {
            "low",
            "high",
            "max",
        }:
            raise ValueError(
                "reasoning_effort debe ser "
                "'low', 'high' o 'max'."
            )

        request_kwargs = {
            "model": model or settings.deepseek_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "reasoning_effort": reasoning_effort,
            "stream_options": {
                "include_usage": True,
            },
            "extra_body": {
                "thinking": {
                    "type": (
                        "enabled"
                        if thinking_enabled
                        else "disabled"
                    ),
                }
            },
        }

        active_metrics = metrics or StreamMetrics()

        active_metrics.request_started_at = perf_counter()

        try:
            stream = await self.client.chat.completions.create(
                **request_kwargs,
            )

            async for chunk in stream:
                # El chunk de usage llega con choices=[]. Lo
                # procesamos antes de comprobar choices para no
                # perder la información de tokens.
                usage = getattr(
                    chunk,
                    "usage",
                    None,
                )

                if usage is not None:
                    active_metrics.usage = TokenUsage.from_usage(
                        usage
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                finish_reason = getattr(
                    choice,
                    "finish_reason",
                    None,
                )

                if finish_reason is not None:
                    active_metrics.finish_reason = (
                        finish_reason
                    )

                delta = choice.delta

                if not delta:
                    continue

                reasoning_content = getattr(
                    delta,
                    "reasoning_content",
                    None,
                )

                if (
                    reasoning_content
                    and active_metrics.first_reasoning_at is None
                ):
                    active_metrics.first_reasoning_at = perf_counter()

                content = getattr(
                    delta,
                    "content",
                    None,
                )

                if not content:
                    continue

                if active_metrics.first_token_at is None:
                    active_metrics.first_token_at = perf_counter()

                yield content

        finally:
            active_metrics.completed_at = perf_counter()

            logger.debug(
                (
                    "DeepSeek stream completed | "
                    "reasoning_effort=%s | "
                    "max_tokens=%s | "
                    "ttft=%.4fs | "
                    "total=%.4fs | "
                    "finish_reason=%s"
                ),
                reasoning_effort,
                max_tokens,
                active_metrics.ttft_seconds or 0.0,
                active_metrics.total_seconds or 0.0,
                active_metrics.finish_reason,
            )

    async def close(self) -> None:
        """Cierra correctamente el cliente HTTP asíncrono."""

        await self.client.close()


@lru_cache(maxsize=1)
def get_deepseek_client() -> DeepSeekClient:
    """
    Devuelve una única instancia reutilizable de DeepSeekClient
    por proceso de aplicación.
    """

    return DeepSeekClient()