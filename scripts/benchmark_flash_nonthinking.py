"""
benchmark_flash_nonthinking.py

Prueba una consulta sencilla con:

    deepseek-v4-flash
    thinking=disabled
    max_tokens=1536

Objetivo:
    comprobar si una consulta sencilla puede resolverse
    correctamente sin razonamiento explícito.
"""

from __future__ import annotations

import asyncio
import sys

from app.config import settings
from app.services.deepseek.client import (
    DeepSeekClient,
    StreamMetrics,
)


PROMPT = """
Explica qué es la derivada de una función.

Incluye:
1. una definición matemática correcta;
2. la expresión matemática de la derivada;
3. una explicación en lenguaje humano;
4. un ejemplo sencillo.

Responde en español, de forma correcta, clara y concisa.
""".strip()


async def main() -> int:

    if not settings.deepseek_api_key:
        print(
            "ERROR: DEEPSEEK_API_KEY no está configurada."
        )
        return 1

    client = DeepSeekClient()
    metrics = StreamMetrics()
    chunks: list[str] = []

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres Sapientia. "
                    "Responde con precisión, claridad y "
                    "lenguaje humano."
                ),
            },
            {
                "role": "user",
                "content": PROMPT,
            },
        ]

        async for content in client.chat_stream(
            messages=messages,
            max_tokens=1536,
            reasoning_effort="high",
            metrics=metrics,
            model="deepseek-v4-flash",
            thinking_enabled=False,
        ):
            chunks.append(content)

        answer = "".join(chunks).strip()
        usage = metrics.usage

        print("=" * 70)
        print("SAPIENTIA — FLASH — THINKING OFF")
        print("=" * 70)

        print("Modelo              : deepseek-v4-flash")
        print("Thinking            : disabled")
        print("Max tokens          : 1536")
        print()

        print(
            f"TTFC                : "
            f"{metrics.ttft_seconds * 1000:.2f} ms"
            if metrics.ttft_seconds is not None
            else "TTFC                : N/A"
        )

        print(
            f"Tiempo total        : "
            f"{metrics.total_seconds * 1000:.2f} ms"
            if metrics.total_seconds is not None
            else "Tiempo total        : N/A"
        )

        print()

        if usage is not None:
            print(
                f"Prompt tokens       : "
                f"{usage.prompt_tokens}"
            )
            print(
                f"Completion tokens   : "
                f"{usage.completion_tokens}"
            )
            print(
                f"Reasoning tokens    : "
                f"{usage.reasoning_tokens}"
            )
            print(
                f"Total tokens        : "
                f"{usage.total_tokens}"
            )
            print(
                f"Cache hit tokens    : "
                f"{usage.prompt_cache_hit_tokens}"
            )
            print(
                f"Cache miss tokens   : "
                f"{usage.prompt_cache_miss_tokens}"
            )
        else:
            print("USAGE: N/A")

        print()
        print("RESPUESTA")
        print("-" * 70)
        print(answer)
        print("-" * 70)

        return 0

    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(
        asyncio.run(main())
    )
