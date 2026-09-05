"""
benchmark_flash_1536.py

Una única llamada real a DeepSeek V4 Flash.

Objetivo:
    repetir la prueba de 1536 tokens bajo exactamente
    las mismas condiciones.
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
Resuelve la siguiente ecuación cuadrática:

2x² - 7x + 3 = 0

Explica el procedimiento paso a paso.

Incluye:
1. identificación de los coeficientes;
2. la fórmula cuadrática;
3. sustitución de los valores;
4. cálculo de las soluciones;
5. interpretación del resultado en lenguaje humano.

Presenta correctamente las expresiones matemáticas en LaTeX.
Responde en español de forma clara, rigurosa y comprensible.
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
        ):
            chunks.append(content)

        answer = "".join(chunks).strip()
        usage = metrics.usage

        print("=" * 70)
        print("SAPIENTIA — REPETICIÓN FLASH 1536")
        print("=" * 70)

        print(
            f"TTFR                : "
            f"{metrics.time_to_first_reasoning_ms:.2f} ms"
            if metrics.time_to_first_reasoning_ms is not None
            else "TTFR                : N/A"
        )

        print(
            f"TTFC                : "
            f"{metrics.ttft_seconds * 1000:.2f} ms"
            if metrics.ttft_seconds is not None
            else "TTFC                : N/A"
        )

        print(
            f"Reasoning→Content   : "
            f"{metrics.reasoning_to_content_ms:.2f} ms"
            if metrics.reasoning_to_content_ms is not None
            else "Reasoning→Content   : N/A"
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