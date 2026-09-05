"""
benchmark_pro_4096.py

Prueba una consulta compleja con:

    deepseek-v4-pro
    reasoning_effort="high"
    max_tokens=4096

Objetivo:
    validar si 4096 tokens son suficientes para una consulta
    compleja y medir consumo, latencia y completitud.
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
Resuelve la integral:

∫ x e^x dx

Explica el procedimiento paso a paso.

Debes:
1. identificar el método apropiado;
2. aplicar correctamente el método de integración;
3. mostrar los pasos matemáticos importantes;
4. obtener la solución final;
5. verificar por derivación que el resultado es correcto;
6. explicar en lenguaje humano por qué el método funciona.

Presenta las expresiones matemáticas correctamente en LaTeX.
Responde en español con rigor y claridad.
No omitas pasos importantes, pero evita una explicación innecesariamente extensa.
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
                    "Responde con precisión, rigor y "
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
            max_tokens=2048,
            reasoning_effort="high",
            metrics=metrics,
            model="deepseek-v4-pro",
        ):
            chunks.append(content)

        answer = "".join(chunks).strip()
        usage = metrics.usage

        print("=" * 70)
        print("SAPIENTIA — V4 PRO — CONSULTA COMPLEJA")
        print("=" * 70)
        print("Modelo              : deepseek-v4-pro")
        print("Reasoning effort     : high")
        print("Max tokens           : 2048")
        print()

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

        print(
            f"Streaming           : "
            f"{metrics.stream_seconds * 1000:.2f} ms"
            if metrics.stream_seconds is not None
            else "Streaming           : N/A"
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
