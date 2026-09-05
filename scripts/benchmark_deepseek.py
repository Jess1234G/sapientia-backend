"""
benchmark_deepseek.py — Comparación V4 Flash vs V4 Pro.

No forma parte de pytest.

Realiza dos llamadas reales a DeepSeek con:
- misma pregunta
- mismo prompt
- mismo max_tokens
- mismo reasoning_effort

Solo cambia:
    deepseek-v4-flash
    deepseek-v4-pro
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

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


@dataclass
class BenchmarkResult:
    model: str

    ttfr_ms: float | None
    ttfc_ms: float | None
    reasoning_to_content_ms: float | None
    total_ms: float | None
    stream_ms: float | None

    chunks: int
    characters: int

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_cache_hit_tokens: int | None
    prompt_cache_miss_tokens: int | None
    reasoning_tokens: int | None

    answer: str


async def run_case(
    client: DeepSeekClient,
    model: str,
) -> BenchmarkResult:

    metrics = StreamMetrics()

    chunks = []
    chunk_count = 0
    characters = 0

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
        model=model,
    ):
        chunks.append(content)
        chunk_count += 1
        characters += len(content)

    answer = "".join(chunks).strip()

    usage = metrics.usage

    return BenchmarkResult(
        model=model,

        ttfr_ms=(
            metrics.time_to_first_reasoning_ms
            if hasattr(
                metrics,
                "time_to_first_reasoning_ms",
            )
            else None
        ),

        ttfc_ms=(
            metrics.ttft_seconds * 1000
            if metrics.ttft_seconds is not None
            else None
        ),

        reasoning_to_content_ms=(
            metrics.reasoning_to_content_ms
            if hasattr(
                metrics,
                "reasoning_to_content_ms",
            )
            else None
        ),

        total_ms=(
            metrics.total_seconds * 1000
            if metrics.total_seconds is not None
            else None
        ),

        stream_ms=(
            metrics.stream_seconds * 1000
            if metrics.stream_seconds is not None
            else None
        ),

        chunks=chunk_count,
        characters=characters,

        prompt_tokens=(
            usage.prompt_tokens
            if usage is not None
            else None
        ),

        completion_tokens=(
            usage.completion_tokens
            if usage is not None
            else None
        ),

        total_tokens=(
            usage.total_tokens
            if usage is not None
            else None
        ),

        prompt_cache_hit_tokens=(
            usage.prompt_cache_hit_tokens
            if usage is not None
            else None
        ),

        prompt_cache_miss_tokens=(
            usage.prompt_cache_miss_tokens
            if usage is not None
            else None
        ),

        reasoning_tokens=(
            usage.reasoning_tokens
            if usage is not None
            else None
        ),

        answer=answer,
    )


def print_result(
    result: BenchmarkResult,
) -> None:

    print()
    print("=" * 70)
    print(result.model)
    print("=" * 70)

    print(
        f"TTFR                : "
        f"{result.ttfr_ms:.2f} ms"
        if result.ttfr_ms is not None
        else "TTFR                : N/A"
    )

    print(
        f"TTFC                : "
        f"{result.ttfc_ms:.2f} ms"
        if result.ttfc_ms is not None
        else "TTFC                : N/A"
    )

    print(
        f"Reasoning→Content   : "
        f"{result.reasoning_to_content_ms:.2f} ms"
        if result.reasoning_to_content_ms is not None
        else "Reasoning→Content   : N/A"
    )

    print(
        f"Tiempo total        : "
        f"{result.total_ms:.2f} ms"
        if result.total_ms is not None
        else "Tiempo total        : N/A"
    )

    print(
        f"Streaming           : "
        f"{result.stream_ms:.2f} ms"
        if result.stream_ms is not None
        else "Streaming           : N/A"
    )

    print(
        f"Chunks              : "
        f"{result.chunks}"
    )

    print(
        f"Caracteres          : "
        f"{result.characters}"
    )

    print(
        f"Prompt tokens       : "
        f"{result.prompt_tokens}"
        if result.prompt_tokens is not None
        else "Prompt tokens       : N/A"
    )

    print(
        f"Completion tokens   : "
        f"{result.completion_tokens}"
        if result.completion_tokens is not None
        else "Completion tokens   : N/A"
    )

    print(
        f"Reasoning tokens    : "
        f"{result.reasoning_tokens}"
        if result.reasoning_tokens is not None
        else "Reasoning tokens    : N/A"
    )

    print(
        f"Total tokens        : "
        f"{result.total_tokens}"
        if result.total_tokens is not None
        else "Total tokens        : N/A"
    )

    print(
        f"Cache hit tokens    : "
        f"{result.prompt_cache_hit_tokens}"
        if result.prompt_cache_hit_tokens is not None
        else "Cache hit tokens    : N/A"
    )

    print(
        f"Cache miss tokens   : "
        f"{result.prompt_cache_miss_tokens}"
        if result.prompt_cache_miss_tokens is not None
        else "Cache miss tokens   : N/A"
    )

    print()
    print("RESPUESTA:")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)


async def main() -> int:

    if not settings.deepseek_api_key:
        print(
            "ERROR: DEEPSEEK_API_KEY "
            "no está configurada."
        )
        return 1

    print("=" * 70)
    print("SAPIENTIA — V4 FLASH vs V4 PRO")
    print("=" * 70)
    print(
        "Misma pregunta | high | 1536 tokens"
    )
    print()

    client = DeepSeekClient()

    try:
        flash = await run_case(
            client=client,
            model="deepseek-v4-flash",
        )

        print_result(flash)

        pro = await run_case(
            client=client,
            model="deepseek-v4-pro",
        )

        print_result(pro)

        print()
        print("=" * 70)
        print("COMPARACIÓN")
        print("=" * 70)

        if (
            flash.ttfc_ms is not None
            and pro.ttfc_ms is not None
            and pro.ttfc_ms > 0
        ):
            speed_ratio = (
                pro.ttfc_ms
                / flash.ttfc_ms
            )

            print(
                f"Flash vs Pro — TTFC: "
                f"{speed_ratio:.2f}x"
            )

        if (
            flash.total_ms is not None
            and pro.total_ms is not None
            and pro.total_ms > 0
        ):
            total_ratio = (
                pro.total_ms
                / flash.total_ms
            )

            print(
                f"Flash vs Pro — total: "
                f"{total_ratio:.2f}x"
            )

        print()
        print(
            "La selección automática de modelo "
            "NO se ha modificado."
        )

        return 0

    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(
        asyncio.run(main())
    )