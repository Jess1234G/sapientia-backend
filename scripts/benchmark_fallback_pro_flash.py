"""
benchmark_fallback_pro_flash.py

Prueba real de una estrategia de recuperación:

    Intento 1:
        deepseek-v4-pro
        high
        2048 tokens

    Si el BudgetGuard detecta:
        TRUNCATED
        NO_CONTENT

    Intento 2:
        deepseek-v4-flash
        low
        1536 tokens

Presupuesto global:
    2048 + 1536 = 3584 tokens de generación.

IMPORTANTE:
    Este script es únicamente experimental.
    No modifica la política de fallback de producción.
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from app.services.deepseek.budget_guard import (
    BudgetGuard,
    BudgetVerdict,
)
from app.services.deepseek.client import (
    DeepSeekClient,
    StreamMetrics,
)
from app.services.deepseek.fallback_engine import (
    FallbackEngine,
)
from app.services.deepseek.fallback_policy import (
    FALLBACK_POLICIES,
    FallbackAttempt,
    FallbackPolicy,
)
from app.services.deepseek.request_budget import (
    RequestBudget,
    RequestBudgetConfig,
)


PROMPT = """
Resuelve la siguiente ecuación diferencial:

dy/dx = y(1 - y)

Obtén la solución general paso a paso.
Explica el método utilizado, realiza la separación de variables,
integra correctamente, despeja y verifica la solución.

Incluye una explicación clara en lenguaje humano y presenta las
expresiones matemáticas correctamente en LaTeX.

La respuesta debe ser completa, rigurosa y no omitir pasos
matemáticos importantes.
""".strip()


SYSTEM_PROMPT = """
Eres Sapientia.

Responde en español con precisión matemática, rigor y claridad.
No inventes resultados. Explica los pasos importantes y utiliza
LaTeX para las expresiones matemáticas.
""".strip()


GLOBAL_BUDGET = 3584


def print_metrics(
    label: str,
    metrics: StreamMetrics,
    answer: str,
) -> None:
    """Muestra las métricas de un intento."""

    usage = metrics.usage

    print("=" * 70)
    print(label)
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
        f"Tiempo total        : "
        f"{metrics.total_seconds * 1000:.2f} ms"
        if metrics.total_seconds is not None
        else "Tiempo total        : N/A"
    )

    print(
        f"Finish reason       : "
        f"{metrics.finish_reason}"
    )

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

    print()
    print("RESPUESTA")
    print("-" * 70)
    print(answer)
    print("-" * 70)


async def run_attempt(
    client: DeepSeekClient,
    *,
    model: str,
    reasoning_effort: str,
    max_tokens: int,
) -> tuple[str, StreamMetrics]:
    """Ejecuta un único intento real."""

    metrics = StreamMetrics()
    parts: list[str] = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": PROMPT,
        },
    ]

    async for content in client.chat_stream(
        messages=messages,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        model=model,
        thinking_enabled=True,
        metrics=metrics,
    ):
        if content:
            parts.append(content)

    return "".join(parts).strip(), metrics


async def main() -> int:
    """
    Ejecuta el experimento completo.
    """

    print("=" * 70)
    print("SAPIENTIA — FALLBACK REAL PRO → FLASH")
    print("=" * 70)
    print()
    print(
        "Presupuesto global de generación : "
        f"{GLOBAL_BUDGET}"
    )
    print()

    # ------------------------------------------------------------
    # Presupuesto global
    # ------------------------------------------------------------

    request_budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=GLOBAL_BUDGET,
            max_attempts=2,
        )
    )

    # ------------------------------------------------------------
    # Política experimental LOCAL
    # ------------------------------------------------------------
    #
    # No modificamos permanentemente fallback_policy.py.
    # Guardamos la política original y la restauramos al final.

    original_truncated = FALLBACK_POLICIES[
        BudgetVerdict.TRUNCATED
    ]

    original_no_content = FALLBACK_POLICIES[
        BudgetVerdict.NO_CONTENT
    ]

    experimental_fallback = FallbackPolicy(
        attempts=(
            FallbackAttempt(
                model="deepseek-v4-flash",
                thinking_enabled=True,
                reasoning_effort="low",
                max_tokens=1536,
            ),
        )
    )

    FALLBACK_POLICIES[
        BudgetVerdict.TRUNCATED
    ] = experimental_fallback

    FALLBACK_POLICIES[
        BudgetVerdict.NO_CONTENT
    ] = experimental_fallback

    guard = BudgetGuard()

    client = DeepSeekClient()

    try:
        # ========================================================
        # INTENTO 1 — PRO
        # ========================================================

        if not request_budget.can_start(2048):
            print(
                "ERROR: el intento Pro no cabe "
                "en el presupuesto global."
            )
            return 1

        request_budget.reserve(2048)

        started_at = perf_counter()

        answer_1, metrics_1 = await run_attempt(
            client,
            model="deepseek-v4-pro",
            reasoning_effort="high",
            max_tokens=2048,
        )

        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000.0

        verdict_1 = guard.evaluate(
            metrics_1
        )

        print_metrics(
            "INTENTO 1 — V4 PRO",
            metrics_1,
            answer_1,
        )

        print(
            f"Tiempo medido externamente: "
            f"{elapsed_ms:.2f} ms"
        )

        print(
            f"Budget verdict      : "
            f"{verdict_1.value}"
        )

        print(
            f"Presupuesto reservado: "
            f"{request_budget.reserved_generation_tokens}"
        )

        print(
            f"Presupuesto restante : "
            f"{request_budget.remaining_generation_tokens}"
        )

        # ========================================================
        # SI PRO FUNCIONÓ
        # ========================================================

        if verdict_1 == BudgetVerdict.OK:
            print()
            print(
                "PRO TERMINÓ CORRECTAMENTE."
            )
            print(
                "No se ejecuta fallback."
            )
            return 0

        # ========================================================
        # FALLBACK
        # ========================================================

        fallback_engine = FallbackEngine(
            request_budget=request_budget
        )

        fallback_attempt = (
            fallback_engine.next_attempt(
                verdict_1
            )
        )

        if fallback_attempt is None:
            print()
            print(
                "No existe un fallback permitido "
                "dentro del presupuesto global."
            )
            return 0

        print()
        print("=" * 70)
        print("FALLBACK ACTIVADO")
        print("=" * 70)

        print(
            f"Modelo             : "
            f"{fallback_attempt.model}"
        )
        print(
            f"Reasoning effort   : "
            f"{fallback_attempt.reasoning_effort}"
        )
        print(
            f"Max tokens         : "
            f"{fallback_attempt.max_tokens}"
        )

        print(
            f"Presupuesto reservado: "
            f"{request_budget.reserved_generation_tokens}"
        )

        print(
            f"Presupuesto restante: "
            f"{request_budget.remaining_generation_tokens}"
        )


        # ========================================================
        # INTENTO 2 — FLASH
        # ========================================================

        answer_2, metrics_2 = await run_attempt(
            client,
            model=fallback_attempt.model,
            reasoning_effort=(
                fallback_attempt.reasoning_effort
            ),
            max_tokens=fallback_attempt.max_tokens,
        )

        verdict_2 = guard.evaluate(
            metrics_2
        )

        print_metrics(
            "INTENTO 2 — FALLBACK FLASH",
            metrics_2,
            answer_2,
        )

        print(
            f"Budget verdict      : "
            f"{verdict_2.value}"
        )

        print(
            f"Presupuesto reservado: "
            f"{request_budget.reserved_generation_tokens}"
        )

        print(
            f"Presupuesto restante : "
            f"{request_budget.remaining_generation_tokens}"
        )

        # ========================================================
        # RESULTADO FINAL
        # ========================================================

        print()
        print("=" * 70)
        print("RESULTADO DEL EXPERIMENTO")
        print("=" * 70)

        print(
            f"Intento 1           : "
            f"{verdict_1.value}"
        )

        print(
            f"Intento 2           : "
            f"{verdict_2.value}"
        )

        if (
            verdict_2 == BudgetVerdict.OK
            and answer_2
        ):
            print(
                "Fallback recuperó la solicitud."
            )
        elif not answer_2:
            print(
                "Fallback no produjo contenido."
            )
        else:
            print(
                "Fallback tampoco alcanzó "
                "un estado OK."
            )

        return 0

    finally:
        FALLBACK_POLICIES[
            BudgetVerdict.TRUNCATED
        ] = original_truncated

        FALLBACK_POLICIES[
            BudgetVerdict.NO_CONTENT
        ] = original_no_content

        await client.close()


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )

