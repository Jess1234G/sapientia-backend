"""
fallback_engine.py — Coordinación segura de intentos de fallback.

Este componente:

- no realiza llamadas a DeepSeek;
- no ejecuta reintentos;
- no modifica modelos;
- no decide una política nueva.

Solo coordina FallbackPolicy + RequestBudget.
"""

from __future__ import annotations

from app.services.deepseek.budget_guard import (
    BudgetVerdict,
)
from app.services.deepseek.fallback_policy import (
    FallbackAttempt,
    get_fallback_policy,
)
from app.services.deepseek.request_budget import (
    RequestBudget,
)


class FallbackEngine:
    """
    Selecciona el siguiente intento de fallback permitido.

    La reserva del presupuesto se realiza antes de devolver
    el intento, de modo que el caller nunca reciba una alternativa
    que no haya sido previamente reservada.
    """

    def __init__(
        self,
        request_budget: RequestBudget,
    ) -> None:
        self.request_budget = request_budget

    def next_attempt(
        self,
        verdict: BudgetVerdict,
    ) -> FallbackAttempt | None:
        """
        Devuelve el siguiente intento permitido.

        Reglas:

        1. Obtiene la FallbackPolicy asociada al veredicto.
        2. Recorre los intentos en orden.
        3. Comprueba si cada intento cabe en RequestBudget.
        4. Si cabe, reserva su max_tokens y lo devuelve.
        5. Si no cabe, prueba la siguiente alternativa.
        6. Si ninguna cabe, devuelve None.
        """

        policy = get_fallback_policy(
            verdict
        )

        for attempt in policy.attempts:
            if not self.request_budget.can_start(
                attempt.max_tokens
            ):
                continue

            self.request_budget.reserve(
                attempt.max_tokens
            )

            return attempt

        return None
