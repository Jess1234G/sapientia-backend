"""
fallback_policy.py — Política declarativa de fallback de Sapientia.

Este módulo NO realiza llamadas externas.
NO ejecuta reintentos.
NO decide dinámicamente durante el streaming.

Solo describe qué alternativas están permitidas para cada
veredicto del BudgetGuard.

Los valores concretos de producción deberán validarse
experimentalmente antes de activarse.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.deepseek.budget_guard import (
    BudgetVerdict,
)
from app.services.deepseek.client import (
    ReasoningEffort,
)


@dataclass(frozen=True)
class FallbackAttempt:
    """
    Describe un intento alternativo de generación.
    """

    model: str
    thinking_enabled: bool
    reasoning_effort: ReasoningEffort
    max_tokens: int


@dataclass(frozen=True)
class FallbackPolicy:
    """
    Secuencia ordenada de alternativas disponibles.

    Una tupla vacía significa:
        no existe fallback automático definido.
    """

    attempts: tuple[FallbackAttempt, ...] = ()

    @property
    def has_fallback(self) -> bool:
        """Indica si existe al menos una alternativa."""

        return bool(self.attempts)


# ============================================================
# POLÍTICAS
# ============================================================

# IMPORTANTE:
# Estas políticas están deliberadamente vacías por ahora.
#
# Todavía no hemos validado experimentalmente una segunda
# estrategia que pueda utilizarse después de un fallo sin
# sacrificar calidad o generar un costo desproporcionado.
#
# No activamos reintentos basándonos en suposiciones.

FALLBACK_POLICIES: dict[
    BudgetVerdict,
    FallbackPolicy,
] = {
    BudgetVerdict.OK: FallbackPolicy(),

    BudgetVerdict.TRUNCATED: FallbackPolicy(),

    BudgetVerdict.NO_CONTENT: FallbackPolicy(),

    BudgetVerdict.TTFC_EXCEEDED: FallbackPolicy(),

    BudgetVerdict.TOTAL_EXCEEDED: FallbackPolicy(),
}


def get_fallback_policy(
    verdict: BudgetVerdict,
) -> FallbackPolicy:
    """
    Obtiene la política de fallback asociada al veredicto.

    La función siempre devuelve una política válida.
    """

    return FALLBACK_POLICIES.get(
        verdict,
        FallbackPolicy(),
    )
