"""
budget_guard.py — Detección de problemas de presupuesto y latencia.

No realiza llamadas externas.
No ejecuta fallbacks.
No modifica modelos.

Su única responsabilidad es evaluar el estado de una generación.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter

from app.services.deepseek.client import StreamMetrics


class BudgetVerdict(str, Enum):
    """
    Resultado de la evaluación de una generación.
    """

    OK = "ok"
    TRUNCATED = "truncated"
    NO_CONTENT = "no_content"
    TTFC_EXCEEDED = "ttfc_exceeded"
    TOTAL_EXCEEDED = "total_exceeded"


@dataclass(frozen=True)
class BudgetLimits:
    """
    Límites configurables.

    Ningún valor aquí se considera todavía una política definitiva
    de producción. Los umbrales deberán establecerse mediante
    benchmarks posteriores.
    """

    max_ttfc_ms: float | None = None
    max_total_ms: float | None = None


class BudgetGuard:
    """
    Evaluador determinista del estado de una generación.

    No realiza reintentos.
    No cancela solicitudes.
    """

    def __init__(
        self,
        limits: BudgetLimits | None = None,
    ) -> None:
        self.limits = limits or BudgetLimits()

    def evaluate(
        self,
        metrics: StreamMetrics,
    ) -> BudgetVerdict:
        """
        Evalúa una solicitud ya finalizada o cuyo estado sea conocido.
        """

        # Un finish_reason="length" es evidencia directa
        # de que se alcanzó el límite de generación.
        if metrics.truncated:
            return BudgetVerdict.TRUNCATED

        # Si terminó sin producir contenido visible.
        if metrics.first_token_at is None:
            return BudgetVerdict.NO_CONTENT

        # TTFC conocido.
        ttfc_ms = metrics.ttft_seconds

        if (
            self.limits.max_ttfc_ms is not None
            and ttfc_ms is not None
            and ttfc_ms * 1000.0
            > self.limits.max_ttfc_ms
        ):
            return BudgetVerdict.TTFC_EXCEEDED

        # Tiempo total conocido.
        total_ms = metrics.total_seconds

        if (
            self.limits.max_total_ms is not None
            and total_ms is not None
            and total_ms * 1000.0
            > self.limits.max_total_ms
        ):
            return BudgetVerdict.TOTAL_EXCEEDED

        return BudgetVerdict.OK

    def ttfc_exceeded_live(
        self,
        metrics: StreamMetrics,
    ) -> bool:
        """
        Permite comprobar TTFC mientras el stream sigue abierto.

        No cancela la petición.
        """

        limit = self.limits.max_ttfc_ms

        if limit is None:
            return False

        if metrics.first_token_at is not None:
            return False

        if metrics.request_started_at is None:
            return False

        elapsed_ms = (
            perf_counter()
            - metrics.request_started_at
        ) * 1000.0

        return elapsed_ms > limit

    def total_exceeded_live(
        self,
        metrics: StreamMetrics,
    ) -> bool:
        """
        Permite comprobar el tiempo total mientras el stream sigue abierto.

        No cancela la petición.
        """

        limit = self.limits.max_total_ms

        if limit is None:
            return False

        if metrics.request_started_at is None:
            return False

        elapsed_ms = (
            perf_counter()
            - metrics.request_started_at
        ) * 1000.0

        return elapsed_ms > limit
