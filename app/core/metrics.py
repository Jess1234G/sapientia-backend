"""
metrics.py — Métricas ligeras por petición de Sapientia.

Las métricas son locales a una sola petición y no contienen
datos del usuario ni contenido de la conversación.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RequestMetrics:
    """
    Medidor independiente para una única petición.

    Los tiempos se expresan en milisegundos cuando se exponen
    mediante summary().
    """

    started_at: float = field(
        default_factory=perf_counter
    )

    phases: dict[str, float] = field(
        default_factory=dict
    )

    first_token_at: float | None = None
    finished_at: float | None = None

    difficulty: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    budget_verdict: str | None = None

    def mark_phase(
        self,
        name: str,
        started_at: float,
    ) -> None:
        """
        Registra la duración de una fase.
        """

        self.phases[name] = max(
            0.0,
            perf_counter() - started_at,
        )

    def mark_first_token(self) -> None:
        """Registra el instante del primer contenido recibido."""

        if self.first_token_at is None:
            self.first_token_at = perf_counter()

    def finish(self) -> None:
        """Registra el final del procesamiento."""

        self.finished_at = perf_counter()

    def set_routing(
        self,
        *,
        difficulty: str,
        model: str,
        reasoning_effort: str,
        max_tokens: int,
    ) -> None:
        """Almacena la decisión de enrutamiento del modelo."""

        self.difficulty = difficulty
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    def set_budget_verdict(
        self,
        verdict: str,
    ) -> None:
        """Almacena el resultado de la evaluación del BudgetGuard."""

        self.budget_verdict = verdict

    @property
    def time_to_first_token_ms(self) -> float | None:
        """Tiempo desde el inicio del endpoint hasta el primer token."""

        if self.first_token_at is None:
            return None

        return max(
            0.0,
            (
                self.first_token_at
                - self.started_at
            ) * 1000.0,
        )

    @property
    def total_ms(self) -> float | None:
        """Tiempo desde entrada al endpoint hasta finalización."""

        if self.finished_at is None:
            return None

        return max(
            0.0,
            (
                self.finished_at
                - self.started_at
            ) * 1000.0,
        )

    def summary(self) -> dict[str, object]:
        """
        Devuelve métricas de rendimiento junto con la decisión
        de routing. Nunca incluye contenido del usuario.
        """

        result: dict[str, object] = {
            "time_to_first_token_ms": (
                self.time_to_first_token_ms
            ),
            "total_ms": self.total_ms,
        }

        result.update(
            {
                f"{name}_ms": duration * 1000.0
                for name, duration
                in self.phases.items()
            }
        )

        # Metadatos de routing (no son numéricos, pero tampoco
        # contienen datos del usuario).
        if self.difficulty is not None:
            result["difficulty"] = self.difficulty  # type: ignore[assignment]

        if self.model is not None:
            result["model"] = self.model  # type: ignore[assignment]

        if self.reasoning_effort is not None:
            result["reasoning_effort"] = self.reasoning_effort  # type: ignore[assignment]

        if self.max_tokens is not None:
            result["max_tokens"] = float(self.max_tokens)

        if self.budget_verdict is not None:
            result["budget_verdict"] = self.budget_verdict

        return result

    def log_summary(
        self,
        *,
        reasoning_effort: str | None = None,
    ) -> None:
        """Registra el resumen sin exponer contenido del usuario."""

        logger.info(
            "Sapientia request metrics | effort=%s | metrics=%s",
            reasoning_effort,
            self.summary(),
        )