"""
request_budget.py — Presupuesto global de generación por solicitud.

Este componente no realiza llamadas externas.

Su responsabilidad es impedir que una única solicitud pueda
encadenar indefinidamente reintentos/fallbacks consumiendo
un presupuesto de generación superior al permitido.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestBudgetConfig:
    """
    Configuración del presupuesto global.

    max_generation_tokens:
        Máximo de tokens de generación que estamos dispuestos
        a reservar para todos los intentos combinados.

    max_attempts:
        Máximo absoluto de intentos permitidos.
    """

    max_generation_tokens: int
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if self.max_generation_tokens <= 0:
            raise ValueError(
                "max_generation_tokens debe ser mayor que 0."
            )

        if self.max_attempts <= 0:
            raise ValueError(
                "max_attempts debe ser mayor que 0."
            )


@dataclass
class RequestBudget:
    """
    Estado mutable del presupuesto de una solicitud.

    El presupuesto se consume por reserva de max_tokens de cada
    intento, no por tokens realmente utilizados.

    Esto es deliberadamente conservador: garantiza que no iniciemos
    una segunda llamada cuyo peor caso supere el presupuesto global.
    """

    config: RequestBudgetConfig

    reserved_generation_tokens: int = 0
    attempts_started: int = 0

    @property
    def remaining_generation_tokens(self) -> int:
        """Tokens de generación todavía disponibles."""

        return max(
            0,
            self.config.max_generation_tokens
            - self.reserved_generation_tokens,
        )

    @property
    def remaining_attempts(self) -> int:
        """Intentos todavía disponibles."""

        return max(
            0,
            self.config.max_attempts
            - self.attempts_started,
        )

    def can_start(
        self,
        max_tokens: int,
    ) -> bool:
        """
        Determina si un nuevo intento cabe dentro del presupuesto.

        No modifica el estado.
        """

        if max_tokens <= 0:
            return False

        if self.remaining_attempts <= 0:
            return False

        return (
            self.reserved_generation_tokens
            + max_tokens
            <= self.config.max_generation_tokens
        )

    def reserve(
        self,
        max_tokens: int,
    ) -> None:
        """
        Reserva el presupuesto para un nuevo intento.

        Raises:
            ValueError:
                Si el intento no cabe en el presupuesto.
        """

        if not self.can_start(max_tokens):
            raise ValueError(
                "El intento supera el presupuesto global "
                "disponible de la solicitud."
            )

        self.reserved_generation_tokens += max_tokens
        self.attempts_started += 1

    def reset_reservation(
        self,
    ) -> None:
        """
        Reinicia el estado.

        Útil para tests y para construir una nueva solicitud.

        No debe utilizarse durante una solicitud real.
        """

        self.reserved_generation_tokens = 0
        self.attempts_started = 0
