"""
budget.py — Política central de modelo y presupuesto de Sapientia.

Este módulo no realiza llamadas externas.

Define qué recursos utilizar para cada nivel de dificultad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.services.deepseek.client import ReasoningEffort
from app.services.deepseek.difficulty import DifficultyLevel


ModelName = Literal[
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]


@dataclass(frozen=True)
class BudgetPolicy:
    """
    Política técnica asociada a un nivel de dificultad.

    model:
        Modelo que debe utilizarse.

    thinking_enabled:
        Si el modo thinking debe estar habilitado.

    reasoning_effort:
        Nivel de razonamiento enviado al cliente.

    max_tokens:
        Presupuesto máximo de generación.

    Nota:
        max_tokens representa el presupuesto de generación,
        incluyendo el razonamiento cuando thinking está habilitado.
    """

    model: ModelName
    thinking_enabled: bool
    reasoning_effort: ReasoningEffort
    max_tokens: int


BUDGET_POLICIES: dict[
    DifficultyLevel,
    BudgetPolicy,
] = {
    "simple": BudgetPolicy(
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort="high",
        max_tokens=1536,
    ),
    "normal": BudgetPolicy(
        model="deepseek-v4-flash",
        thinking_enabled=True,
        reasoning_effort="low",
        max_tokens=1536,
    ),
    "complex": BudgetPolicy(
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=2048,
    ),
}


def get_budget_policy(
    level: DifficultyLevel,
) -> BudgetPolicy:
    """
    Obtiene la política correspondiente al nivel de dificultad.
    """

    try:
        return BUDGET_POLICIES[level]
    except KeyError as exc:
        raise ValueError(
            f"No existe una política de presupuesto para "
            f"el nivel '{level}'."
        ) from exc
