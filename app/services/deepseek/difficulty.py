"""
difficulty.py — Clasificación local de dificultad para Sapientia.

No realiza llamadas a modelos ni APIs externas.

La clasificación determina:

1. dificultad estimada;
2. modelo recomendado;
3. esfuerzo de razonamiento recomendado.

Política actual:

    simple  -> deepseek-v4-flash + high
    normal  -> deepseek-v4-flash + high
    complex -> deepseek-v4-pro   + high
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


DifficultyLevel = Literal[
    "simple",
    "normal",
    "complex",
]

ModelName = Literal[
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]

ReasoningEffort = Literal[
    "low",
    "high",
    "max",
]


@dataclass(frozen=True)
class DifficultyResult:
    """
    Resultado de clasificación local.

    level:
        Nivel de dificultad estimado.

    model:
        Modelo recomendado para resolver la consulta.

    reasoning_effort:
        Esfuerzo de razonamiento recomendado.

    score:
        Puntuación interna utilizada por el clasificador.
    """

    level: DifficultyLevel
    model: ModelName
    reasoning_effort: ReasoningEffort
    score: int


# ============================================================
# PATRONES COMPLEJOS
# ============================================================

COMPLEX_PATTERNS = (
    # Matemática avanzada
    r"\bintegral(?:es)?\b",
    r"\bintegral definida\b",
    r"\bintegral doble\b",
    r"\bintegral triple\b",
    r"\becuaci[oó]n diferencial\b",
    r"\becuaciones diferenciales\b",
    r"\btransformada de laplace\b",
    r"\bserie de fourier\b",
    r"\bdiagonaliza(?:r|ción)\b",
    r"\bautovalores?\b",
    r"\bautovectores?\b",
    r"\bjacobiano\b",
    r"\bhessiano\b",

    # Física / ingeniería
    r"\brelatividad\b",
    r"\belectromagnetismo\b",
    r"\btermodin[aá]mica\b",
    r"\bmec[aá]nica cu[aá]ntica\b",
    r"\bmec[aá]nica lagrangiana\b",
    r"\bmec[aá]nica hamiltoniana\b",

    # Programación
    r"\bdebug(?:uear|ging|ger)?\b",
    r"\brefactor(?:izar|ización)?\b",
    r"\barquitectura de software\b",
    r"\bconcurrencia\b",
    r"\bmicroservicios?\b",
    r"\bapi\b.*\bbackend\b",
    r"\bbackend\b.*\bfrontend\b",

    # Razonamiento
    r"\bdemuestra\b",
    r"\bdemostrar\b",
    r"\bdemostraci[oó]n\b",
    r"\bjustifica\b",
    r"\bjustificar\b",
    r"\bderiva\b.*\bexpresi[oó]n\b",
    r"\bcompara\b.*\bmodelos?\b",
    r"\banaliza\b.*\bcasos?\b",
    r"\bpaso a paso\b",
    r"\bresuelve\b.*\bsistema\b",
)


# ============================================================
# SEÑALES
# ============================================================

def _count_math_signals(text: str) -> int:
    """Detecta indicadores matemáticos simples."""

    score = 0

    if re.search(r"[∫∑√∞≈≠≤≥²³]", text):
        score += 2

    if re.search(
        r"\b(?:sin|cos|tan|log|ln|exp)\b",
        text,
        re.IGNORECASE,
    ):
        score += 1

    if re.search(r"[=^_{}]", text):
        score += 1

    if re.search(
        r"\b\d+(?:\.\d+)?\s*[*x·]\s*\b",
        text,
        re.IGNORECASE,
    ):
        score += 1

    return score


def _count_complex_patterns(text: str) -> int:
    """Cuenta indicadores lingüísticos de complejidad."""

    score = 0

    for pattern in COMPLEX_PATTERNS:
        if re.search(
            pattern,
            text,
            re.IGNORECASE,
        ):
            score += 2

    return score


def _classify_level(
    score: int,
    text: str,
) -> DifficultyLevel:
    """
    Convierte la puntuación interna en un nivel.

    Política conservadora:

        score < 2  -> simple
        score 2-3  -> normal
        score >= 4 -> complex

    El objetivo es evitar sobrecargar consultas sencillas.
    """

    if score >= 4:
        return "complex"

    if score >= 2:
        return "normal"

    # Mensajes muy largos o con múltiples preguntas
    # tienen como mínimo dificultad normal.
    word_count = len(text.split())
    question_count = text.count("?")

    if word_count >= 120 or question_count >= 2:
        return "normal"

    return "simple"


def _model_for_level(
    level: DifficultyLevel,
) -> ModelName:
    """
    Selecciona el modelo mínimo que actualmente consideramos
    suficiente para cada nivel.
    """

    if level == "complex":
        return "deepseek-v4-pro"

    return "deepseek-v4-flash"


def classify_difficulty(
    user_message: str,
    vision_text: str = "",
    rag_context: str = "",
) -> DifficultyResult:
    """
    Clasifica localmente una consulta y selecciona el modelo.

    No realiza ninguna llamada externa.

    Política actual:

        simple:
            deepseek-v4-flash + high

        normal:
            deepseek-v4-flash + high

        complex:
            deepseek-v4-pro + high
    """

    message = (
        user_message or ""
    ).strip()

    vision = (
        vision_text or ""
    ).strip()

    rag = (
        rag_context or ""
    ).strip()

    combined = "\n".join(
        part
        for part in (
            message,
            vision,
            rag,
        )
        if part
    )

    if not combined:
        return DifficultyResult(
            level="simple",
            model="deepseek-v4-flash",
            reasoning_effort="high",
            score=0,
        )

    score = 0

    # Señales matemáticas.
    score += _count_math_signals(
        combined
    )

    # Patrones de complejidad.
    score += _count_complex_patterns(
        combined
    )

    # Extensión del contexto.
    word_count = len(
        combined.split()
    )

    if word_count >= 250:
        score += 2
    elif word_count >= 120:
        score += 1

    # Varias preguntas.
    question_count = combined.count("?")

    if question_count >= 3:
        score += 2
    elif question_count == 2:
        score += 1

    # Código.
    if "```" in combined:
        score += 2

    # Contexto visual extenso.
    if len(vision) >= 1500:
        score += 1

    level = _classify_level(
        score,
        combined,
    )

    model = _model_for_level(
        level
    )

    reasoning_effort: ReasoningEffort = (
        "low"
        if level == "normal"
        else "high"
    )

    return DifficultyResult(
        level=level,
        model=model,
        reasoning_effort=reasoning_effort,
        score=score,
    )