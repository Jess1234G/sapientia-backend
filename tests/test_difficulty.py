"""
Pruebas del clasificador local de dificultad.
"""

from app.services.deepseek.difficulty import (
    classify_difficulty,
)


def test_simple_question_uses_flash():
    result = classify_difficulty(
        "¿Cuánto es 2 + 2?"
    )

    assert result.level == "simple"
    assert result.model == (
        "deepseek-v4-flash"
    )
    assert result.reasoning_effort == "high"


def test_basic_definition_uses_flash():
    result = classify_difficulty(
        "¿Qué es la fotosíntesis?"
    )

    assert result.level == "simple"
    assert result.model == (
        "deepseek-v4-flash"
    )
    assert result.reasoning_effort == "high"


def test_normal_physics_problem_uses_flash():
    result = classify_difficulty(
        """
        Un automóvil parte del reposo y acelera
        de manera constante a 3 m/s² durante 8 segundos.
        Calcula la velocidad final y la distancia recorrida.
        """
    )

    assert result.level == "normal"
    assert result.model == (
        "deepseek-v4-flash"
    )
    assert result.reasoning_effort == "low"


def test_normal_math_problem_uses_flash():
    result = classify_difficulty(
        """
        Resuelve la ecuación cuadrática:
        2x^2 - 7x + 3 = 0.
        Explica el procedimiento.
        """
    )

    assert result.level == "normal"
    assert result.model == (
        "deepseek-v4-flash"
    )
    assert result.reasoning_effort == "low"


def test_complex_integral_uses_pro():
    result = classify_difficulty(
        "Resuelve esta integral definida y "
        "explica cada paso del procedimiento."
    )

    assert result.level == "complex"
    assert result.model == (
        "deepseek-v4-pro"
    )
    assert result.reasoning_effort == "high"


def test_differential_equation_uses_pro():
    result = classify_difficulty(
        "Resuelve la ecuación diferencial y "
        "demuestra la solución general."
    )

    assert result.level == "complex"
    assert result.model == (
        "deepseek-v4-pro"
    )
    assert result.reasoning_effort == "high"


def test_long_programming_problem_uses_pro():
    result = classify_difficulty(
        """
        Diseña y depura una API backend completa con
        autenticación JWT, concurrencia, manejo de errores,
        persistencia y pruebas automatizadas.
        """
    )

    assert result.level == "complex"
    assert result.model == (
        "deepseek-v4-pro"
    )
    assert result.reasoning_effort == "high"


def test_math_context_can_increase_difficulty():
    result = classify_difficulty(
        "Resuelve el problema.",
        vision_text=(
            "∫₀∞ x² e^{-x} dx = ? "
            "Demuestra el resultado y explica cada paso."
        ),
    )

    assert result.level == "complex"
    assert result.model == (
        "deepseek-v4-pro"
    )
    assert result.reasoning_effort == "high"


def test_empty_question_defaults_to_flash():
    result = classify_difficulty("")

    assert result.level == "simple"
    assert result.model == (
        "deepseek-v4-flash"
    )
    assert result.reasoning_effort == "high"
    assert result.score == 0