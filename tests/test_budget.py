"""
Pruebas de la política central de presupuesto de Sapientia.
"""

from app.services.deepseek.budget import (
    get_budget_policy,
)


def test_simple_policy():
    policy = get_budget_policy(
        "simple"
    )

    assert policy.model == (
        "deepseek-v4-flash"
    )

    assert policy.thinking_enabled is False
    assert policy.reasoning_effort == "high"
    assert policy.max_tokens == 1536


def test_normal_policy():
    policy = get_budget_policy(
        "normal"
    )

    assert policy.model == (
        "deepseek-v4-flash"
    )

    assert policy.thinking_enabled is True
    assert policy.reasoning_effort == "low"
    assert policy.max_tokens == 1536


def test_complex_policy():
    policy = get_budget_policy(
        "complex"
    )

    assert policy.model == (
        "deepseek-v4-pro"
    )

    assert policy.thinking_enabled is True
    assert policy.reasoning_effort == "high"
    assert policy.max_tokens == 2048


def test_unknown_policy_raises_error():
    try:
        get_budget_policy("extreme")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "extreme" in str(exc)
    else:
        raise AssertionError(
            "Se esperaba ValueError."
        )
