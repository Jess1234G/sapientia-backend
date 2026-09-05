"""
Pruebas de la política declarativa de fallback.
"""

from app.services.deepseek.budget_guard import (
    BudgetVerdict,
)
from app.services.deepseek.fallback_policy import (
    FALLBACK_POLICIES,
    FallbackAttempt,
    FallbackPolicy,
    get_fallback_policy,
)


def test_ok_has_no_fallback():
    policy = get_fallback_policy(
        BudgetVerdict.OK
    )

    assert policy.has_fallback is False
    assert policy.attempts == ()


def test_truncated_policy_is_defined():
    policy = get_fallback_policy(
        BudgetVerdict.TRUNCATED
    )

    assert isinstance(
        policy,
        FallbackPolicy,
    )

    assert isinstance(
        policy.attempts,
        tuple,
    )


def test_no_content_policy_is_defined():
    policy = get_fallback_policy(
        BudgetVerdict.NO_CONTENT
    )

    assert isinstance(
        policy,
        FallbackPolicy,
    )


def test_ttfc_policy_is_defined():
    policy = get_fallback_policy(
        BudgetVerdict.TTFC_EXCEEDED
    )

    assert isinstance(
        policy,
        FallbackPolicy,
    )


def test_total_policy_is_defined():
    policy = get_fallback_policy(
        BudgetVerdict.TOTAL_EXCEEDED
    )

    assert isinstance(
        policy,
        FallbackPolicy,
    )


def test_fallback_attempt_preserves_exact_parameters():
    attempt = FallbackAttempt(
        model="deepseek-v4-flash",
        thinking_enabled=True,
        reasoning_effort="low",
        max_tokens=1536,
    )

    assert attempt.model == (
        "deepseek-v4-flash"
    )

    assert attempt.thinking_enabled is True

    assert attempt.reasoning_effort == "low"

    assert attempt.max_tokens == 1536


def test_fallback_policy_is_immutable():
    policy = FallbackPolicy(
        attempts=(
            FallbackAttempt(
                model="deepseek-v4-flash",
                thinking_enabled=False,
                reasoning_effort="high",
                max_tokens=1536,
            ),
        )
    )

    try:
        policy.attempts = ()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "FallbackPolicy debe ser inmutable."
        )


def test_all_verdicts_have_a_policy():
    for verdict in BudgetVerdict:
        assert verdict in FALLBACK_POLICIES
        assert isinstance(
            FALLBACK_POLICIES[verdict],
            FallbackPolicy,
        )
