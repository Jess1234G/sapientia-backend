"""
Pruebas del FallbackEngine.
"""

from app.services.deepseek.budget_guard import (
    BudgetVerdict,
)
from app.services.deepseek.fallback_engine import (
    FallbackEngine,
)
from app.services.deepseek.fallback_policy import (
    FallbackAttempt,
    FallbackPolicy,
    FALLBACK_POLICIES,
)
from app.services.deepseek.request_budget import (
    RequestBudget,
    RequestBudgetConfig,
)


def make_budget(
    *,
    max_generation_tokens: int,
    max_attempts: int,
) -> RequestBudget:
    return RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=(
                max_generation_tokens
            ),
            max_attempts=max_attempts,
        )
    )


def test_engine_returns_none_when_policy_is_empty():
    budget = make_budget(
        max_generation_tokens=4096,
        max_attempts=2,
    )

    engine = FallbackEngine(
        request_budget=budget,
    )

    result = engine.next_attempt(
        BudgetVerdict.TRUNCATED
    )

    assert result is None

    assert budget.attempts_started == 0
    assert budget.reserved_generation_tokens == 0


def test_engine_does_not_fallback_after_ok():
    budget = make_budget(
        max_generation_tokens=4096,
        max_attempts=2,
    )

    engine = FallbackEngine(
        request_budget=budget,
    )

    result = engine.next_attempt(
        BudgetVerdict.OK
    )

    assert result is None


def test_engine_reserves_first_allowed_fallback():
    original = FALLBACK_POLICIES[
        BudgetVerdict.TRUNCATED
    ]

    FALLBACK_POLICIES[
        BudgetVerdict.TRUNCATED
    ] = FallbackPolicy(
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
        budget = make_budget(
            max_generation_tokens=2048,
            max_attempts=1,
        )

        engine = FallbackEngine(
            request_budget=budget,
        )

        result = engine.next_attempt(
            BudgetVerdict.TRUNCATED
        )

        assert result is not None
        assert result.model == (
            "deepseek-v4-flash"
        )
        assert result.max_tokens == 1536

        assert budget.attempts_started == 1
        assert (
            budget.reserved_generation_tokens
            == 1536
        )

    finally:
        FALLBACK_POLICIES[
            BudgetVerdict.TRUNCATED
        ] = original


def test_engine_rejects_fallback_that_exceeds_budget():
    original = FALLBACK_POLICIES[
        BudgetVerdict.TRUNCATED
    ]

    FALLBACK_POLICIES[
        BudgetVerdict.TRUNCATED
    ] = FallbackPolicy(
        attempts=(
            FallbackAttempt(
                model="deepseek-v4-pro",
                thinking_enabled=True,
                reasoning_effort="high",
                max_tokens=4096,
            ),
        )
    )

    try:
        budget = make_budget(
            max_generation_tokens=2048,
            max_attempts=2,
        )

        engine = FallbackEngine(
            request_budget=budget,
        )

        result = engine.next_attempt(
            BudgetVerdict.TRUNCATED
        )

        assert result is None

        assert budget.attempts_started == 0
        assert (
            budget.reserved_generation_tokens
            == 0
        )

    finally:
        FALLBACK_POLICIES[
            BudgetVerdict.TRUNCATED
        ] = original


def test_engine_uses_next_fallback_when_first_does_not_fit():
    original = FALLBACK_POLICIES[
        BudgetVerdict.NO_CONTENT
    ]

    FALLBACK_POLICIES[
        BudgetVerdict.NO_CONTENT
    ] = FallbackPolicy(
        attempts=(
            FallbackAttempt(
                model="deepseek-v4-pro",
                thinking_enabled=True,
                reasoning_effort="high",
                max_tokens=4096,
            ),
            FallbackAttempt(
                model="deepseek-v4-flash",
                thinking_enabled=False,
                reasoning_effort="high",
                max_tokens=1024,
            ),
        )
    )

    try:
        budget = make_budget(
            max_generation_tokens=2048,
            max_attempts=2,
        )

        engine = FallbackEngine(
            request_budget=budget,
        )

        result = engine.next_attempt(
            BudgetVerdict.NO_CONTENT
        )

        assert result is not None
        assert result.model == (
            "deepseek-v4-flash"
        )
        assert result.max_tokens == 1024

        assert budget.attempts_started == 1
        assert (
            budget.reserved_generation_tokens
            == 1024
        )

    finally:
        FALLBACK_POLICIES[
            BudgetVerdict.NO_CONTENT
        ] = original
