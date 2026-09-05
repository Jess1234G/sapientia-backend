"""
Pruebas del presupuesto global por solicitud.
"""

import pytest

from app.services.deepseek.request_budget import (
    RequestBudget,
    RequestBudgetConfig,
)


def test_budget_allows_first_attempt():
    budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=2048,
            max_attempts=1,
        )
    )

    assert budget.can_start(2048) is True

    budget.reserve(2048)

    assert budget.reserved_generation_tokens == 2048
    assert budget.attempts_started == 1
    assert budget.remaining_generation_tokens == 0
    assert budget.remaining_attempts == 0


def test_budget_rejects_attempt_beyond_token_limit():
    budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=2048,
            max_attempts=2,
        )
    )

    assert budget.can_start(2049) is False


def test_budget_rejects_attempt_after_max_attempts():
    budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=4096,
            max_attempts=1,
        )
    )

    budget.reserve(2048)

    assert budget.can_start(1024) is False


def test_budget_allows_two_attempts_within_global_budget():
    budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=4096,
            max_attempts=2,
        )
    )

    assert budget.can_start(2048) is True

    budget.reserve(2048)

    assert budget.can_start(1536) is True

    budget.reserve(1536)

    assert budget.reserved_generation_tokens == 3584
    assert budget.remaining_generation_tokens == 512
    assert budget.remaining_attempts == 0


def test_budget_rejects_second_attempt_that_exceeds_global_budget():
    budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=4096,
            max_attempts=2,
        )
    )

    budget.reserve(2048)

    assert budget.can_start(2049) is False

    with pytest.raises(ValueError):
        budget.reserve(2049)


def test_budget_rejects_zero_or_negative_reservations():
    budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=4096,
            max_attempts=2,
        )
    )

    assert budget.can_start(0) is False
    assert budget.can_start(-1) is False

    with pytest.raises(ValueError):
        budget.reserve(0)

    with pytest.raises(ValueError):
        budget.reserve(-1)


def test_budget_configuration_rejects_invalid_values():
    with pytest.raises(ValueError):
        RequestBudgetConfig(
            max_generation_tokens=0,
        )

    with pytest.raises(ValueError):
        RequestBudgetConfig(
            max_generation_tokens=-1,
        )

    with pytest.raises(ValueError):
        RequestBudgetConfig(
            max_generation_tokens=4096,
            max_attempts=0,
        )


def test_budget_reset():
    budget = RequestBudget(
        RequestBudgetConfig(
            max_generation_tokens=4096,
            max_attempts=2,
        )
    )

    budget.reserve(2048)

    budget.reset_reservation()

    assert budget.reserved_generation_tokens == 0
    assert budget.attempts_started == 0
    assert budget.remaining_generation_tokens == 4096
    assert budget.remaining_attempts == 2
