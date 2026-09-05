"""
Pruebas del BudgetGuard.
"""

from time import perf_counter

from app.services.deepseek.budget_guard import (
    BudgetGuard,
    BudgetLimits,
    BudgetVerdict,
)
from app.services.deepseek.client import (
    StreamMetrics,
)


def make_metrics(
    *,
    finish_reason: str | None = "stop",
    has_content: bool = True,
) -> StreamMetrics:
    """
    Construye métricas controladas para las pruebas.
    """

    metrics = StreamMetrics()

    metrics.request_started_at = 100.0

    if has_content:
        metrics.first_token_at = 101.0

    metrics.completed_at = 102.0
    metrics.finish_reason = finish_reason

    return metrics


def test_budget_guard_accepts_normal_completion():
    metrics = make_metrics()

    guard = BudgetGuard()

    assert guard.evaluate(metrics) == (
        BudgetVerdict.OK
    )


def test_budget_guard_detects_truncation():
    metrics = make_metrics(
        finish_reason="length",
    )

    guard = BudgetGuard()

    assert guard.evaluate(metrics) == (
        BudgetVerdict.TRUNCATED
    )


def test_budget_guard_detects_no_content():
    metrics = make_metrics(
        finish_reason="stop",
        has_content=False,
    )

    guard = BudgetGuard()

    assert guard.evaluate(metrics) == (
        BudgetVerdict.NO_CONTENT
    )


def test_budget_guard_detects_ttfc_exceeded():
    metrics = make_metrics()

    guard = BudgetGuard(
        BudgetLimits(
            max_ttfc_ms=500.0,
        )
    )

    assert guard.evaluate(metrics) == (
        BudgetVerdict.TTFC_EXCEEDED
    )


def test_budget_guard_detects_total_exceeded():
    metrics = make_metrics()

    guard = BudgetGuard(
        BudgetLimits(
            max_total_ms=1500.0,
        )
    )

    assert guard.evaluate(metrics) == (
        BudgetVerdict.TOTAL_EXCEEDED
    )


def test_budget_guard_ignores_unconfigured_limits():
    metrics = make_metrics()

    guard = BudgetGuard()

    assert guard.evaluate(metrics) == (
        BudgetVerdict.OK
    )


def test_ttfc_live_guard_is_false_before_limit():
    metrics = StreamMetrics()
    metrics.request_started_at = (
        perf_counter()
    )

    guard = BudgetGuard(
        BudgetLimits(
            max_ttfc_ms=10_000,
        )
    )

    assert guard.ttfc_exceeded_live(metrics) is False


def test_total_live_guard_is_false_before_limit():
    metrics = StreamMetrics()
    metrics.request_started_at = (
        perf_counter()
    )

    guard = BudgetGuard(
        BudgetLimits(
            max_total_ms=10_000,
        )
    )

    assert guard.total_exceeded_live(metrics) is False
