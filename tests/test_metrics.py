"""
Pruebas de las métricas internas de Sapientia.
"""

from __future__ import annotations

from app.core.metrics import RequestMetrics


def test_metrics_can_measure_first_token():
    metrics = RequestMetrics()

    metrics.mark_first_token()

    assert metrics.first_token_at is not None
    assert (
        metrics.time_to_first_token_ms
        is not None
    )

    assert (
        metrics.time_to_first_token_ms >= 0
    )


def test_metrics_can_measure_phases():
    metrics = RequestMetrics()

    started_at = metrics.started_at

    metrics.mark_phase(
        "memory",
        started_at,
    )

    summary = metrics.summary()

    assert "memory_ms" in summary
    assert summary["memory_ms"] is not None
    assert summary["memory_ms"] >= 0


def test_metrics_can_finish_request():
    metrics = RequestMetrics()

    metrics.mark_first_token()
    metrics.finish()

    assert metrics.total_ms is not None
    assert metrics.total_ms >= 0

    assert (
        metrics.time_to_first_token_ms
        is not None
    )

    assert (
        metrics.total_ms
        >= metrics.time_to_first_token_ms
    )


def test_metrics_can_store_routing_metadata():
    """Las métricas deben conservar la decisión de routing."""

    metrics = RequestMetrics()

    metrics.set_routing(
        difficulty="normal",
        model="deepseek-v4-flash",
        reasoning_effort="high",
        max_tokens=2048,
    )

    assert metrics.difficulty == "normal"
    assert metrics.model == "deepseek-v4-flash"
    assert metrics.reasoning_effort == "high"
    assert metrics.max_tokens == 2048

    summary = metrics.summary()

    assert summary["difficulty"] == "normal"
    assert summary["model"] == "deepseek-v4-flash"
    assert summary["reasoning_effort"] == "high"
    assert summary["max_tokens"] == 2048


def test_metrics_summary_does_not_contain_user_content():
    """El resumen de métricas no debe incluir contenido de usuario."""

    metrics = RequestMetrics()

    metrics.set_routing(
        difficulty="simple",
        model="deepseek-v4-flash",
        reasoning_effort="high",
        max_tokens=1024,
    )

    summary = metrics.summary()

    serialized = str(summary)

    assert "usuario" not in serialized.lower()
    assert "mensaje" not in serialized.lower()
    assert "pregunta" not in serialized.lower()


def test_metrics_can_store_budget_verdict():
    metrics = RequestMetrics()

    metrics.set_budget_verdict(
        "truncated"
    )

    assert metrics.budget_verdict == "truncated"

    summary = metrics.summary()

    assert summary["budget_verdict"] == (
        "truncated"
    )


def test_metrics_budget_verdict_is_optional():
    metrics = RequestMetrics()

    assert metrics.budget_verdict is None

    summary = metrics.summary()

    assert "budget_verdict" not in summary
