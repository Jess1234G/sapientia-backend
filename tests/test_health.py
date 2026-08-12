"""
test_health.py — Pruebas del liveness /healthz.
"""
from __future__ import annotations


def test_healthz(client):
    """/healthz responde 200 con estado ok."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_available(client):
    """El schema OpenAPI se expone en desarrollo."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()
