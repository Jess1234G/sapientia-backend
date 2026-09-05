"""
Pruebas básicas de disponibilidad del backend Sapientia.
"""

from __future__ import annotations


def test_healthz(client):
    """Healthcheck ligero."""
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health(client):
    """Healthcheck detallado."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "Sapientia API"
    assert "version" in data
    assert "deepseek_model" in data
    assert "firebase" in data
    assert "pinecone" in data
    assert "tavily" in data
    assert "e2b_configured" in data


def test_openapi_available(client):
    """FastAPI expone correctamente OpenAPI."""
    response = client.get("/openapi.json")

    assert response.status_code == 200

    data = response.json()

    assert "openapi" in data
    assert "paths" in data


def test_root(client):
    """La raíz de la API responde correctamente."""
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Sapientia API"
    assert data["status"] == "online"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"