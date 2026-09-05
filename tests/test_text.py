"""
Pruebas del endpoint /preguntar-texto.
"""

from __future__ import annotations


def test_text_empty_message(client):
    """
    Una solicitud de texto vacía debe rechazarse con HTTP 400.
    """

    response = client.post(
        "/preguntar-texto",
        json={
            "mensaje": "",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"] == (
        "El campo 'mensaje' no puede estar vacío."
    )


def test_text_whitespace_message(client):
    """
    Una solicitud compuesta únicamente por espacios también
    debe rechazarse.
    """

    response = client.post(
        "/preguntar-texto",
        json={
            "mensaje": "   ",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"] == (
        "El campo 'mensaje' no puede estar vacío."
    )


def test_text_success(client, mock_legacy_deepseek):
    """
    Una consulta válida devuelve la respuesta simulada de DeepSeek.
    """

    response = client.post(
        "/preguntar-texto",
        json={
            "mensaje": "Explica qué es una derivada.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["respuesta"] == (
        "Respuesta simulada de Sapientia para pruebas."
    )

    assert data["modelo_deepseek"] == "deepseek-v4-pro"

    assert "rag_disponible" in data
    assert "web_disponible" in data