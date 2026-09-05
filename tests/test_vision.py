"""
Pruebas básicas del endpoint /preguntar-vision.
"""

from __future__ import annotations


def test_vision_empty_message(client):
    """
    Una solicitud de visión con mensaje vacío debe rechazarse
    antes de contactar con SimpleTex.
    """

    response = client.post(
        "/preguntar-vision",
        data={
            "mensaje": "",
            "generar_grafico": "false",
        },
        files={
            "archivo": (
                "test.png",
                b"fake-image-content",
                "image/png",
            ),
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"] == (
        "El campo 'mensaje' no puede estar vacío."
    )


def test_vision_rejects_non_image_file(client):
    """
    Un archivo que no sea imagen debe rechazarse antes de
    contactar con SimpleTex.
    """

    response = client.post(
        "/preguntar-vision",
        data={
            "mensaje": "Analiza este archivo.",
            "generar_grafico": "false",
        },
        files={
            "archivo": (
                "documento.txt",
                b"contenido de prueba",
                "text/plain",
            ),
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "El archivo debe ser una imagen." in data["detail"]


def test_vision_empty_file(client):
    """
    Un archivo de imagen vacío debe rechazarse con HTTP 400.
    """

    response = client.post(
        "/preguntar-vision",
        data={
            "mensaje": "Analiza esta imagen.",
            "generar_grafico": "false",
        },
        files={
            "archivo": (
                "vacio.png",
                b"",
                "image/png",
            ),
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"] == (
        "El archivo de imagen está vacío."
    )


def test_vision_missing_simpletex_credentials(
    client,
    monkeypatch,
):
    """
    Si faltan las credenciales de SimpleTex, el endpoint
    debe devolver HTTP 500 antes de intentar llamar a la API.
    """

    import app.main as main_module

    monkeypatch.setattr(
        main_module,
        "DEEPSEEK_API_KEY",
        "test-deepseek-key",
    )

    monkeypatch.delenv(
        "SIMPLETEX_APP_ID",
        raising=False,
    )

    monkeypatch.delenv(
        "SIMPLETEX_APP_SECRET",
        raising=False,
    )

    response = client.post(
        "/preguntar-vision",
        data={
            "mensaje": "Analiza esta imagen.",
            "generar_grafico": "false",
        },
        files={
            "archivo": (
                "test.png",
                b"fake-image-content",
                "image/png",
            ),
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == (
        "Faltan SIMPLETEX_APP_ID o SIMPLETEX_APP_SECRET en .env."
    )


def test_vision_success_without_graph(
    client,
    mock_legacy_deepseek,
    monkeypatch,
):
    """
    Verifica el flujo exitoso de visión sin generar gráfico 3D.

    Todas las dependencias externas se simulan.
    """

    import app.main as main_module
    from types import SimpleNamespace

    # ---------------------------------------------------------
    # Configuración controlada
    # ---------------------------------------------------------

    monkeypatch.setattr(
        main_module,
        "DEEPSEEK_API_KEY",
        "test-deepseek-key",
    )

    monkeypatch.setattr(
        main_module,
        "db",
        None,
    )

    monkeypatch.setenv(
        "SIMPLETEX_APP_ID",
        "test-app-id",
    )

    monkeypatch.setenv(
        "SIMPLETEX_APP_SECRET",
        "test-app-secret",
    )

    # ---------------------------------------------------------
    # Simular respuesta de SimpleTex
    # ---------------------------------------------------------

    fake_simpletex_response = SimpleNamespace(
        status_code=200,
        headers={},
        text='{"status": true}',
        json=lambda: {
            "status": True,
            "res": {
                "latex": r"x^2 + 2x + 1 = 0",
                "text": "x² + 2x + 1 = 0",
            },
        },
    )

    def fake_post(*args, **kwargs):
        return fake_simpletex_response

    monkeypatch.setattr(
        main_module.requests,
        "post",
        fake_post,
    )

    # ---------------------------------------------------------
    # Ejecutar endpoint
    # ---------------------------------------------------------

    response = client.post(
        "/preguntar-vision",
        data={
            "mensaje": "Resuelve esta ecuación.",
            "generar_grafico": "false",
        },
        files={
            "archivo": (
                "ecuacion.png",
                b"fake-image-content",
                "image/png",
            ),
        },
    )

    # ---------------------------------------------------------
    # Validaciones
    # ---------------------------------------------------------

    assert response.status_code == 200

    data = response.json()

    assert data["respuesta"] == (
        "Respuesta simulada de Sapientia para pruebas."
    )

    assert data["texto_extraido_ocr"] == (
        "x² + 2x + 1 = 0"
    )

    assert data["latex_extraido"] == (
        r"x^2 + 2x + 1 = 0"
    )

    assert data["grafico_3d_generado"] is False

    assert data["grafico_3d_html_base64"] is None

    assert data["modelo_deepseek"] == "deepseek-v4-pro"

    assert data["memoria_guardada"] == (
        "⚠️ Memoria no guardada (Firebase no conectado)"
    )
