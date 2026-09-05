"""
Pruebas unitarias de SimpleTexService.

No realizan llamadas reales a SimpleTex.
"""

from __future__ import annotations

import pytest
import requests

import app.services.vision.simpletex_service as simpletex_module
from app.services.vision.simpletex_service import (
    SimpleTexError,
    SimpleTexService,
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload=None,
        text: str = "",
        headers=None,
        json_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise ValueError("JSON inválido")
        return self._payload


@pytest.fixture
def configure_simpletex(monkeypatch):
    monkeypatch.setattr(
        simpletex_module.settings,
        "simpletex_app_id",
        "test-app-id",
    )
    monkeypatch.setattr(
        simpletex_module.settings,
        "simpletex_app_secret",
        "test-secret",
    )
    monkeypatch.setattr(
        simpletex_module.settings,
        "simpletex_api_url",
        "https://simpletex.test/api/latex_ocr",
    )


@pytest.mark.asyncio
async def test_simpletex_requires_app_id(monkeypatch):
    monkeypatch.setattr(
        simpletex_module.settings,
        "simpletex_app_id",
        "",
    )
    monkeypatch.setattr(
        simpletex_module.settings,
        "simpletex_app_secret",
        "secret",
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="SIMPLETEX_APP_ID",
    ):
        await service.analyze(
            b"image"
        )


@pytest.mark.asyncio
async def test_simpletex_requires_app_secret(monkeypatch):
    monkeypatch.setattr(
        simpletex_module.settings,
        "simpletex_app_id",
        "app-id",
    )
    monkeypatch.setattr(
        simpletex_module.settings,
        "simpletex_app_secret",
        "",
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="SIMPLETEX_APP_SECRET",
    ):
        await service.analyze(
            b"image"
        )


@pytest.mark.asyncio
async def test_simpletex_rejects_empty_image(
    configure_simpletex,
):
    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="imagen está vacía",
    ):
        await service.analyze(b"")


@pytest.mark.asyncio
async def test_simpletex_success(
    configure_simpletex,
    monkeypatch,
):
    captured = {}

    def fake_post(
        url,
        headers,
        files,
        timeout,
    ):
        captured["url"] = url
        captured["headers"] = headers
        captured["files"] = files
        captured["timeout"] = timeout

        return FakeResponse(
            payload={
                "status": True,
                "res": {
                    "latex": r"x^2 + 1",
                    "text": "x cuadrado más uno",
                },
            }
        )

    monkeypatch.setattr(
        simpletex_module.requests,
        "post",
        fake_post,
    )

    service = SimpleTexService()

    result = await service.analyze(
        b"fake-image",
        filename="test.png",
        content_type="image/png",
    )

    assert result == {
        "text": "x cuadrado más uno",
        "latex": "x^2 + 1",
    }

    assert (
        captured["url"]
        == "https://simpletex.test/api/latex_ocr"
    )

    assert captured["timeout"] == (
        10,
        60,
    )

    assert (
        captured["headers"]["app-id"]
        == "test-app-id"
    )

    assert (
        len(captured["headers"]["sign"]) == 32
    )


@pytest.mark.asyncio
async def test_simpletex_timeout(
    configure_simpletex,
    monkeypatch,
):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(
        simpletex_module.requests,
        "post",
        fake_post,
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="Tiempo de espera",
    ):
        await service.analyze(
            b"image"
        )


@pytest.mark.asyncio
async def test_simpletex_request_error(
    configure_simpletex,
    monkeypatch,
):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.RequestException(
            "network"
        )

    monkeypatch.setattr(
        simpletex_module.requests,
        "post",
        fake_post,
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="No fue posible conectar",
    ):
        await service.analyze(
            b"image"
        )


@pytest.mark.asyncio
async def test_simpletex_http_error(
    configure_simpletex,
    monkeypatch,
):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            status_code=403,
            text="forbidden",
            headers={
                "CF-Ray": "test-ray"
            },
        )

    monkeypatch.setattr(
        simpletex_module.requests,
        "post",
        fake_post,
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="HTTP 403",
    ):
        await service.analyze(
            b"image"
        )


@pytest.mark.asyncio
async def test_simpletex_invalid_json(
    configure_simpletex,
    monkeypatch,
):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            json_error=True,
        )

    monkeypatch.setattr(
        simpletex_module.requests,
        "post",
        fake_post,
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="JSON válido",
    ):
        await service.analyze(
            b"image"
        )


@pytest.mark.asyncio
async def test_simpletex_rejected_response(
    configure_simpletex,
    monkeypatch,
):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            payload={
                "status": False,
                "res": {},
            }
        )

    monkeypatch.setattr(
        simpletex_module.requests,
        "post",
        fake_post,
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="rechazó",
    ):
        await service.analyze(
            b"image"
        )


@pytest.mark.asyncio
async def test_simpletex_empty_result(
    configure_simpletex,
    monkeypatch,
):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            payload={
                "status": True,
                "res": {},
            }
        )

    monkeypatch.setattr(
        simpletex_module.requests,
        "post",
        fake_post,
    )

    service = SimpleTexService()

    with pytest.raises(
        SimpleTexError,
        match="no devolvió texto ni LaTeX",
    ):
        await service.analyze(
            b"image"
        )

