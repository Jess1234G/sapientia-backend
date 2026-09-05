"""
Pruebas unitarias de VisionService.

No realizan llamadas reales a SimpleTex.
"""

from __future__ import annotations

import pytest

import app.services.vision.vision_service as vision_module
from app.services.vision.vision_service import (
    VisionService,
    VisionServiceError,
)


class FakeSimpleTex:
    """SimpleTex falso para pruebas."""

    def __init__(
        self,
        result: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or {}
        self.error = error
        self.calls: list[dict] = []

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> dict[str, str]:
        self.calls.append(
            {
                "image_bytes": image_bytes,
                "filename": filename,
                "content_type": content_type,
            }
        )

        if self.error is not None:
            raise self.error

        return self.result


@pytest.fixture
def small_image() -> bytes:
    return b"fake-image"


@pytest.mark.asyncio
async def test_vision_service_returns_structured_result(
    small_image,
):
    simpletex = FakeSimpleTex(
        result={
            "text": "2x + 1 = 5",
            "latex": r"2x + 1 = 5",
        }
    )

    service = VisionService(
        simpletex=simpletex
    )

    result = await service.analyze(
        small_image,
        filename="ejercicio.png",
        content_type="image/png",
    )

    assert result.texto == "2x + 1 = 5"
    assert result.formulas == [
        r"2x + 1 = 5"
    ]
    assert result.tipo == "otro"
    assert result.descripcion == ""

    assert len(simpletex.calls) == 1
    assert (
        simpletex.calls[0]["filename"]
        == "ejercicio.png"
    )
    assert (
        simpletex.calls[0]["content_type"]
        == "image/png"
    )


@pytest.mark.asyncio
async def test_vision_service_allows_text_without_latex():
    simpletex = FakeSimpleTex(
        result={
            "text": "texto detectado",
            "latex": "",
        }
    )

    service = VisionService(
        simpletex=simpletex
    )

    result = await service.analyze(
        b"image",
        content_type="image/jpeg",
    )

    assert result.texto == "texto detectado"
    assert result.formulas == []


@pytest.mark.asyncio
async def test_vision_service_allows_latex_without_text():
    simpletex = FakeSimpleTex(
        result={
            "text": "",
            "latex": r"\frac{1}{2}",
        }
    )

    service = VisionService(
        simpletex=simpletex
    )

    result = await service.analyze(
        b"image",
        content_type="image/png",
    )

    assert result.texto == ""
    assert result.formulas == [
        r"\frac{1}{2}"
    ]


@pytest.mark.asyncio
async def test_vision_service_rejects_empty_image():
    simpletex = FakeSimpleTex()

    service = VisionService(
        simpletex=simpletex
    )

    with pytest.raises(
        VisionServiceError,
        match="imagen está vacía",
    ):
        await service.analyze(
            b"",
            content_type="image/png",
        )

    assert simpletex.calls == []


@pytest.mark.asyncio
async def test_vision_service_rejects_non_image():
    simpletex = FakeSimpleTex()

    service = VisionService(
        simpletex=simpletex
    )

    with pytest.raises(
        VisionServiceError,
        match="debe ser una imagen",
    ):
        await service.analyze(
            b"data",
            content_type="application/pdf",
        )

    assert simpletex.calls == []


@pytest.mark.asyncio
async def test_vision_service_rejects_oversized_image(
    monkeypatch,
):
    monkeypatch.setattr(
        vision_module.settings,
        "max_image_size_mb",
        1,
    )

    simpletex = FakeSimpleTex()

    service = VisionService(
        simpletex=simpletex
    )

    oversized = b"x" * (
        1024 * 1024 + 1
    )

    with pytest.raises(
        VisionServiceError,
        match="tamaño máximo permitido",
    ):
        await service.analyze(
            oversized,
            content_type="image/png",
        )

    assert simpletex.calls == []


@pytest.mark.asyncio
async def test_vision_service_wraps_simpletex_error():
    simpletex = FakeSimpleTex(
        error=RuntimeError(
            "SimpleTex falló"
        )
    )

    service = VisionService(
        simpletex=simpletex
    )

    with pytest.raises(
        VisionServiceError,
        match="Error inesperado durante el análisis",
    ):
        await service.analyze(
            b"image",
            content_type="image/png",
        )


@pytest.mark.asyncio
async def test_vision_service_wraps_simpletex_error_type():
    from app.services.vision.simpletex_service import (
        SimpleTexError,
    )

    simpletex = FakeSimpleTex(
        error=SimpleTexError(
            "OCR rechazado"
        )
    )

    service = VisionService(
        simpletex=simpletex
    )

    with pytest.raises(
        VisionServiceError,
        match="Error en el análisis OCR",
    ):
        await service.analyze(
            b"image",
            content_type="image/png",
        )


@pytest.mark.asyncio
async def test_vision_service_uses_configured_size_limit(
    monkeypatch,
):
    monkeypatch.setattr(
        vision_module.settings,
        "max_image_size_mb",
        2,
    )

    simpletex = FakeSimpleTex(
        result={
            "text": "ok",
            "latex": "",
        }
    )

    service = VisionService(
        simpletex=simpletex
    )

    image = b"x" * (
        2 * 1024 * 1024
    )

    result = await service.analyze(
        image,
        content_type="image/png",
    )

    assert result.texto == "ok"


@pytest.mark.asyncio
async def test_vision_service_does_not_invent_type_or_description():
    simpletex = FakeSimpleTex(
        result={
            "text": "ecuación",
            "latex": r"x^2",
        }
    )

    service = VisionService(
        simpletex=simpletex
    )

    result = await service.analyze(
        b"image",
        content_type="image/png",
    )

    assert result.tipo == "otro"
    assert result.descripcion == ""

