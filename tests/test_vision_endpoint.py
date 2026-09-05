"""
Pruebas del endpoint POST /api/v1/vision/analyze.
"""

from __future__ import annotations

from app.api.v1.schemas.vision import VisionAnalysisResult
from app.core.security import get_current_user
from app.services.vision.vision_service import (
    VisionServiceError,
    get_vision_service,
)
from app.main import app


class FakeVisionService:
    """VisionService falso para pruebas de endpoint."""

    def __init__(
        self,
        result: VisionAnalysisResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> VisionAnalysisResult:
        self.calls.append(
            {
                "image_bytes": image_bytes,
                "filename": filename,
                "content_type": content_type,
            }
        )

        if self.error is not None:
            raise self.error

        return self.result  # type: ignore[return-value]


def _override_auth() -> None:
    app.dependency_overrides[
        get_current_user
    ] = lambda: "test-user"


def _override_vision(service: FakeVisionService) -> None:
    app.dependency_overrides[
        get_vision_service
    ] = lambda: service


def _clear_overrides() -> None:
    app.dependency_overrides.clear()


def test_vision_endpoint_requires_authentication(
    client,
):
    """Sin token, el endpoint debe devolver 401."""

    response = client.post(
        "/api/v1/vision/analyze",
        files={
            "image": (
                "test.png",
                b"fake-image",
                "image/png",
            ),
        },
    )

    assert response.status_code == 401


def test_vision_endpoint_analyzes_image(
    client,
):
    """
    Con autenticación, el endpoint debe llamar a VisionService
    y devolver texto y fórmulas.
    """

    service = FakeVisionService(
        result=VisionAnalysisResult(
            texto="x cuadrado más uno",
            formulas=["x^2 + 1"],
            tipo="otro",
            descripcion="",
        )
    )

    _override_auth()
    _override_vision(service)

    try:
        response = client.post(
            "/api/v1/vision/analyze",
            files={
                "image": (
                    "ejercicio.png",
                    b"fake-image",
                    "image/png",
                ),
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["texto"] == "x cuadrado más uno"
        assert data["formulas"] == ["x^2 + 1"]
        assert data["tipo"] == "otro"
        assert data["descripcion"] == ""

        assert len(service.calls) == 1
        assert service.calls[0]["filename"] == (
            "ejercicio.png"
        )
        assert service.calls[0]["content_type"] == (
            "image/png"
        )

    finally:
        _clear_overrides()


def test_vision_endpoint_rejects_vision_service_error(
    client,
):
    """Un VisionServiceError debe traducirse a HTTP 400."""

    service = FakeVisionService(
        error=VisionServiceError(
            "El archivo debe ser una imagen."
        )
    )

    _override_auth()
    _override_vision(service)

    try:
        response = client.post(
            "/api/v1/vision/analyze",
            files={
                "image": (
                    "test.png",
                    b"fake-image",
                    "image/png",
                ),
            },
        )

        assert response.status_code == 400

        data = response.json()
        assert data["detail"] == (
            "El archivo debe ser una imagen."
        )

    finally:
        _clear_overrides()
