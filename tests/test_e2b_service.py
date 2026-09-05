"""
Pruebas unitarias de E2BService.

Estas pruebas NO realizan llamadas reales a E2B.
El sandbox se sustituye por un fake controlado.
"""

from __future__ import annotations

import pytest

import app.services.sandbox.e2b_service as e2b_module
from app.services.sandbox.code_runner import ExecutionArtifacts
from app.services.sandbox.e2b_service import (
    E2BService,
    SandboxError,
)


class FakeFileSystem:
    """Filesystem falso del sandbox."""

    def __init__(
        self,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.files = files or {}

    async def list(
        self,
        path: str,
        depth: int = 1,
        **kwargs,
    ):
        return [
            type(
                "Entry",
                (),
                {
                    "name": name,
                    "path": name,
                },
            )
            for name in self.files
        ]

    async def read(
        self,
        path: str,
        format: str = "text",
        **kwargs,
    ):
        if path not in self.files:
            raise FileNotFoundError(path)

        content = self.files[path]

        if format == "bytes":
            return content

        return content.decode("utf-8")


class FakeExecution:
    """Resultado falso de run_code()."""

    def __init__(
        self,
        *,
        error=None,
        text: str = "",
    ) -> None:
        self.error = error
        self.text = text
        self.results = []
        self.logs = None
        self.execution_count = 1


class FakeSandbox:
    """Sandbox falso para pruebas."""

    def __init__(
        self,
        *,
        execution: FakeExecution,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.execution = execution
        self.files = FakeFileSystem(files)
        self.killed = False
        self.received_code: str | None = None
        self.received_timeout: float | None = None

    async def run_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        request_timeout: float | None = None,
        **kwargs,
    ):
        self.received_code = code
        self.received_timeout = timeout
        return self.execution

    async def kill(self) -> bool:
        self.killed = True
        return True


class FakeAsyncSandbox:
    """Fábrica falsa compatible con AsyncSandbox."""

    sandbox: FakeSandbox | None = None
    received_template: str | None = None
    received_timeout: float | None = None
    received_api_key: str | None = None

    @classmethod
    async def create(
        cls,
        *,
        template=None,
        timeout=None,
        **kwargs,
    ):
        cls.received_template = template
        cls.received_timeout = timeout
        cls.received_api_key = kwargs.get(
            "api_key"
        )

        if cls.sandbox is None:
            raise RuntimeError(
                "FakeAsyncSandbox no configurado."
            )

        return cls.sandbox


@pytest.fixture
def patch_sandbox(monkeypatch):
    """
    Sustituye AsyncSandbox por nuestro sandbox falso.
    """

    monkeypatch.setattr(
        e2b_module,
        "AsyncSandbox",
        FakeAsyncSandbox,
        raising=False,
    )

    return FakeAsyncSandbox


@pytest.fixture
def configure_e2b(monkeypatch):
    """Configura una API key y template de prueba."""

    monkeypatch.setattr(
        e2b_module.settings,
        "e2b_api_key",
        "test-api-key",
    )

    monkeypatch.setattr(
        e2b_module.settings,
        "e2b_template",
        "test-template",
    )


@pytest.mark.asyncio
async def test_e2b_requires_api_key(
    configure_e2b,
):
    """
    Sin API key el servicio debe rechazar la ejecución.
    """

    # El fixture pone una key válida; la sobreescribimos.
    e2b_module.settings.e2b_api_key = ""

    service = E2BService()

    with pytest.raises(SandboxError):
        await service.run(
            "print('hola')"
        )


@pytest.mark.asyncio
async def test_e2b_runs_code_and_returns_3d_html(
    configure_e2b,
    patch_sandbox,
):
    """
    Una ejecución correcta debe recuperar figura_3d.html.
    """

    html = (
        b"<html><body>"
        b"Sapientia 3D"
        b"</body></html>"
    )

    sandbox = FakeSandbox(
        execution=FakeExecution(),
        files={
            "figura_3d.html": html,
        },
    )

    patch_sandbox.sandbox = sandbox

    service = E2BService()

    result = await service.run(
        """
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.write_html(
            "figura_3d.html",
            include_plotlyjs="cdn",
        )
        """
    )

    assert result["files"]
    assert result["files"][0]["name"] == (
        "figura_3d.html"
    )
    assert result["files"][0]["content"] == html

    assert (
        sandbox.received_code is not None
    )

    assert sandbox.killed is True

    assert (
        patch_sandbox.received_template
        == "test-template"
    )

    assert (
        patch_sandbox.received_api_key
        == "test-api-key"
    )


@pytest.mark.asyncio
async def test_e2b_raises_on_execution_error(
    configure_e2b,
    patch_sandbox,
):
    """
    Un error del intérprete debe convertirse en SandboxError.
    """

    sandbox = FakeSandbox(
        execution=FakeExecution(
            error=RuntimeError(
                "error de ejecución"
            )
        ),
    )

    patch_sandbox.sandbox = sandbox

    service = E2BService()

    with pytest.raises(SandboxError):
        await service.run(
            "raise RuntimeError('error')"
        )

    assert sandbox.killed is True


@pytest.mark.asyncio
async def test_e2b_raises_when_3d_file_is_missing(
    configure_e2b,
    patch_sandbox,
):
    """
    Una ejecución correcta sin figura_3d.html debe fallar.
    """

    sandbox = FakeSandbox(
        execution=FakeExecution(),
        files={},
    )

    patch_sandbox.sandbox = sandbox

    service = E2BService()

    with pytest.raises(SandboxError):
        await service.run(
            "print('sin gráfico')"
        )

    assert sandbox.killed is True


@pytest.mark.asyncio
async def test_e2b_kills_sandbox_after_success(
    configure_e2b,
    patch_sandbox,
):
    """
    El sandbox debe cerrarse después de una ejecución correcta.
    """

    sandbox = FakeSandbox(
        execution=FakeExecution(),
        files={
            "figura_3d.html": (
                b"<html>ok</html>"
            ),
        },
    )

    patch_sandbox.sandbox = sandbox

    service = E2BService()

    await service.run(
        "print('ok')"
    )

    assert sandbox.killed is True


@pytest.mark.asyncio
async def test_e2b_execute_returns_execution_artifacts(
    configure_e2b,
    patch_sandbox,
):
    sandbox = FakeSandbox(
        execution=FakeExecution(),
        files={
            "figura_3d.html": b"<html>test</html>",
        },
    )

    patch_sandbox.sandbox = sandbox

    service = E2BService()

    result = await service.execute(
        code="print('ok')",
        timeout_s=30,
    )

    assert isinstance(
        result,
        ExecutionArtifacts,
    )

    assert result.files == {
        "figura_3d.html": b"<html>test</html>",
    }


@pytest.mark.asyncio
async def test_e2b_execute_preserves_3d_file(
    configure_e2b,
    patch_sandbox,
):
    sandbox = FakeSandbox(
        execution=FakeExecution(),
        files={
            "figura_3d.html": b"3d-content",
        },
    )

    patch_sandbox.sandbox = sandbox

    service = E2BService()

    result = await service.execute(
        code="print('graph')",
    )

    assert "figura_3d.html" in result.files
    assert (
        result.files["figura_3d.html"]
        == b"3d-content"
    )


@pytest.mark.asyncio
async def test_e2b_execute_propagates_sandbox_error(
    configure_e2b,
    patch_sandbox,
):
    sandbox = FakeSandbox(
        execution=FakeExecution(
            error=RuntimeError(
                "execution failed"
            )
        ),
    )

    patch_sandbox.sandbox = sandbox

    service = E2BService()

    with pytest.raises(
        SandboxError,
        match="Error ejecutando código en E2B",
    ):
        await service.execute(
            code="raise Exception()",
        )

    assert sandbox.killed is True

