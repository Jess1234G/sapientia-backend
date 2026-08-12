"""
code_runner.py — Protocolo base de ejecución de código Python.

Define la interfaz que debe implementar cualquier runner seguro
(E2B, Pyodide, subprocess aislado...). El worker usa esta abstracción.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ExecutionArtifacts:
    """Artefactos producidos por la ejecución del código."""

    stdout: str = ""
    stderr: str = ""
    files: dict[str, bytes] = field(default_factory=dict)  # nombre → bytes


class CodeRunner(Protocol):
    """Contrato de un runner de código Python seguro."""

    async def execute(self, code: str, timeout_s: int = 60) -> ExecutionArtifacts:
        """Ejecuta código y devuelve stdout, stderr y archivos generados."""
        ...
