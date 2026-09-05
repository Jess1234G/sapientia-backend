from __future__ import annotations

import asyncio

from app.services.sandbox.e2b_service import (
    E2BService,
    SandboxError,
)


async def main() -> int:
    print("=" * 60)
    print("SAPIENTIA — PRUEBA REAL E2B")
    print("=" * 60)

    service = E2BService()

    code = """
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=[0, 1, 2],
        y=[0, 1, 4],
        z=[0, 1, 8],
        mode="lines+markers",
    )
)

fig.update_layout(
    title="Sapientia E2B Test",
)

fig.write_html(
    "figura_3d.html",
    include_plotlyjs="cdn",
)
"""

    try:
        result = await service.run(
            code=code,
            timeout_s=30,
        )

        print()
        print("E2B RESPONDIÓ CORRECTAMENTE")
        print(
            f"Archivos devueltos: {len(result['files'])}"
        )

        for file in result["files"]:
            print(
                f"- {file['name']}: "
                f"{len(file['content'])} bytes"
            )

        return 0

    except SandboxError as exc:
        print()
        print("ERROR CONTROLADO DE E2B")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
