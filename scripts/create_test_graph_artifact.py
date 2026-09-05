import asyncio

from app.services.firebase.firestore_service import (
    get_firestore_service,
)


async def main() -> None:
    firestore = get_firestore_service()

    artifact_id = await firestore.create_graph_artifact(
        user_id="test-real-user",
        conversation_id="test-real-conversation",
        code=(
            "import plotly.graph_objects as go\n"
            "fig = go.Figure(\n"
            "    data=[go.Scatter3d(\n"
            "        x=[1, 2, 3],\n"
            "        y=[1, 4, 9],\n"
            "        z=[1, 8, 27],\n"
            "    )]\n"
            ")\n"
            "fig.write_html(\n"
            "    'figura_3d.html',\n"
            "    include_plotlyjs='cdn'\n"
            ")\n"
        ),
        status="pending",
    )

    print(f"ARTIFACT_ID={artifact_id}")

    artifact = await firestore.get_graph_artifact(
        artifact_id
    )

    print("ARTIFACT:")
    print(artifact)


if __name__ == "__main__":
    asyncio.run(main())
