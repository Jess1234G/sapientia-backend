"""
firestore_service.py — CRUD de users, conversations y graph_artifacts.

Usa el SDK Admin de Firebase para persistir en Firestore.
Colecciones: `users`, `conversations`, `graph_artifacts`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.cloud.firestore import Client as FirestoreClient

from app.utils.ids import new_id

logger = logging.getLogger(__name__)


class FirestoreService:
    """Acceso a Firestore (persistencia de negocio)."""

    def __init__(self, client: FirestoreClient) -> None:
        self.client = client

    # ---------- users ----------
    async def upsert_user(self, uid: str, email: str, name: str, picture: str) -> dict:
        """Crea o actualiza el documento users/{uid} y lo devuelve."""
        now = datetime.now(timezone.utc).isoformat()
        doc_ref = self.client.collection("users").document(uid)
        doc = doc_ref.get()
        data = doc.to_dict() if doc.exists else {}

        updated = {
            **data,
            "uid": uid,
            "email": email or data.get("email"),
            "display_name": name or data.get("display_name"),
            "photo_url": picture or data.get("photo_url"),
            "updated_at": now,
        }
        updated.setdefault("created_at", now)
        updated.setdefault("carrera", data.get("carrera", ""))
        updated.setdefault("semestre", data.get("semestre", 0))
        doc_ref.set(updated, merge=True)
        return updated

    async def get_user(self, uid: str) -> dict | None:
        """Devuelve el documento users/{uid} o None."""
        doc = self.client.collection("users").document(uid).get()
        return doc.to_dict() if doc.exists else None

    # ---------- conversations ----------
    async def create_conversation(self, user_id: str, title: str) -> str:
        """Crea una conversación y devuelve su ID."""
        now = datetime.now(timezone.utc).isoformat()
        ref = self.client.collection("conversations").document()
        ref.set(
            {
                "conversation_id": ref.id,
                "user_id": user_id,
                "title": title,
                "messages": [],
                "attachments": [],
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
        )
        return ref.id

    async def add_message(self, conversation_id: str, message: dict) -> None:
        """Añade un mensaje a la conversación y actualiza updated_at."""
        ref = self.client.collection("conversations").document(conversation_id)
        ref.update(
            {
                "messages": message,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def list_conversations(self, user_id: str) -> list[dict]:
        """Devuelve las conversaciones del usuario ordenadas por fecha."""
        query = (
            self.client.collection("conversations")
            .where("user_id", "==", user_id)
            .order_by("updated_at", direction="DESCENDING")
        )
        return [doc.to_dict() for doc in query.stream()]

    async def get_conversation(self, conversation_id: str) -> dict | None:
        """Devuelve una conversación por ID."""
        doc = self.client.collection("conversations").document(conversation_id).get()
        return doc.to_dict() if doc.exists else None

    # ---------- graph_artifacts ----------
    async def create_graph_artifact(self, **fields) -> str:
        """Crea un artefacto de gráfico y devuelve su ID."""
        artifact_id = new_id()
        ref = self.client.collection("graph_artifacts").document(artifact_id)
        payload = {
            "artifact_id": artifact_id,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        ref.set(payload)
        return artifact_id

    async def get_graph_artifact(self, artifact_id: str) -> dict | None:
        """Devuelve un artefacto por ID."""
        doc = self.client.collection("graph_artifacts").document(artifact_id).get()
        return doc.to_dict() if doc.exists else None


def get_firestore_service() -> FirestoreService:
    """Dependencia FastAPI: FirestoreService (usa app de Firebase)."""
    from firebase_admin import firestore as firestore_admin

    from app.services.firebase.client import init_firebase

    client: FirestoreClient = firestore_admin.client(app=init_firebase())
    return FirestoreService(client)
