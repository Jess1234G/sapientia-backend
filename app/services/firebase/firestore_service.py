# backend/app/services/firebase/firestore_service.py

from __future__ import annotations

import logging
from datetime import datetime, timezone

from firebase_admin import firestore as firestore_admin
from google.cloud.firestore import Client as FirestoreClient

from app.utils.ids import new_id

logger = logging.getLogger(__name__)


class FirestoreService:
    """Acceso a Firestore para usuarios, conversaciones y gráficos."""

    def __init__(self, client: FirestoreClient) -> None:
        self.client = client

    # =========================================================
    # USERS
    # =========================================================

    async def upsert_user(
        self,
        uid: str,
        email: str,
        name: str,
        picture: str,
    ) -> dict:
        """Crea o actualiza users/{uid} y devuelve el documento."""

        now = datetime.now(timezone.utc).isoformat()

        doc_ref = self.client.collection("users").document(uid)

        doc = doc_ref.get()
        data = doc.to_dict() if doc.exists else {}

        updated = {
            **data,
            "uid": uid,
            "email": email or data.get("email", ""),
            "display_name": name or data.get("display_name", ""),
            "photo_url": picture or data.get("photo_url", ""),
            "updated_at": now,
        }

        updated.setdefault("created_at", now)
        updated.setdefault("carrera", data.get("carrera", ""))
        updated.setdefault("semestre", data.get("semestre", 0))

        doc_ref.set(updated, merge=True)

        return updated

    async def get_user(
        self,
        uid: str,
    ) -> dict | None:
        """Devuelve users/{uid} o None."""

        doc = (
            self.client
            .collection("users")
            .document(uid)
            .get()
        )

        return doc.to_dict() if doc.exists else None

    # =========================================================
    # PERSISTENT MEMORY
    # =========================================================

    async def get_user_memory(
        self,
        uid: str,
    ) -> dict:
        """
        Devuelve la memoria persistente del usuario.

        La estructura se mantiene deliberadamente pequeña y
        controlada para evitar que la memoria crezca sin límite.
        """

        user = await self.get_user(uid)

        if not user:
            return {
                "preferences": [],
                "facts": [],
                "goals": [],
                "projects": [],
            }

        memory = user.get("memory", {})

        return {
            "preferences": list(
                memory.get("preferences", [])
            ),
            "facts": list(
                memory.get("facts", [])
            ),
            "goals": list(
                memory.get("goals", [])
            ),
            "projects": list(
                memory.get("projects", [])
            ),
        }

    async def add_memory_items(
        self,
        uid: str,
        category: str,
        items: list[str],
    ) -> dict:
        """
        Añade elementos a una categoría de memoria persistente.

        Categorías permitidas:
            preferences
            facts
            goals
            projects
        """

        allowed_categories = {
            "preferences",
            "facts",
            "goals",
            "projects",
        }

        if category not in allowed_categories:
            raise ValueError(
                f"Categoría de memoria inválida: {category}"
            )

        cleaned_items: list[str] = []

        for item in items:
            value = str(item).strip()

            if not value:
                continue

            # Evitamos recuerdos excesivamente grandes.
            value = value[:500]

            if value not in cleaned_items:
                cleaned_items.append(value)

        if not cleaned_items:
            return await self.get_user_memory(uid)

        user_ref = (
            self.client
            .collection("users")
            .document(uid)
        )

        # ArrayUnion evita duplicados y evita reemplazar
        # la memoria existente.
        user_ref.update(
            {
                f"memory.{category}": (
                    firestore_admin.ArrayUnion(
                        cleaned_items
                    )
                ),
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

        return await self.get_user_memory(uid)

    # =========================================================
    # CONVERSATIONS
    # =========================================================

    async def create_conversation(
        self,
        user_id: str,
        title: str,
    ) -> str:
        """
        Crea una conversación perteneciente al usuario.

        Devuelve el conversation_id generado por Firestore.
        """

        now = datetime.now(timezone.utc).isoformat()

        ref = (
            self.client
            .collection("conversations")
            .document()
        )

        payload = {
            "conversation_id": ref.id,
            "user_id": user_id,
            "title": title or "Nueva conversación",
            "messages": [],
            "attachments": [],
            "status": "active",
            "is_pinned": False,
            "created_at": now,
            "updated_at": now,
        }

        ref.set(payload)

        return ref.id

    async def add_message(
        self,
        conversation_id: str,
        message: dict,
    ) -> None:
        """
        Añade un mensaje al array de mensajes de la conversación.

        Importante:
        ArrayUnion evita reemplazar el historial existente.
        """

        ref = (
            self.client
            .collection("conversations")
            .document(conversation_id)
        )

        snapshot = ref.get()

        if not snapshot.exists:
            raise ValueError(
                f"La conversación '{conversation_id}' no existe."
            )

        ref.update(
            {
                "messages": firestore_admin.ArrayUnion(
                    [message]
                ),
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

    async def list_conversations(
        self,
        user_id: str,
    ) -> list[dict]:
        """
        Devuelve las conversaciones pertenecientes exclusivamente
        al usuario autenticado, ordenadas por updated_at descendente.
        """

        query = (
            self.client
            .collection("conversations")
            .where(
                "user_id",
                "==",
                user_id,
            )
            .order_by(
                "updated_at",
                direction="DESCENDING",
            )
        )

        return [
            document.to_dict()
            for document in query.stream()
        ]

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> dict | None:
        """Devuelve una conversación por ID o None."""

        document = (
            self.client
            .collection("conversations")
            .document(conversation_id)
            .get()
        )

        if not document.exists:
            return None

        return document.to_dict()

    async def get_user_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> dict | None:
        """
        Devuelve una conversación únicamente si pertenece
        al usuario solicitado.
        """

        conversation = await self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return None

        if conversation.get("user_id") != user_id:
            return None

        return conversation

    async def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: str | None = None,
        is_pinned: bool | None = None,
    ) -> dict | None:
        """
        Actualiza parcialmente una conversación (renombrar/fijar).

        Devuelve None si la conversación no existe o no pertenece
        al usuario indicado.
        """

        conversation = await self.get_user_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            return None

        fields: dict = {}

        if title is not None:
            cleaned_title = title.strip()

            if not cleaned_title:
                raise ValueError(
                    "El título no puede estar vacío."
                )

            fields["title"] = cleaned_title[:80]

        if is_pinned is not None:
            fields["is_pinned"] = bool(is_pinned)

        if not fields:
            return conversation

        fields["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        ref = (
            self.client
            .collection("conversations")
            .document(conversation_id)
        )

        ref.update(fields)

        return await self.get_conversation(conversation_id)

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> bool:
        """
        Elimina una conversación únicamente si pertenece al usuario.

        Devuelve True si se eliminó y False si no existe o no
        pertenece al usuario.
        """

        conversation = await self.get_user_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if conversation is None:
            return False

        ref = (
            self.client
            .collection("conversations")
            .document(conversation_id)
        )

        ref.delete()

        return True

    # =========================================================
    # GRAPH ARTIFACTS
    # =========================================================

    async def create_graph_artifact(
        self,
        **fields,
    ) -> str:
        """Crea un artefacto de gráfico y devuelve su ID."""

        artifact_id = new_id()

        ref = (
            self.client
            .collection("graph_artifacts")
            .document(artifact_id)
        )

        payload = {
            "artifact_id": artifact_id,
            "status": "pending",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            **fields,
        }

        ref.set(payload)

        return artifact_id

    async def get_graph_artifact(
        self,
        artifact_id: str,
    ) -> dict | None:
        """Devuelve un artefacto de gráfico por ID."""

        doc = (
            self.client
            .collection("graph_artifacts")
            .document(artifact_id)
            .get()
        )

        return doc.to_dict() if doc.exists else None

    async def update_graph_artifact(
        self,
        artifact_id: str,
        **fields,
    ) -> None:
        """
        Actualiza los campos de un artefacto de gráfico.

        El documento debe existir. Si no existe, se genera un
        ValueError para evitar crear accidentalmente un artefacto
        incompleto durante una actualización.
        """

        ref = (
            self.client
            .collection("graph_artifacts")
            .document(artifact_id)
        )

        snapshot = ref.get()

        if not snapshot.exists:
            raise ValueError(
                f"El artefacto '{artifact_id}' no existe."
            )

        if not fields:
            return

        ref.update(fields)

    # =========================================================
    # ATTACHMENTS
    # =========================================================

    async def create_attachment_metadata(
        self,
        *,
        attachment_id: str,
        user_id: str,
        filename: str,
        content_type: str,
        size: int,
        storage_key: str,
    ) -> dict:
        """Guarda la metadata de un adjunto en la colección attachments."""

        now = datetime.now(timezone.utc).isoformat()

        payload = {
            "attachment_id": attachment_id,
            "user_id": user_id,
            "filename": filename,
            "content_type": content_type,
            "size": size,
            "storage_key": storage_key,
            "created_at": now,
        }

        ref = (
            self.client
            .collection("attachments")
            .document(attachment_id)
        )

        ref.set(payload)

        return payload

    async def get_user_attachment(
        self,
        attachment_id: str,
        user_id: str,
    ) -> dict | None:
        """Devuelve un adjunto únicamente si pertenece al usuario."""

        document = (
            self.client
            .collection("attachments")
            .document(attachment_id)
            .get()
        )

        if not document.exists:
            return None

        data = document.to_dict()

        if data.get("user_id") != user_id:
            return None

        return data


# =============================================================
# DEPENDENCY
# =============================================================

def get_firestore_service() -> FirestoreService:
    """
    Dependencia FastAPI para obtener FirestoreService
    usando la aplicación Firebase Admin configurada.
    """

    from app.services.firebase.client import init_firebase

    client: FirestoreClient = firestore_admin.client(
        app=init_firebase()
    )

    return FirestoreService(client)