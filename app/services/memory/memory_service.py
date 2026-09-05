"""
memory_service.py — Memoria contextual y persistente de Sapientia.

Memoria contextual:
    Perfil + conversaciones recientes.

Memoria persistente:
    Preferencias, hechos, objetivos y proyectos.
"""

from __future__ import annotations

import logging

from app.services.firebase.firestore_service import (
    FirestoreService,
)


logger = logging.getLogger(__name__)


PERSISTENT_MEMORY_CATEGORIES = {
    "preferences",
    "facts",
    "goals",
    "projects",
}


class MemoryService:
    """
    Construye memoria contextual de bajo costo y administra
    memoria persistente explícita.
    """

    def __init__(
        self,
        firestore: FirestoreService,
    ) -> None:
        self.firestore = firestore

    async def get_persistent_memory(
        self,
        user_id: str,
    ) -> dict:
        """Obtiene la memoria persistente del usuario."""

        return await self.firestore.get_user_memory(
            user_id
        )

    async def save_memory(
        self,
        user_id: str,
        category: str,
        items: list[str],
    ) -> dict:
        """
        Guarda elementos de memoria persistente.

        La memoria no se genera automáticamente todavía.
        Esta función será utilizada posteriormente por la capa
        que determine qué información merece ser recordada.
        """

        if category not in PERSISTENT_MEMORY_CATEGORIES:
            raise ValueError(
                f"Categoría de memoria inválida: {category}"
            )

        return await self.firestore.add_memory_items(
            uid=user_id,
            category=category,
            items=items,
        )

    @staticmethod
    def _normalize_persistent_memory(
        user: dict,
    ) -> dict:
        """
        Extrae y normaliza la memoria persistente desde
        el documento del usuario ya cargado.

        No realiza ninguna llamada adicional a Firestore.
        """

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

    async def build_context(
        self,
        user_id: str,
        current_conversation_id: str = "",
        max_conversations: int = 3,
        max_messages_per_conversation: int = 4,
        max_content_length: int = 1200,
    ) -> str:
        """
        Construye un contexto compacto para ReasoningService.

        Orden:
            1. Memoria persistente.
            2. Perfil.
            3. Conversaciones recientes.

        La conversación actual se excluye del historial contextual.

        Optimización:
            - Una sola lectura del documento del usuario.
            - No se consulta el historial si el usuario no existe.
        """

        # ====================================================
        # PERFIL + MEMORIA PERSISTENTE
        # ====================================================

        user = await self.firestore.get_user(
            user_id
        )

        # Si el usuario no existe, no tiene sentido realizar
        # una segunda consulta para recuperar conversaciones.
        if not user:
            return ""

        sections: list[str] = []

        persistent_memory = (
            self._normalize_persistent_memory(user)
        )

        # ====================================================
        # MEMORIA PERSISTENTE
        # ====================================================

        memory_lines: list[str] = []

        labels = {
            "preferences": "Preferencias",
            "facts": "Datos importantes",
            "goals": "Objetivos",
            "projects": "Proyectos",
        }

        for category, label in labels.items():
            items = persistent_memory.get(
                category,
                [],
            )

            if not items:
                continue

            memory_lines.append(
                f"{label}:"
            )

            for item in items:
                memory_lines.append(
                    f"- {item}"
                )

        if memory_lines:
            sections.append(
                "MEMORIA PERSISTENTE\n"
                + "\n".join(memory_lines)
            )

        # ====================================================
        # PERFIL
        # ====================================================

        profile_lines: list[str] = []

        display_name = (
            user.get("display_name")
            or ""
        ).strip()

        email = (
            user.get("email")
            or ""
        ).strip()

        career = (
            user.get("carrera")
            or ""
        ).strip()

        semester = user.get(
            "semestre",
            0,
        )

        if display_name:
            profile_lines.append(
                f"Nombre: {display_name}"
            )

        if email:
            profile_lines.append(
                f"Correo: {email}"
            )

        if career:
            profile_lines.append(
                f"Carrera: {career}"
            )

        if semester:
            profile_lines.append(
                f"Semestre: {semester}"
            )

        if profile_lines:
            sections.append(
                "PERFIL DEL USUARIO\n"
                + "\n".join(profile_lines)
            )

        # ====================================================
        # CONVERSACIONES RECIENTES
        # ====================================================

        conversations = (
            await self.firestore.list_conversations(
                user_id
            )
        )

        conversation_sections: list[str] = []

        selected_count = 0

        for conversation in conversations:
            conversation_id = (
                conversation.get(
                    "conversation_id",
                    "",
                )
            )

            if (
                current_conversation_id
                and conversation_id
                == current_conversation_id
            ):
                continue

            messages = conversation.get(
                "messages",
                [],
            )

            if not messages:
                continue

            title = (
                conversation.get("title")
                or "Nueva conversación"
            )

            recent_messages = messages[
                -max_messages_per_conversation:
            ]

            message_lines: list[str] = [
                f"Conversación: {title}"
            ]

            for message in recent_messages:
                role = (
                    message.get("role")
                    or "unknown"
                )

                content = (
                    message.get("content")
                    or ""
                ).strip()

                if not content:
                    continue

                if len(content) > max_content_length:
                    content = (
                        content[
                            :max_content_length
                        ]
                        + "..."
                    )

                message_lines.append(
                    f"{role}: {content}"
                )

            if len(message_lines) <= 1:
                continue

            conversation_sections.append(
                "\n".join(message_lines)
            )

            selected_count += 1

            if selected_count >= max_conversations:
                break

        if conversation_sections:
            sections.append(
                "CONTEXTO DE CONVERSACIONES RECIENTES\n"
                + "\n\n".join(
                    conversation_sections
                )
            )

        # ====================================================
        # RESULTADO
        # ====================================================

        if not sections:
            return ""

        return (
            "MEMORIA CONTEXTUAL DE SAPIENTIA\n\n"
            + "\n\n".join(sections)
        )