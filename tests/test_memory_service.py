"""
Pruebas del MemoryService.
"""

from __future__ import annotations

import pytest

from app.services.memory.memory_service import (
    MemoryService,
)


class FakeMemoryFirestore:
    """Firestore falso para pruebas de memoria."""

    def __init__(self):
        self.get_user_calls = 0
        self.list_conversations_calls = 0

        self.users = {
            "user-123": {
                "uid": "user-123",
                "email": "user@example.com",
                "display_name": "Usuario de Prueba",
                "photo_url": "",
                "carrera": "Ingeniería",
                "semestre": 4,
            }
        }

        self.conversations = [
            {
                "conversation_id": "chat-1",
                "user_id": "user-123",
                "title": "Matemáticas",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explícame derivadas.",
                    },
                    {
                        "role": "assistant",
                        "content": "Una derivada representa una tasa de cambio.",
                    },
                ],
            },
            {
                "conversation_id": "chat-2",
                "user_id": "user-123",
                "title": "Programación",
                "messages": [
                    {
                        "role": "user",
                        "content": "Necesito ayuda con Python.",
                    },
                ],
            },
            {
                "conversation_id": "chat-3",
                "user_id": "user-123",
                "title": "Física",
                "messages": [
                    {
                        "role": "user",
                        "content": "Ayúdame con cinemática.",
                    },
                ],
            },
        ]

    async def get_user(self, user_id: str):
        self.get_user_calls += 1
        return self.users.get(user_id)

    async def get_user_memory(self, user_id: str):
        user = self.users.get(user_id)

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

    async def list_conversations(
        self,
        user_id: str,
    ):
        self.list_conversations_calls += 1

        return [
            conversation
            for conversation in self.conversations
            if conversation["user_id"] == user_id
        ]


@pytest.mark.asyncio
async def test_memory_includes_user_profile():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
    )

    assert "PERFIL DEL USUARIO" in context
    assert "Nombre: Usuario de Prueba" in context
    assert "Carrera: Ingeniería" in context
    assert "Semestre: 4" in context


@pytest.mark.asyncio
async def test_memory_includes_recent_conversations():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
    )

    assert "Matemáticas" in context
    assert "Programación" in context
    assert "Explícame derivadas." in context


@pytest.mark.asyncio
async def test_memory_excludes_current_conversation():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
        current_conversation_id="chat-1",
    )

    assert "Matemáticas" not in context
    assert "Explícame derivadas." not in context

    assert "Programación" in context


@pytest.mark.asyncio
async def test_memory_limits_number_of_conversations():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
        max_conversations=1,
    )

    assert "Matemáticas" in context

    assert "Programación" not in context
    assert "Física" not in context


@pytest.mark.asyncio
async def test_memory_truncates_long_messages():
    firestore = FakeMemoryFirestore()

    long_text = "A" * 3000

    firestore.conversations[0]["messages"] = [
        {
            "role": "user",
            "content": long_text,
        }
    ]

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
        max_content_length=100,
    )

    assert len(context) < 2000
    assert "..." in context


@pytest.mark.asyncio
async def test_memory_empty_when_no_data():
    firestore = FakeMemoryFirestore()

    firestore.users = {}
    firestore.conversations = []

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
    )

    assert context == ""


@pytest.mark.asyncio
async def test_persistent_memory_has_default_structure():
    firestore = FakeMemoryFirestore()

    firestore.users["user-123"]["memory"] = {
        "preferences": [],
        "facts": [],
        "goals": [],
        "projects": [],
    }

    service = MemoryService(
        firestore=firestore,
    )

    memory = await service.get_persistent_memory(
        "user-123"
    )

    assert set(memory.keys()) == {
        "preferences",
        "facts",
        "goals",
        "projects",
    }


@pytest.mark.asyncio
async def test_save_memory_adds_items():
    firestore = FakeMemoryFirestore()

    firestore.users["user-123"]["memory"] = {
        "preferences": [],
        "facts": [],
        "goals": [],
        "projects": [],
    }

    async def add_memory_items(
        uid,
        category,
        items,
    ):
        memory = firestore.users[uid]["memory"]

        for item in items:
            if item not in memory[category]:
                memory[category].append(item)

        return memory

    firestore.add_memory_items = add_memory_items

    service = MemoryService(
        firestore=firestore,
    )

    memory = await service.save_memory(
        user_id="user-123",
        category="preferences",
        items=[
            "Explicaciones paso a paso",
        ],
    )

    assert (
        "Explicaciones paso a paso"
        in memory["preferences"]
    )


@pytest.mark.asyncio
async def test_save_memory_rejects_invalid_category():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    with pytest.raises(ValueError):
        await service.save_memory(
            user_id="user-123",
            category="random",
            items=["dato"],
        )


@pytest.mark.asyncio
async def test_build_context_includes_persistent_memory():
    firestore = FakeMemoryFirestore()

    firestore.users["user-123"]["memory"] = {
        "preferences": [
            "Prefiere explicaciones paso a paso."
        ],
        "facts": [
            "Está estudiando ingeniería."
        ],
        "goals": [
            "Construir Sapientia."
        ],
        "projects": [
            "Desarrolla una IA educativa."
        ],
    }

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
    )

    assert "MEMORIA PERSISTENTE" in context
    assert (
        "Prefiere explicaciones paso a paso."
        in context
    )
    assert (
        "Construir Sapientia."
        in context
    )


@pytest.mark.asyncio
async def test_build_context_keeps_persistent_memory_when_excluding_chat():
    firestore = FakeMemoryFirestore()

    firestore.users["user-123"]["memory"] = {
        "preferences": [
            "Prefiere lenguaje humano."
        ],
        "facts": [],
        "goals": [],
        "projects": [],
    }

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="user-123",
        current_conversation_id="chat-1",
    )

    assert (
        "Prefiere lenguaje humano."
        in context
    )

    assert "Matemáticas" not in context


@pytest.mark.asyncio
async def test_unknown_user_has_empty_persistent_memory():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    memory = await service.get_persistent_memory(
        "does-not-exist"
    )

    assert memory == {
        "preferences": [],
        "facts": [],
        "goals": [],
        "projects": [],
    }


@pytest.mark.asyncio
async def test_build_context_reads_user_only_once():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    await service.build_context(
        user_id="user-123",
    )

    assert firestore.get_user_calls == 1
    assert firestore.list_conversations_calls == 1


@pytest.mark.asyncio
async def test_unknown_user_skips_conversation_query():
    firestore = FakeMemoryFirestore()

    service = MemoryService(
        firestore=firestore,
    )

    context = await service.build_context(
        user_id="does-not-exist",
    )

    assert context == ""
    assert firestore.get_user_calls == 1
    assert firestore.list_conversations_calls == 0
