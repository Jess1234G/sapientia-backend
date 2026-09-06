"""Pruebas unitarias de AttachmentService (R2 + Firestore fakes)."""
from __future__ import annotations

import pytest

from app.config import settings
from app.services.storage.attachment_service import (
    AttachmentService,
    AttachmentServiceError,
)


class FakeR2:
    def __init__(self):
        self.uploaded = []
        self.deleted = []
        self.fail_upload = False
        self.fail_delete = False

    def upload_bytes_with_key(self, storage_key, content, content_type):
        if self.fail_upload:
            raise RuntimeError("R2 upload failed")
        self.uploaded.append((storage_key, content, content_type))

    def generate_presigned_url(self, storage_key, expires_in=3600):
        return f"https://fake.example/{storage_key}"

    def delete_object(self, storage_key):
        if self.fail_delete:
            raise RuntimeError("R2 delete failed")
        self.deleted.append(storage_key)


class FakeFirestore:
    def __init__(self):
        self.attachments = {}
        self.fail_create = False

    async def create_attachment_metadata(self, **kwargs):
        if self.fail_create:
            raise RuntimeError("Firestore failed")
        data = dict(kwargs)
        data["created_at"] = "2026-01-01T00:00:00+00:00"
        self.attachments[kwargs["attachment_id"]] = data
        return data

    async def get_user_attachment(self, attachment_id, user_id):
        data = self.attachments.get(attachment_id)
        if data is None or data["user_id"] != user_id:
            return None
        return data


@pytest.fixture
def service():
    r2 = FakeR2()
    firestore = FakeFirestore()
    return AttachmentService(r2=r2, firestore=firestore), r2, firestore


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake" * 10
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"fake" * 10
WEBP_BYTES = b"RIFF" + b"fake" * 10
PDF_BYTES = b"%PDF-1.4" + b"fake" * 10
TXT_BYTES = b"hola mundo " * 10


# ============================================================
# TIPOS PERMITIDOS
# ============================================================

@pytest.mark.asyncio
async def test_create_png(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.png", "image/png", PNG_BYTES
    )
    assert meta["content_type"] == "image/png"
    assert meta["size"] == len(PNG_BYTES)
    assert len(r2.uploaded) == 1


@pytest.mark.asyncio
async def test_create_jpeg(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.jpeg", "image/jpeg", JPEG_BYTES
    )
    assert meta["content_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_create_jpg_variant(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.jpg", "image/jpeg", JPEG_BYTES
    )
    assert meta["content_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_create_webp(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.webp", "image/webp", WEBP_BYTES
    )
    assert meta["content_type"] == "image/webp"


@pytest.mark.asyncio
async def test_create_pdf(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "documento.pdf", "application/pdf", PDF_BYTES
    )
    assert meta["content_type"] == "application/pdf"


@pytest.mark.asyncio
async def test_create_txt(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "notas.txt", "text/plain", TXT_BYTES
    )
    assert meta["content_type"] == "text/plain"


# ============================================================
# RECHAZOS
# ============================================================

@pytest.mark.asyncio
async def test_disallowed_type(service):
    srv, r2, firestore = service
    with pytest.raises(AttachmentServiceError) as exc:
        await srv.create_attachment(
            "user-1",
            "doc.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            b"x" * 100,
        )
    assert exc.value.status_code == 400
    assert len(r2.uploaded) == 0


@pytest.mark.asyncio
async def test_empty_content(service):
    srv, r2, firestore = service
    with pytest.raises(AttachmentServiceError) as exc:
        await srv.create_attachment(
            "user-1", "foto.png", "image/png", b""
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_empty_filename(service):
    srv, r2, firestore = service
    with pytest.raises(AttachmentServiceError) as exc:
        await srv.create_attachment(
            "user-1", "   ", "image/png", PNG_BYTES
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_incompatible_extension(service):
    srv, r2, firestore = service
    with pytest.raises(AttachmentServiceError) as exc:
        await srv.create_attachment(
            "user-1", "foto.pdf", "image/png", PNG_BYTES
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_size_limit(service, monkeypatch):
    monkeypatch.setattr(settings, "max_attachment_size_mb", 1)
    srv, r2, firestore = service
    content = b"x" * (1024 * 1024 + 1)
    with pytest.raises(AttachmentServiceError) as exc:
        await srv.create_attachment(
            "user-1", "file.txt", "text/plain", content
        )
    assert exc.value.status_code == 413
    assert len(r2.uploaded) == 0


# ============================================================
# METADATA / STORAGE KEY / ID
# ============================================================

@pytest.mark.asyncio
async def test_metadata_correct(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.png", "image/png", PNG_BYTES
    )
    assert meta["user_id"] == "user-1"
    assert meta["filename"] == "foto.png"
    assert meta["content_type"] == "image/png"
    assert meta["size"] == len(PNG_BYTES)
    assert meta["storage_key"].startswith("attachments/user-1/")
    assert "created_at" in meta


@pytest.mark.asyncio
async def test_storage_key_correct(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.png", "image/png", PNG_BYTES
    )
    key = meta["storage_key"]
    parts = key.split("/")
    assert parts[0] == "attachments"
    assert parts[1] == "user-1"
    assert parts[2].endswith(".png")
    # El uuid no debe contener el filename del usuario.
    assert "foto" not in parts[2]


@pytest.mark.asyncio
async def test_attachment_id_generated(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.png", "image/png", PNG_BYTES
    )
    assert meta["attachment_id"]
    assert meta["attachment_id"] != meta["storage_key"]


# ============================================================
# OWNERSHIP
# ============================================================

@pytest.mark.asyncio
async def test_ownership(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.png", "image/png", PNG_BYTES
    )
    got = await srv.get_attachment(meta["attachment_id"], "user-1")
    assert got is not None
    assert got["attachment_id"] == meta["attachment_id"]


@pytest.mark.asyncio
async def test_other_user_cannot_retrieve(service):
    srv, r2, firestore = service
    meta = await srv.create_attachment(
        "user-1", "foto.png", "image/png", PNG_BYTES
    )
    got = await srv.get_attachment(meta["attachment_id"], "user-2")
    assert got is None


# ============================================================
# ROLLBACK
# ============================================================

@pytest.mark.asyncio
async def test_rollback_on_firestore_failure(service):
    srv, r2, firestore = service
    firestore.fail_create = True
    with pytest.raises(RuntimeError):
        await srv.create_attachment(
            "user-1", "foto.png", "image/png", PNG_BYTES
        )
    assert len(r2.deleted) == 1
    assert r2.deleted[0] == r2.uploaded[0][0]


@pytest.mark.asyncio
async def test_rollback_failure_preserves_original_error(service):
    srv, r2, firestore = service
    firestore.fail_create = True
    r2.fail_delete = True
    with pytest.raises(RuntimeError) as exc:
        await srv.create_attachment(
            "user-1", "foto.png", "image/png", PNG_BYTES
        )
    assert "Firestore failed" in str(exc.value)
