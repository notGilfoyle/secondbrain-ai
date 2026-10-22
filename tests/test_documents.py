"""Application and API tests for local document ingestion."""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from secondbrain.api.dependencies import get_document_service
from secondbrain.application.documents import (
    DocumentChunk,
    DocumentService,
    DocumentTooLarge,
    InvalidDocument,
    UnsupportedDocumentType,
)
from secondbrain.chunking import chunk_text
from secondbrain.config.settings import Settings
from secondbrain.infrastructure.document_parser import DocumentTextExtractor
from secondbrain.infrastructure.local_file_storage import LocalFileStorage
from secondbrain.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from secondbrain.main import app


def make_pdf(text: str) -> bytes:
    """Build a small valid single-page PDF fixture with a text content stream."""
    stream = f"BT /F1 14 Tf 40 90 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, pdf_object in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(pdf_object)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


@pytest.fixture
def service(tmp_path: Path) -> DocumentService:
    """Create a service with isolated filesystem and SQLite persistence."""
    return DocumentService(
        extractor=DocumentTextExtractor(),
        storage=LocalFileStorage(tmp_path / "files"),
        repository=SQLiteDocumentRepository(tmp_path / "database" / "documents.db"),
        max_upload_size_bytes=1024,
    )


@pytest.fixture
def client(service: DocumentService) -> Iterator[TestClient]:
    """Provide an API client whose document dependencies use temporary storage."""
    app.dependency_overrides[get_document_service] = lambda: service
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_document_service, None)


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "expected_type"),
    [
        ("notes.txt", "text/plain", b"plain text body", "text/plain"),
        ("notes.md", "text/markdown", b"# Heading\nMarkdown body", "text/markdown"),
    ],
)
def test_ingest_text_formats_persists_text_and_original(
    service: DocumentService,
    tmp_path: Path,
    filename: str,
    content_type: str,
    content: bytes,
    expected_type: str,
) -> None:
    document = service.ingest(filename, content_type, content)

    assert document.content_type == expected_type
    assert document.size_bytes == len(content)
    assert document.extracted_text_length == len(content.decode("utf-8"))
    assert (tmp_path / "files" / document.storage_path).read_bytes() == content
    with closing(sqlite3.connect(tmp_path / "database" / "documents.db")) as connection:
        with closing(
            connection.execute(
                "SELECT extracted_text FROM documents WHERE id = ?", (document.id,)
            )
        ) as cursor:
            stored_text = cursor.fetchone()[0]
    assert stored_text == content.decode("utf-8")


def test_pdf_ingestion_uses_real_parser(
    service: DocumentService, tmp_path: Path
) -> None:
    pdf_bytes = make_pdf("PDF text")
    document = service.ingest("sample.pdf", "application/pdf", pdf_bytes)

    assert document.content_type == "application/pdf"
    assert document.extracted_text_length == len("PDF text")
    assert service.list_documents()[0].id == document.id
    with closing(sqlite3.connect(tmp_path / "database" / "documents.db")) as connection:
        with closing(
            connection.execute(
                "SELECT extracted_text FROM documents WHERE id = ?", (document.id,)
            )
        ) as cursor:
            extracted_text = cursor.fetchone()[0]
    assert extracted_text.strip() == "PDF text"


def test_persistence_survives_new_service_instance(tmp_path: Path) -> None:
    first_service = DocumentService(
        DocumentTextExtractor(),
        LocalFileStorage(tmp_path / "files"),
        SQLiteDocumentRepository(tmp_path / "documents.db"),
        1024,
    )
    saved = first_service.ingest("saved.txt", "text/plain", b"still here")
    restarted_service = DocumentService(
        DocumentTextExtractor(),
        LocalFileStorage(tmp_path / "files"),
        SQLiteDocumentRepository(tmp_path / "documents.db"),
        1024,
    )

    assert restarted_service.list_documents() == [saved]
    retrieved = restarted_service.get_document(saved.id)
    assert retrieved is not None
    assert retrieved.metadata == saved
    assert retrieved.extracted_text == "still here"
    app.dependency_overrides[get_document_service] = lambda: restarted_service
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/v1/documents/{saved.id}")
    finally:
        app.dependency_overrides.pop(get_document_service, None)
    assert response.status_code == 200
    assert response.json()["extracted_text"] == "still here"
    assert (tmp_path / "files" / saved.storage_path).read_bytes() == b"still here"


def test_list_empty_database_returns_empty_result(service: DocumentService) -> None:
    assert service.list_documents() == []


def test_unsupported_extension_is_rejected(service: DocumentService) -> None:
    with pytest.raises(UnsupportedDocumentType):
        service.ingest("payload.csv", "text/csv", b"a,b")


def test_mismatched_content_type_is_rejected(service: DocumentService) -> None:
    with pytest.raises(UnsupportedDocumentType):
        service.ingest("payload.pdf", "text/plain", b"not a pdf")


def test_malformed_pdf_is_rejected(service: DocumentService) -> None:
    with pytest.raises(InvalidDocument, match="PDF"):
        service.ingest("broken.pdf", "application/pdf", b"not a PDF")


def test_empty_and_unextractable_documents_are_rejected(
    service: DocumentService,
) -> None:
    with pytest.raises(InvalidDocument, match="empty"):
        service.ingest("empty.txt", "text/plain", b"")
    with pytest.raises(InvalidDocument, match="no extractable text"):
        service.ingest("whitespace.txt", "text/plain", b"   \n")
    with pytest.raises(InvalidDocument, match="no extractable text"):
        service.ingest("textless.pdf", "application/pdf", make_pdf(""))


def test_invalid_utf8_is_rejected(service: DocumentService) -> None:
    with pytest.raises(InvalidDocument, match="UTF-8"):
        service.ingest("invalid.txt", "text/plain", b"\xff")


def test_maximum_upload_size_is_enforced(service: DocumentService) -> None:
    with pytest.raises(DocumentTooLarge):
        service.ingest("large.txt", "text/plain", b"x" * 1025)


def test_untrusted_and_duplicate_filenames_use_unique_safe_paths(
    service: DocumentService, tmp_path: Path
) -> None:
    first = service.ingest("../../outside.txt", "text/plain", b"first")
    second = service.ingest("../../outside.txt", "text/plain", b"second")
    storage_directory = (tmp_path / "files").resolve()

    assert first.storage_path != second.storage_path
    assert Path(first.storage_path).name == first.storage_path
    assert (
        (storage_directory / first.storage_path)
        .resolve()
        .is_relative_to(storage_directory)
    )
    assert (storage_directory / second.storage_path).read_bytes() == b"second"
    assert not (tmp_path.parent / "outside.txt").exists()


def test_database_failure_removes_written_original(tmp_path: Path) -> None:
    class FailingRepository:
        def add(
            self,
            document: object,
            extracted_text: str,
            chunks: list[DocumentChunk],
        ) -> None:
            raise sqlite3.OperationalError("simulated database failure")

        def list(self) -> list[object]:
            return []

    storage = LocalFileStorage(tmp_path / "files")
    failing_service = DocumentService(
        DocumentTextExtractor(), storage, FailingRepository(), 1024
    )

    with pytest.raises(sqlite3.OperationalError):
        failing_service.ingest("failed.txt", "text/plain", b"content")

    assert list((tmp_path / "files").iterdir()) == []


def test_sqlite_insert_failure_rolls_back_and_removes_original(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    storage = LocalFileStorage(tmp_path / "files")
    repository = SQLiteDocumentRepository(database_path)
    assert repository.list() == []

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            with closing(
                connection.execute(
                    """
                    CREATE TRIGGER reject_document_insert
                    BEFORE INSERT ON documents
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated insert failure');
                    END;
                    """
                )
            ):
                pass

    service = DocumentService(DocumentTextExtractor(), storage, repository, 1024)
    with pytest.raises(sqlite3.IntegrityError, match="simulated insert failure"):
        service.ingest("failed.txt", "text/plain", b"content")

    assert list((tmp_path / "files").iterdir()) == []
    with closing(sqlite3.connect(database_path)) as connection:
        with closing(connection.execute("SELECT COUNT(*) FROM documents")) as cursor:
            assert cursor.fetchone()[0] == 0
        with connection:
            with closing(connection.execute("DROP TRIGGER reject_document_insert")):
                pass

    recovered = service.ingest("recovered.txt", "text/plain", b"database usable")
    assert [document.id for document in repository.list()] == [recovered.id]
    assert (
        tmp_path / "files" / recovered.storage_path
    ).read_bytes() == b"database usable"


def test_ingestion_persists_ordered_chunks_and_survives_new_repository(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    service = DocumentService(
        DocumentTextExtractor(),
        LocalFileStorage(tmp_path / "files"),
        SQLiteDocumentRepository(database_path),
        4096,
    )
    text = "0123456789" * 120 + "tail"
    document = service.ingest("long.txt", "text/plain", text.encode())
    expected = chunk_text(text)

    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT id, document_id, chunk_index, text
            FROM document_chunks WHERE document_id = ? ORDER BY chunk_index
            """,
            (document.id,),
        ).fetchall()

    assert [row[2] for row in rows] == list(range(len(expected)))
    assert [row[3] for row in rows] == expected
    assert all(row[0] and row[1] == document.id for row in rows)

    restarted = SQLiteDocumentRepository(database_path)
    assert restarted.list()[0].id == document.id
    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE document_id = ?",
            (document.id,),
        ).fetchone()[0] == len(expected)


def test_multiple_documents_have_independent_chunk_indexes(
    tmp_path: Path,
) -> None:
    service = DocumentService(
        DocumentTextExtractor(),
        LocalFileStorage(tmp_path / "files"),
        SQLiteDocumentRepository(tmp_path / "database" / "documents.db"),
        2048,
    )
    first = service.ingest("first.txt", "text/plain", b"a" * 1100)
    second = service.ingest("second.txt", "text/plain", b"b" * 10)

    with closing(sqlite3.connect(tmp_path / "database" / "documents.db")) as connection:
        rows = connection.execute(
            "SELECT document_id, chunk_index FROM document_chunks "
            "ORDER BY document_id, chunk_index"
        ).fetchall()
    assert [row for row in rows if row[0] == first.id] == [(first.id, 0), (first.id, 1)]
    assert [row for row in rows if row[0] == second.id] == [(second.id, 0)]


def test_blank_extracted_text_is_rejected_without_persisting_chunks(
    service: DocumentService, tmp_path: Path
) -> None:
    with pytest.raises(InvalidDocument, match="no extractable text"):
        service.ingest("blank.txt", "text/plain", b"   ")
    assert service.list_documents() == []
    database_path = tmp_path / "database" / "documents.db"
    if database_path.exists():
        with closing(sqlite3.connect(database_path)) as connection:
            assert (
                connection.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
                == 0
            )


def test_chunk_insert_failure_rolls_back_document_and_cleans_original(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    storage = LocalFileStorage(tmp_path / "files")
    repository = SQLiteDocumentRepository(database_path)
    repository.list()  # Initialize schema before installing the failure trigger.
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TRIGGER reject_chunk_insert
                BEFORE INSERT ON document_chunks
                BEGIN
                    SELECT RAISE(ABORT, 'simulated chunk failure');
                END
                """
            )

    service = DocumentService(DocumentTextExtractor(), storage, repository, 4096)
    with pytest.raises(sqlite3.IntegrityError, match="simulated chunk failure"):
        service.ingest("failed.txt", "text/plain", b"x" * 1200)

    assert list((tmp_path / "files").iterdir()) == []
    with closing(sqlite3.connect(database_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
            == 0
        )


def test_api_upload_list_and_health_contract(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("hello.txt", b"hello api", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "hello.txt"
    assert body["extracted_text_length"] == len("hello api")
    assert "storage_path" not in body
    assert "extracted_text" not in body

    listing = client.get("/api/v1/documents")
    assert listing.status_code == 200
    assert listing.json() == {"documents": [body]}
    retrieved = client.get(f"/api/v1/documents/{body['id']}")
    assert retrieved.status_code == 200
    assert retrieved.json() == {
        **body,
        "extracted_text": "hello api",
    }
    assert "storage_path" not in retrieved.json()
    assert client.get("/health").json() == {"status": "ok"}


def test_api_document_retrieval_not_found_and_invalid_id(client: TestClient) -> None:
    missing = client.get(f"/api/v1/documents/{'0' * 32}")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Document not found"}

    invalid = client.get("/api/v1/documents/not-a-uuid")
    assert invalid.status_code == 422
    assert "detail" in invalid.json()


def test_api_document_retrieval_hides_database_errors(
    client: TestClient,
    service: DocumentService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    service.ingest("error.txt", "text/plain", b"stored text")
    sentinel = "PRIVATE_DATABASE_EXCEPTION_SENTINEL"

    def fail_lookup(document_id: str) -> None:
        raise sqlite3.OperationalError(sentinel)

    monkeypatch.setattr(
        SQLiteDocumentRepository,
        "get",
        lambda _repository, document_id: fail_lookup(document_id),
    )
    response = client.get(f"/api/v1/documents/{'a' * 32}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Document retrieval failed"}
    assert sentinel not in response.text
    assert sentinel not in caplog.text
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "secondbrain.api.routes.documents"
    ]
    assert events == [
        {
            "event": "document.get.failed",
            "operation": "get",
            "failure_category": "unexpected",
        }
    ]


def test_api_document_listing_hides_unexpected_exception(
    client: TestClient,
    service: DocumentService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    sentinel = "PRIVATE_LIST_EXCEPTION_SENTINEL"

    def fail_listing() -> None:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(service, "list_documents", fail_listing)
    response = client.get("/api/v1/documents")

    assert response.status_code == 500
    assert response.json() == {"detail": "Document listing failed"}
    assert sentinel not in response.text
    assert sentinel not in caplog.text
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "secondbrain.api.routes.documents"
    ]
    assert events == [
        {
            "event": "document.list.failed",
            "operation": "list",
            "failure_category": "unexpected",
        }
    ]


def test_api_document_ingestion_hides_unexpected_exception(
    client: TestClient,
    service: DocumentService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    sentinel = "PRIVATE_INGESTION_EXCEPTION_SENTINEL"

    def fail_ingestion(*_args: object) -> None:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(service, "ingest", fail_ingestion)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("private.txt", b"PRIVATE_DOCUMENT_TEXT", "text/plain")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Document ingestion failed"}
    assert sentinel not in response.text
    assert "PRIVATE_DOCUMENT_TEXT" not in caplog.text
    assert sentinel not in caplog.text
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "secondbrain.api.routes.documents"
    ]
    assert events == [
        {
            "event": "document.ingestion.failed",
            "operation": "ingest",
            "failure_category": "unexpected",
        }
    ]


def test_api_empty_list_and_bounded_upload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.get("/api/v1/documents").json() == {"documents": []}
    monkeypatch.setattr(
        "secondbrain.api.routes.documents.get_settings",
        lambda: Settings(max_upload_size_bytes=4),
    )
    response = client.post(
        "/api/v1/documents",
        files={"file": ("too-large.txt", b"12345", "text/plain")},
    )
    assert response.status_code == 413


def test_api_rejects_invalid_pdf(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", b"not a PDF", "application/pdf")},
    )
    assert response.status_code == 422
