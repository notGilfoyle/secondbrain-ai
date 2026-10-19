"""Document-ingestion application behavior and its infrastructure contracts."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4


class DocumentError(Exception):
    """Base class for expected document-ingestion errors."""


class UnsupportedDocumentType(DocumentError):
    """Raised when the uploaded format is not supported."""


class InvalidDocument(DocumentError):
    """Raised when the uploaded document cannot be read or has no text."""


class DocumentTooLarge(DocumentError):
    """Raised when the uploaded file exceeds the configured limit."""


@dataclass(frozen=True)
class DocumentMetadata:
    """Persisted metadata returned to API clients."""

    id: str
    filename: str
    content_type: str
    storage_path: str
    size_bytes: int
    created_at: datetime
    extracted_text_length: int


@dataclass(frozen=True)
class RetrievedDocument:
    """Persisted document metadata and extracted text for retrieval."""

    metadata: DocumentMetadata
    extracted_text: str


class TextExtractor(Protocol):
    """Extract text from bytes in a supported document format."""

    def extract(self, extension: str, content: bytes) -> str: ...


class DocumentStorage(Protocol):
    """Store and remove original document bytes using safe references."""

    def save(self, document_id: str, extension: str, content: bytes) -> str: ...

    def delete(self, storage_path: str) -> None: ...


class DocumentRepository(Protocol):
    """Persist document metadata and extracted text."""

    def add(self, document: DocumentMetadata, extracted_text: str) -> None: ...

    def list(self) -> list[DocumentMetadata]: ...

    def get(self, document_id: str) -> RetrievedDocument | None: ...


class DocumentService:
    """Coordinate document validation, extraction, storage, and persistence."""

    _CONTENT_TYPES = {
        ".pdf": ("application/pdf", {"application/pdf", "application/octet-stream"}),
        ".md": (
            "text/markdown",
            {
                "text/markdown",
                "text/x-markdown",
                "text/plain",
                "application/octet-stream",
            },
        ),
        ".txt": (
            "text/plain",
            {"text/plain", "application/octet-stream"},
        ),
    }

    def __init__(
        self,
        extractor: TextExtractor,
        storage: DocumentStorage,
        repository: DocumentRepository,
        max_upload_size_bytes: int,
    ) -> None:
        self._extractor = extractor
        self._storage = storage
        self._repository = repository
        self._max_upload_size_bytes = max_upload_size_bytes

    def ingest(
        self, filename: str, content_type: str | None, content: bytes
    ) -> DocumentMetadata:
        """Validate and persist one document, compensating if persistence fails."""
        if not filename or len(filename) > 255:
            raise InvalidDocument("A filename between 1 and 255 characters is required")
        if len(content) > self._max_upload_size_bytes:
            raise DocumentTooLarge("The uploaded document exceeds the configured limit")
        if not content:
            raise InvalidDocument("The uploaded document is empty")

        extension = (
            "." + filename.rsplit(".", maxsplit=1)[-1].lower()
            if "." in filename
            else ""
        )
        content_type = (
            (content_type or "application/octet-stream")
            .split(";", maxsplit=1)[0]
            .strip()
            .lower()
        )
        format_info = self._CONTENT_TYPES.get(extension)
        if format_info is None:
            raise UnsupportedDocumentType(
                "Only PDF, Markdown, and plain-text files are supported"
            )
        canonical_content_type, allowed_content_types = format_info
        if content_type not in allowed_content_types:
            raise UnsupportedDocumentType(
                "The declared content type does not match the file extension"
            )

        extracted_text = self._extractor.extract(extension, content)
        if not extracted_text.strip():
            raise InvalidDocument("The document contains no extractable text")

        document_id = uuid4().hex
        storage_path = self._storage.save(document_id, extension, content)
        document = DocumentMetadata(
            id=document_id,
            filename=filename,
            content_type=canonical_content_type,
            storage_path=storage_path,
            size_bytes=len(content),
            created_at=datetime.now(UTC),
            extracted_text_length=len(extracted_text),
        )
        try:
            self._repository.add(document, extracted_text)
        except Exception:
            try:
                self._storage.delete(storage_path)
            except OSError:
                # Preserve the persistence error; cleanup failure is a documented
                # local-storage consistency limitation.
                pass
            raise
        return document

    def list_documents(self) -> list[DocumentMetadata]:
        """Return stored document metadata without extracted text or file bytes."""
        return self._repository.list()

    def get_document(self, document_id: str) -> RetrievedDocument | None:
        """Return persisted metadata and extracted text for one document."""
        return self._repository.get(document_id)
