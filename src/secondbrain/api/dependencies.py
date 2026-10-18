"""FastAPI dependency providers."""

from secondbrain.application.documents import DocumentService
from secondbrain.application.health import HealthService
from secondbrain.config.settings import get_settings
from secondbrain.infrastructure.document_parser import DocumentTextExtractor
from secondbrain.infrastructure.local_file_storage import LocalFileStorage
from secondbrain.infrastructure.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


def get_health_service() -> HealthService:
    """Construct the health application service."""
    return HealthService()


def get_document_service() -> DocumentService:
    """Build the document use case with configured local adapters."""
    settings = get_settings()
    return DocumentService(
        extractor=DocumentTextExtractor(),
        storage=LocalFileStorage(settings.document_storage_path),
        repository=SQLiteDocumentRepository(settings.database_path),
        max_upload_size_bytes=settings.max_upload_size_bytes,
    )
