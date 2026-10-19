"""HTTP endpoints for ingesting and listing documents."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status
from starlette.concurrency import run_in_threadpool

from secondbrain.api.dependencies import get_document_service
from secondbrain.api.schemas.documents import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
)
from secondbrain.application.documents import (
    DocumentService,
    DocumentTooLarge,
    InvalidDocument,
    UnsupportedDocumentType,
)
from secondbrain.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
DocumentServiceDependency = Annotated[DocumentService, Depends(get_document_service)]


async def _read_bounded(upload: UploadFile, maximum_bytes: int) -> bytes:
    """Read no more than the configured limit plus one byte into memory."""
    if upload.size is not None and upload.size > maximum_bytes:
        raise DocumentTooLarge("The uploaded document exceeds the configured limit")

    content = bytearray()
    while True:
        remaining = maximum_bytes - len(content)
        chunk = await upload.read(min(64 * 1024, remaining + 1))
        if not chunk:
            return bytes(content)
        content.extend(chunk)
        if len(content) > maximum_bytes:
            raise DocumentTooLarge("The uploaded document exceeds the configured limit")


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    service: DocumentServiceDependency,
    upload: Annotated[UploadFile, File(alias="file")],
) -> DocumentResponse:
    """Extract text and retain a PDF, Markdown, or plain-text upload."""
    settings = get_settings()
    try:
        try:
            content = await _read_bounded(upload, settings.max_upload_size_bytes)
        finally:
            await upload.close()
        document = await run_in_threadpool(
            service.ingest, upload.filename or "", upload.content_type, content
        )
        return DocumentResponse.model_validate(document)
    except DocumentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected document ingestion failure")
        raise HTTPException(
            status_code=500, detail="Document ingestion failed"
        ) from exc


@router.get("", response_model=DocumentListResponse)
def list_documents(service: DocumentServiceDependency) -> DocumentListResponse:
    """List stored document metadata without returning extracted text."""
    try:
        return DocumentListResponse(
            documents=[
                DocumentResponse.model_validate(document)
                for document in service.list_documents()
            ]
        )
    except Exception as exc:
        logger.exception("Unexpected document listing failure")
        raise HTTPException(status_code=500, detail="Document listing failed") from exc


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    responses={404: {"description": "Document not found"}},
)
def get_document(
    document_id: Annotated[str, Path(pattern="^[0-9a-fA-F]{32}$")],
    service: DocumentServiceDependency,
) -> DocumentDetailResponse:
    """Retrieve document metadata and extracted text by its UUID hex ID."""
    try:
        document = service.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentDetailResponse(
            **DocumentResponse.model_validate(document.metadata).model_dump(),
            extracted_text=document.extracted_text,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected document retrieval failure")
        raise HTTPException(
            status_code=500, detail="Document retrieval failed"
        ) from exc
