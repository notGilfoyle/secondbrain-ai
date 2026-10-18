"""Public document API response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    """Metadata returned after document ingestion or in a list response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
    extracted_text_length: int


class DocumentListResponse(BaseModel):
    """Collection of document metadata."""

    documents: list[DocumentResponse]
