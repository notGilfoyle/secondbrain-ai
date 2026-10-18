"""Text extraction for supported local document formats."""

from io import BytesIO

from pypdf import PdfReader

from secondbrain.application.documents import InvalidDocument


class DocumentTextExtractor:
    """Extract UTF-8 text or PDF page text without applying OCR."""

    def extract(self, extension: str, content: bytes) -> str:
        """Extract text based on the already-validated file extension."""
        if extension in {".md", ".txt"}:
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InvalidDocument("The document is not valid UTF-8") from exc

        if extension == ".pdf":
            try:
                reader = PdfReader(BytesIO(content), strict=True)
                if reader.is_encrypted:
                    raise InvalidDocument("Encrypted PDFs are not supported")
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except InvalidDocument:
                raise
            except Exception as exc:
                raise InvalidDocument("The PDF is malformed or cannot be read") from exc

        raise InvalidDocument("The document format is not supported")
