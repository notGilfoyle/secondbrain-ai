"""SQLite persistence for document metadata and extracted text."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from secondbrain.application.documents import (
    DocumentChunk,
    DocumentMetadata,
    RetrievedDocument,
)


class SQLiteDocumentRepository:
    """Persist document records in a local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def add(
        self,
        document: DocumentMetadata,
        extracted_text: str,
        chunks: list[DocumentChunk],
    ) -> None:
        """Insert a document and all its chunks in one transaction."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                self._create_table(connection)
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, filename, content_type, storage_path, size_bytes,
                        extracted_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.filename,
                        document.content_type,
                        document.storage_path,
                        document.size_bytes,
                        extracted_text,
                        document.created_at.isoformat(),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, chunk_index, text, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            chunk.document_id,
                            chunk.chunk_index,
                            chunk.text,
                            chunk.created_at.isoformat(),
                        )
                        for chunk in chunks
                    ],
                )

    def list(self) -> list[DocumentMetadata]:
        """Return metadata newest-first, without loading extracted text."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                self._create_table(connection)
                rows = connection.execute(
                    """
                    SELECT id, filename, content_type, storage_path, size_bytes,
                           created_at, length(extracted_text)
                    FROM documents
                    ORDER BY created_at DESC, id DESC
                    """
                ).fetchall()
        return [
            DocumentMetadata(
                id=row[0],
                filename=row[1],
                content_type=row[2],
                storage_path=row[3],
                size_bytes=row[4],
                created_at=datetime.fromisoformat(row[5]),
                extracted_text_length=row[6],
            )
            for row in rows
        ]

    def get(self, document_id: str) -> RetrievedDocument | None:
        """Load one document's metadata and extracted text by its ID."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                self._create_table(connection)
                row = connection.execute(
                    """
                    SELECT id, filename, content_type, storage_path, size_bytes,
                           created_at, length(extracted_text), extracted_text
                    FROM documents
                    WHERE id = ?
                    """,
                    (document_id,),
                ).fetchone()
        if row is None:
            return None
        metadata = DocumentMetadata(
            id=row[0],
            filename=row[1],
            content_type=row[2],
            storage_path=row[3],
            size_bytes=row[4],
            created_at=datetime.fromisoformat(row[5]),
            extracted_text_length=row[6],
        )
        return RetrievedDocument(metadata=metadata, extracted_text=row[7])

    @staticmethod
    def _create_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                storage_path TEXT NOT NULL UNIQUE,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                extracted_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (document_id, chunk_index)
            )
            """
        )
