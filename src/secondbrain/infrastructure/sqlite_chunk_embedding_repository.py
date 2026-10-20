"""SQLite persistence for vectors generated from document chunks."""

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path

from secondbrain.application.documents import DocumentChunk
from secondbrain.application.embeddings import ChunkEmbedding


class SQLiteChunkEmbeddingRepository:
    """Persist chunk vectors keyed by chunk ID and model identity."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def list_chunks_without_embeddings(self, model_name: str) -> list[DocumentChunk]:
        """Return chunks missing vectors for the given model, in stable order."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                self._create_tables(connection)
                has_chunks_table = connection.execute(
                    """SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'document_chunks'"""
                ).fetchone()
                if has_chunks_table is None:
                    rows = []
                else:
                    rows = connection.execute(
                        """
                        SELECT c.id, c.document_id, c.chunk_index, c.text, c.created_at
                        FROM document_chunks AS c
                        WHERE NOT EXISTS (
                            SELECT 1 FROM chunk_embeddings AS e
                            WHERE e.chunk_id = c.id AND e.model_name = ?
                        )
                        ORDER BY c.document_id, c.chunk_index
                        """,
                        (model_name,),
                    ).fetchall()
        return [
            DocumentChunk(
                id=row[0],
                document_id=row[1],
                chunk_index=row[2],
                text=row[3],
                created_at=datetime.fromisoformat(row[4]),
            )
            for row in rows
        ]

    def save_embeddings(self, embeddings: Sequence[ChunkEmbedding]) -> int:
        """Persist vectors atomically and return the number newly inserted."""
        if not embeddings:
            return 0
        chunk_ids = [embedding.chunk_id for embedding in embeddings]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("embedding batch contains duplicate chunk IDs")
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        dimensions_by_model: dict[str, int] = {}
        for embedding in embeddings:
            if embedding.dimensions <= 0:
                raise ValueError("embedding dimensions must be positive")
            previous_dimensions = dimensions_by_model.setdefault(
                embedding.model_name, embedding.dimensions
            )
            if previous_dimensions != embedding.dimensions:
                raise ValueError("each model must use consistent vector dimensions")

        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                self._create_tables(connection)
                for model_name, dimensions in dimensions_by_model.items():
                    connection.execute(
                        """
                        INSERT INTO embedding_models (model_name, dimensions)
                        VALUES (?, ?)
                        ON CONFLICT(model_name) DO NOTHING
                        """,
                        (model_name, dimensions),
                    )
                    stored = connection.execute(
                        "SELECT dimensions FROM embedding_models WHERE model_name = ?",
                        (model_name,),
                    ).fetchone()
                    if stored is None or stored[0] != dimensions:
                        raise ValueError(
                            "embedding dimensions do not match the stored "
                            "model identity"
                        )

                cursor = connection.executemany(
                    """
                    INSERT INTO chunk_embeddings (
                        chunk_id, model_name, dimensions, vector_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id, model_name) DO NOTHING
                    """,
                    [
                        (
                            embedding.chunk_id,
                            embedding.model_name,
                            embedding.dimensions,
                            json.dumps(
                                embedding.vector,
                                allow_nan=False,
                                separators=(",", ":"),
                            ),
                            embedding.created_at.isoformat(),
                        )
                        for embedding in embeddings
                    ],
                )
                inserted_count = cursor.rowcount
        return inserted_count

    def list_embeddings(self, model_name: str) -> list[ChunkEmbedding]:
        """Load persisted vectors for one model in stable chunk order."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                self._create_tables(connection)
                rows = connection.execute(
                    """
                    SELECT chunk_id, model_name, dimensions, vector_json, created_at
                    FROM chunk_embeddings
                    WHERE model_name = ?
                    ORDER BY chunk_id
                    """,
                    (model_name,),
                ).fetchall()
        embeddings = []
        for row in rows:
            vector = tuple(float(value) for value in json.loads(row[3]))
            if len(vector) != row[2]:
                raise ValueError("stored embedding vector has an invalid dimension")
            embeddings.append(
                ChunkEmbedding(
                    chunk_id=row[0],
                    model_name=row[1],
                    vector=vector,
                    created_at=datetime.fromisoformat(row[4]),
                )
            )
        return embeddings

    @staticmethod
    def _create_tables(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_models (
                model_name TEXT PRIMARY KEY,
                dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                UNIQUE (model_name, dimensions)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id TEXT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model_name),
                FOREIGN KEY (model_name, dimensions)
                    REFERENCES embedding_models(model_name, dimensions)
                    ON DELETE CASCADE
            )
            """
        )
