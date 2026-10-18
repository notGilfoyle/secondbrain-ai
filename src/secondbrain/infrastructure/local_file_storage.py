"""Local storage for original uploaded document files."""

import os
import tempfile
from pathlib import Path


class LocalFileStorage:
    """Store files under a configured directory using server-generated names."""

    def __init__(self, storage_directory: Path) -> None:
        self._storage_directory = storage_directory

    def save(self, document_id: str, extension: str, content: bytes) -> str:
        """Write content to a temporary file and atomically publish its safe name."""
        self._storage_directory.mkdir(parents=True, exist_ok=True)
        storage_name = f"{document_id}{extension}"
        destination = self._storage_directory / storage_name
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{document_id}-", suffix=".tmp", dir=self._storage_directory
        )
        temporary_path = Path(temporary_name)
        published = False
        try:
            with os.fdopen(file_descriptor, "wb") as stored_file:
                stored_file.write(content)
                stored_file.flush()
                os.fsync(stored_file.fileno())
            # A hard link fails if the destination already exists, avoiding overwrite.
            os.link(temporary_path, destination)
            published = True
            temporary_path.unlink()
        except BaseException:
            if published:
                try:
                    destination.unlink(missing_ok=True)
                except Exception:
                    # Keep the original write or temporary-cleanup exception.
                    pass
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                # Cleanup is best effort and must not mask the original exception.
                pass
            raise
        return storage_name

    def delete(self, storage_path: str) -> None:
        """Remove a stored file using only a safe relative storage reference."""
        if Path(storage_path).name != storage_path or storage_path in {"", ".", ".."}:
            raise ValueError("Storage reference must be a single relative filename")
        (self._storage_directory / storage_path).unlink(missing_ok=True)
