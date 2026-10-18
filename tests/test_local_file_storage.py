"""Failure-path tests for local original-file storage."""

from pathlib import Path

import pytest

from secondbrain.infrastructure.local_file_storage import LocalFileStorage


def test_save_removes_published_file_when_temp_unlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage_directory = tmp_path / "files"
    storage = LocalFileStorage(storage_directory)
    real_unlink = Path.unlink
    temp_unlink_attempts = 0

    def fail_first_temp_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal temp_unlink_attempts
        if path.suffix == ".tmp":
            temp_unlink_attempts += 1
            if temp_unlink_attempts == 1:
                raise PermissionError("original temporary cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_first_temp_unlink)

    with pytest.raises(PermissionError, match="original temporary cleanup failure"):
        storage.save("document-id", ".txt", b"content")

    assert temp_unlink_attempts == 2
    assert list(storage_directory.iterdir()) == []


def test_cleanup_errors_do_not_mask_original_temp_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage_directory = tmp_path / "files"
    storage = LocalFileStorage(storage_directory)
    real_unlink = Path.unlink
    temp_unlink_attempts = 0
    destination_unlink_attempts = 0

    def fail_cleanup_paths(path: Path, missing_ok: bool = False) -> None:
        nonlocal temp_unlink_attempts, destination_unlink_attempts
        if path.suffix == ".tmp":
            temp_unlink_attempts += 1
            if temp_unlink_attempts == 1:
                raise PermissionError("original temporary cleanup failure")
            raise RuntimeError("secondary temporary cleanup failure")
        if path.name == "document-id.txt":
            destination_unlink_attempts += 1
            raise RuntimeError("published-file cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_cleanup_paths)

    with pytest.raises(PermissionError, match="original temporary cleanup failure"):
        storage.save("document-id", ".txt", b"content")

    assert destination_unlink_attempts == 1
    assert temp_unlink_attempts == 2
