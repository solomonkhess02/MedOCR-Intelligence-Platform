"""
Service — Storage
Handles file saving and SHA-256 hash deduplication.
"""

import hashlib
import os
import shutil
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile

from app.config import get_settings

settings = get_settings()


def _ensure_upload_dir() -> Path:
    """Create upload directory if it doesn't exist."""
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


async def save_upload_file(upload_file: UploadFile) -> tuple[str, str]:
    """
    Save an uploaded file to disk.

    Returns:
        (saved_path, sha256_hash) tuple.
        saved_path: absolute path to saved file
        sha256_hash: SHA-256 hex digest for deduplication
    """
    upload_dir = _ensure_upload_dir()

    # Read file content into memory for hashing + saving
    content = await upload_file.read()

    # Compute SHA-256 hash
    sha256_hash = hashlib.sha256(content).hexdigest()

    # Build a unique filename to avoid collisions on disk
    ext = Path(upload_file.filename or "file").suffix
    unique_name = f"{uuid.uuid4()}{ext}"
    save_path = upload_dir / unique_name

    # Write file asynchronously
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    return str(save_path.resolve()), sha256_hash


def delete_file(path: str) -> None:
    """Delete a file from disk (used for cleanup on failure)."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def get_max_size_bytes() -> int:
    """Return max allowed upload size in bytes."""
    return settings.max_file_size_mb * 1024 * 1024
