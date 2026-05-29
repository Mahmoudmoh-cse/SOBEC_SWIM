from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings


class StoredUpload:
    def __init__(self, original_filename: str, stored_path: str, content_type: str | None, size_bytes: int) -> None:
        self.original_filename = original_filename
        self.stored_path = stored_path
        self.content_type = content_type
        self.size_bytes = size_bytes


class StorageProvider(ABC):
    @abstractmethod
    async def save_video(self, file: UploadFile, swimmer_id: str, session_id: str) -> StoredUpload:
        raise NotImplementedError

    @abstractmethod
    async def save_video_bytes(
        self,
        data: bytes,
        original_filename: str,
        content_type: str | None,
        swimmer_id: str,
        session_id: str,
    ) -> StoredUpload:
        raise NotImplementedError


class LocalStorageProvider(StorageProvider):
    def __init__(self, upload_dir: Path | None = None) -> None:
        settings = get_settings()
        self.upload_dir = Path(upload_dir or settings.upload_dir)

    async def save_video(self, file: UploadFile, swimmer_id: str, session_id: str) -> StoredUpload:
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        safe_name = f"{uuid4()}{suffix.lower()}"
        target_dir = self.upload_dir / swimmer_id / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name

        size = 0
        with target_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                handle.write(chunk)

        return StoredUpload(
            original_filename=file.filename or safe_name,
            stored_path=str(target_path),
            content_type=file.content_type,
            size_bytes=size,
        )

    async def save_video_bytes(
        self,
        data: bytes,
        original_filename: str,
        content_type: str | None,
        swimmer_id: str,
        session_id: str,
    ) -> StoredUpload:
        suffix = Path(original_filename or "video.mp4").suffix or ".mp4"
        safe_name = f"{uuid4()}{suffix.lower()}"
        target_dir = self.upload_dir / swimmer_id / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        target_path.write_bytes(data)

        return StoredUpload(
            original_filename=original_filename or safe_name,
            stored_path=str(target_path),
            content_type=content_type,
            size_bytes=len(data),
        )


def get_storage_provider() -> StorageProvider:
    return LocalStorageProvider()
