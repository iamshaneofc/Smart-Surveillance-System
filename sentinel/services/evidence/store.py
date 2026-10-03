import hashlib
from dataclasses import dataclass
from pathlib import Path

from packages.common.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StoredBlob:
    uri: str
    sha256: str
    size_bytes: int
    content_type: str | None


class EvidenceStoreError(Exception):
    pass


class EvidenceStore:
    backend = "base"

    def save(
        self,
        camera_id: str,
        event_id: str,
        name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> StoredBlob:
        raise NotImplementedError

    def delete(self, uri: str) -> None:
        raise NotImplementedError

    def exists(self, uri: str) -> bool:
        raise NotImplementedError


class MemoryEvidenceStore(EvidenceStore):
    backend = "memory"

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}
        self._types: dict[str, str | None] = {}

    def save(self, camera_id, event_id, name, data, content_type=None) -> StoredBlob:
        uri = f"memory://{camera_id}/{event_id}/{name}"
        self._blobs[uri] = data
        self._types[uri] = content_type
        digest = hashlib.sha256(data).hexdigest()
        return StoredBlob(uri=uri, sha256=digest, size_bytes=len(data), content_type=content_type)

    def delete(self, uri: str) -> None:
        self._blobs.pop(uri, None)
        self._types.pop(uri, None)

    def exists(self, uri: str) -> bool:
        return uri in self._blobs

    def read(self, uri: str) -> bytes:
        return self._blobs[uri]


class LocalDiskEvidenceStore(EvidenceStore):
    backend = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, camera_id, event_id, name, data, content_type=None) -> StoredBlob:
        target_dir = self.root / camera_id / event_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        digest = hashlib.sha256(data).hexdigest()
        uri = str(target.relative_to(self.root))
        log.info(
            "evidence stored",
            extra={"camera_id": camera_id, "event_id": event_id, "name": name, "size": len(data)},
        )
        return StoredBlob(uri=uri, sha256=digest, size_bytes=len(data), content_type=content_type)

    def delete(self, uri: str) -> None:
        target = self.root / uri
        if target.exists():
            target.unlink()

    def exists(self, uri: str) -> bool:
        return (self.root / uri).exists()

    def read(self, uri: str) -> bytes:
        return (self.root / uri).read_bytes()
