"""Атомарная запись с уникальными временными файлами для параллельных читателей."""

import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_target(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bytes(path: Path, data: bytes) -> None:
    with atomic_target(path) as temporary:
        temporary.write_bytes(data)
