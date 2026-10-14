from concurrent.futures import ThreadPoolExecutor

import pytest

from recsys_auto.persistence import atomic_target, write_bytes


def test_concurrent_writes_keep_one_complete_payload(tmp_path):
    path = tmp_path / "shared.bin"
    payloads = [bytes([index]) * 10000 for index in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda payload: write_bytes(path, payload), payloads))
    assert path.read_bytes() in payloads
    assert list(tmp_path.iterdir()) == [path]


def test_failed_write_preserves_existing_artifact(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        with atomic_target(path) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("interrupted")
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]
