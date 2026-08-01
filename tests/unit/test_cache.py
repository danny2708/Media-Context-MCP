"""Cache key discrimination and store behaviour."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from media_context_mcp.cache import CacheStore, build_cache_key
from media_context_mcp.config import Settings
from media_context_mcp.models import AnalyzeMediaRequest


def key_for(settings: Settings, **overrides) -> str:
    params = dict(
        sha256="a" * 64,
        request=AnalyzeMediaRequest(path="x.png", max_chars=1000),
        processor="image",
        processor_version="2.0.0",
        prompt_version="2",
        settings=settings,
        ocr_backend="tesseract",
        ocr_version="5.4",
    )
    params.update(overrides)
    return build_cache_key(**params)


BASE = Settings(allowed_roots=[], vision_model="m", vision_api_key="k",
                vision_base_url="https://x/v1")


def test_identical_inputs_identical_key():
    assert key_for(BASE) == key_for(BASE)


def test_content_change_misses():
    assert key_for(BASE) != key_for(BASE, sha256="b" * 64)


def test_question_change_misses_but_whitespace_case_do_not():
    q1 = AnalyzeMediaRequest(path="x.png", max_chars=1000, question="Read the error")
    q2 = AnalyzeMediaRequest(path="x.png", max_chars=1000, question="  read THE error ")
    q3 = AnalyzeMediaRequest(path="x.png", max_chars=1000, question="something else")
    assert key_for(BASE, request=q1) == key_for(BASE, request=q2)
    assert key_for(BASE, request=q1) != key_for(BASE, request=q3)


def test_processor_version_change_misses():
    assert key_for(BASE) != key_for(BASE, processor_version="2.0.1")


def test_prompt_version_change_misses():
    assert key_for(BASE) != key_for(BASE, prompt_version="3")


def test_model_change_misses():
    other = BASE.model_copy(update={"vision_model": "different/model"})
    assert key_for(BASE) != key_for(other)


def test_route_change_misses():
    other = BASE.model_copy(update={"vision_route": "together"})
    assert key_for(BASE) != key_for(other)


def test_ocr_inclusion_setting_misses():
    other = BASE.model_copy(update={"send_ocr_to_cloud": False})
    assert key_for(BASE) != key_for(other)


def test_pages_normalisation():
    p1 = AnalyzeMediaRequest(path="x.pdf", max_chars=1000, pages="1, 3-5")
    p2 = AnalyzeMediaRequest(path="x.pdf", max_chars=1000, pages="1,3-5")
    p3 = AnalyzeMediaRequest(path="x.pdf", max_chars=1000, pages="2-3")
    assert key_for(BASE, request=p1) == key_for(BASE, request=p2)
    assert key_for(BASE, request=p1) != key_for(BASE, request=p3)


def test_max_chars_does_not_fragment_cache():
    r1 = AnalyzeMediaRequest(path="x.png", max_chars=1000)
    r2 = AnalyzeMediaRequest(path="x.png", max_chars=9000)
    assert key_for(BASE, request=r1) == key_for(BASE, request=r2)


def test_api_key_not_in_key_material():
    other = BASE.model_copy(update={"vision_api_key": "totally-different-key"})
    assert key_for(BASE) == key_for(other)


# ---------------------------------------------------------------------- store


def test_store_round_trip(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    key = "ab" * 32
    assert store.get(key) is None
    assert store.put(key, {"value": 42})
    assert store.get(key) == {"value": 42}


def test_corrupt_entry_is_deleted_and_missed(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    key = "cd" * 32
    store.put(key, {"ok": True})
    entry_path = tmp_path / "cache" / f"{key}.json"
    entry_path.write_text("{not valid json", encoding="utf-8")
    assert store.get(key) is None
    assert not entry_path.exists()


def test_schema_mismatch_is_missed(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    key = "ef" * 32
    entry_path = tmp_path / "cache"
    entry_path.mkdir(parents=True)
    (entry_path / f"{key}.json").write_text(
        json.dumps({"schema": 999, "stored_at": 0, "payload": {}}), encoding="utf-8"
    )
    assert store.get(key) is None


def test_ttl_expiry(tmp_path: Path):
    store_short = CacheStore(tmp_path / "cache2", ttl_days=1)
    key = "12" * 32
    store_short.put(key, {"v": 1})
    # backdate the entry beyond the TTL
    entry_path = tmp_path / "cache2" / f"{key}.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["stored_at"] = 0.0
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    assert store_short.get(key) is None


def test_cleanup_evicts_to_size_budget(tmp_path: Path):
    store = CacheStore(tmp_path / "cache", max_bytes=400, ttl_days=30)
    for index in range(6):
        store.put(f"{index:02d}" * 32, {"blob": "x" * 100})
    stats = store.cleanup()
    assert stats["bytes"] <= 400
    assert stats["evicted"] > 0


def test_concurrent_writes_do_not_corrupt(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    key = "aa" * 32
    errors: list[Exception] = []

    def writer(value: int) -> None:
        try:
            for _ in range(20):
                store.put(key, {"value": value, "pad": "y" * 500})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    payload = store.get(key)
    assert payload is not None and payload["value"] in range(6)  # some complete write


def test_disabled_store_is_inert(tmp_path: Path):
    store = CacheStore(tmp_path / "cache", enabled=False)
    assert not store.put("ab" * 32, {"v": 1})
    assert store.get("ab" * 32) is None
