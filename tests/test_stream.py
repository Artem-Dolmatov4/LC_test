"""Тесты потокового чтения корпуса.

Синтетический архив собирается в фикстуре, поэтому тесты идут без исходных
16.9 ГБ. Отдельным блоком — интеграционные проверки на настоящем архиве,
которые пропускаются, если его нет на месте.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from mlaw.oix import OixEntry, read_oix, verify_oix, write_oix
from mlaw.stream import (
    ArchiveCorpus,
    DirectoryCorpus,
    open_corpus,
    oix_name,
    shard_name,
)

REAL_ARCHIVE = Path(__file__).resolve().parents[1] / "MLAW_dataset.tar.zst"
needs_real_archive = pytest.mark.skipif(
    not REAL_ARCHIVE.exists(), reason="настоящий архив MLAW_dataset.tar.zst недоступен"
)


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #


def _records(shard: int, count: int) -> list[dict]:
    return [
        {"doc_id": shard * 100 + i, "db": "MLAW", "text": "текст " * (i + 1)}
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def fake_corpus(tmp_path_factory) -> dict:
    """Собирает маленький корпус: папка MLAW/ и упакованный из неё .tar.zst."""
    root = tmp_path_factory.mktemp("corpus")
    mlaw = root / "MLAW"
    mlaw.mkdir()

    expected: dict[int, list[dict]] = {}
    for shard in (0, 1):
        records = _records(shard, 3)
        expected[shard] = records

        payload = b"".join(
            json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in records
        )
        (mlaw / shard_name(shard)).write_bytes(payload)

        entries, offset = [], 0
        for r, raw in zip(records, payload.split(b"\n")[:-1], strict=True):
            length = len(raw) + 1
            entries.append(
                OixEntry(offset=offset, length=length, doc_id=r["doc_id"], flags=0, text_hash=1)
            )
            offset += length
        write_oix(mlaw / oix_name(shard), entries)

    (mlaw / "mlaw_stats.json").write_text(json.dumps({"db": "MLAW", "shards": 2}))

    tar_path = root / "corpus.tar"
    with tarfile.open(tar_path, "w") as tar:
        for path in sorted(mlaw.iterdir()):
            tar.add(path, arcname=f"MLAW/{path.name}")

    archive = root / "corpus.tar.zst"
    subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(archive)], check=True)

    return {"dir": mlaw, "archive": archive, "expected": expected}


# --------------------------------------------------------------------------- #
# Чтение из архива
# --------------------------------------------------------------------------- #


def test_archive_reads_all_shards(fake_corpus):
    corpus = ArchiveCorpus(fake_corpus["archive"])
    records = list(corpus.iter_records())
    assert [r["doc_id"] for r in records] == [0, 1, 2, 100, 101, 102]


def test_archive_offsets_match_sidecar(fake_corpus):
    """Офсеты, посчитанные при потоковом чтении, обязаны совпасть с сайдкаром.

    Это связка двух модулей: если `stream` считает длину строки без `\\n`,
    а `oix` — с ним, расхождение вылезет здесь.
    """
    corpus = ArchiveCorpus(fake_corpus["archive"])
    streamed = [r for r in corpus.iter_lines(shards=[0])]
    sidecar = read_oix(fake_corpus["dir"] / oix_name(0))

    assert [r.offset for r in streamed] == [e.offset for e in sidecar]
    assert [r.length for r in streamed] == [e.length for e in sidecar]
    # `line` отдаётся уже без перевода строки, а `length` его учитывает.
    assert all(len(r.line) == r.length - 1 for r in streamed)


def test_archive_shard_subset(fake_corpus):
    corpus = ArchiveCorpus(fake_corpus["archive"])
    assert [r["doc_id"] for r in corpus.iter_records(shards=[1])] == [100, 101, 102]


def test_archive_reads_sidecar_and_meta(fake_corpus):
    corpus = ArchiveCorpus(fake_corpus["archive"])
    assert len(corpus.read_sidecar(0)) == 3 * 28
    assert json.loads(corpus.read_meta("mlaw_stats.json"))["shards"] == 2


def test_archive_missing_member_raises(fake_corpus):
    corpus = ArchiveCorpus(fake_corpus["archive"])
    with pytest.raises(KeyError):
        corpus.read_meta("mlaw_nonexistent.json")


def test_archive_extract(fake_corpus, tmp_path):
    corpus = ArchiveCorpus(fake_corpus["archive"])
    written = corpus.extract([shard_name(0), oix_name(0)], tmp_path)
    assert {p.name for p in written} == {shard_name(0), oix_name(0)}
    assert (tmp_path / shard_name(0)).read_bytes() == (
        fake_corpus["dir"] / shard_name(0)
    ).read_bytes()


def test_archive_rejects_bad_shard_index(fake_corpus):
    corpus = ArchiveCorpus(fake_corpus["archive"])
    with pytest.raises(ValueError, match="вне диапазона"):
        list(corpus.iter_lines(shards=[99]))


# --------------------------------------------------------------------------- #
# Папка и эквивалентность источников
# --------------------------------------------------------------------------- #


def test_directory_matches_archive(fake_corpus):
    """Два источника обязаны давать побайтово одно и то же."""
    from_archive = list(ArchiveCorpus(fake_corpus["archive"]).iter_lines())
    from_dir = list(DirectoryCorpus(fake_corpus["dir"]).iter_lines())
    assert from_archive == from_dir


def test_open_corpus_dispatch(fake_corpus):
    assert isinstance(open_corpus(fake_corpus["archive"]), ArchiveCorpus)
    assert isinstance(open_corpus(fake_corpus["dir"]), DirectoryCorpus)
    with pytest.raises(ValueError):
        open_corpus(fake_corpus["dir"] / "mlaw_stats.json")


def test_directory_available_shards(fake_corpus):
    assert DirectoryCorpus(fake_corpus["dir"]).available_shards() == [0, 1]


def test_subprocess_fallback_matches_library(fake_corpus, monkeypatch):
    """Откат на бинарь `zstd` обязан давать тот же результат, что и библиотека."""
    import builtins

    real_import = builtins.__import__

    def no_zstandard(name, *args, **kwargs):
        if name == "zstandard":
            raise ImportError("отключено для теста")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_zstandard)
    fallback = list(ArchiveCorpus(fake_corpus["archive"]).iter_records())
    monkeypatch.undo()

    assert fallback == list(ArchiveCorpus(fake_corpus["archive"]).iter_records())


# --------------------------------------------------------------------------- #
# Интеграция с настоящим корпусом
# --------------------------------------------------------------------------- #


@needs_real_archive
@pytest.mark.slow
def test_real_sidecar_invariants_shard_0000():
    """Инварианты сайдкара на настоящем шарде 0000.

    Числа — собственный замер по этой выгрузке, а не из документации датасета.
    """
    corpus = ArchiveCorpus(REAL_ARCHIVE)
    entries = read_oix(__import__("io").BytesIO(corpus.read_sidecar(0)))

    assert len(entries) == 9523
    assert verify_oix(entries, shard_size=501_970_395) == []

    assert sum(1 for e in entries if e.has_contents) == 4287
    assert sum(1 for e in entries if e.is_current) == 7539
    assert sum(1 for e in entries if e.is_stub) == 0
    assert all(e.has_text for e in entries)

    doc_ids = [e.doc_id for e in entries]
    inversions = sum(1 for a, b in zip(doc_ids, doc_ids[1:], strict=False) if a >= b)
    assert inversions == 458, "doc_id в сайдкаре не отсортирован — bisect применять нельзя"


@needs_real_archive
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MLAW_SLOW_TESTS") != "1",
    reason="полный проход по шарду; включается MLAW_SLOW_TESTS=1",
)
def test_real_stream_offsets_match_sidecar_shard_0000():
    """Офсеты потокового чтения совпадают с сайдкаром на всех 9 523 строках."""
    corpus = ArchiveCorpus(REAL_ARCHIVE)
    sidecar = read_oix(__import__("io").BytesIO(corpus.read_sidecar(0)))

    for record, entry in zip(corpus.iter_lines(shards=[0]), sidecar, strict=True):
        assert record.offset == entry.offset
        assert record.length == entry.length
        assert json.loads(record.line)["doc_id"] == entry.doc_id
