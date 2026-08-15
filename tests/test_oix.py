"""Тесты формата `.oix`.

Проверяется ровно то, что способно тихо сломать всё остальное: ширина полей,
трактовка `length`, непрерывность офсетов и отсутствие предположения о
сортировке `doc_id`.
"""

from __future__ import annotations

import struct

import pytest

from mlaw.oix import (
    ENTRY_SIZE,
    FLAG_HAS_CONTENTS,
    FLAG_IS_CURRENT,
    FLAG_IS_STUB,
    OixEntry,
    OixIndex,
    read_line,
    read_oix,
    verify_oix,
    write_oix,
)


def make_entries(lengths: list[int], doc_ids: list[int] | None = None) -> list[OixEntry]:
    """Строит непрерывную цепочку записей с заданными длинами строк."""
    doc_ids = doc_ids or list(range(1, len(lengths) + 1))
    entries = []
    offset = 0
    for length, doc_id in zip(lengths, doc_ids, strict=True):
        entries.append(OixEntry(offset=offset, length=length, doc_id=doc_id, flags=0, text_hash=1))
        offset += length
    return entries


# --------------------------------------------------------------------------- #
# Раскладка структуры
# --------------------------------------------------------------------------- #


def test_entry_size_is_28_bytes():
    """28 Б на запись — размер, от которого считаются все офсеты в сайдкаре."""
    assert ENTRY_SIZE == 28


def test_padding_is_not_data():
    """Три байта выравнивания обязаны быть пропущены, а не прочитаны как поле."""
    packed = OixEntry(offset=1, length=2, doc_id=3, flags=4, text_hash=5).pack()
    # Подменяем байты выравнивания (позиции 17..19) — разбор не должен измениться.
    corrupted = packed[:17] + b"\xff\xff\xff" + packed[20:]
    assert OixEntry.unpack(corrupted) == OixEntry.unpack(packed)


def test_roundtrip_preserves_all_fields():
    entry = OixEntry(offset=2**40, length=123456, doc_id=445611, flags=0b10110, text_hash=2**63 + 7)
    assert OixEntry.unpack(entry.pack()) == entry


# --------------------------------------------------------------------------- #
# Ловушка 1: offset обязан быть u64
# --------------------------------------------------------------------------- #


def test_offset_beyond_u32_survives_roundtrip():
    """Офсет за границей u32 обязан пережить запись и чтение.

    На этой выгрузке максимальный offset+length шарда 0000 равен 501 970 395 —
    в u32 он сегодня влезает. Именно поэтому ошибка «записать u32» тут и не
    проявляется на глаз: она сработает при первом же росте шарда.
    """
    big = 2**32 + 12345
    entry = OixEntry(offset=big, length=10, doc_id=1, flags=0, text_hash=0)
    assert OixEntry.unpack(entry.pack()).offset == big


def test_u32_layout_would_lose_the_offset():
    """Контроль: та же запись в раскладке с u32-офсетом теряет значение.

    Тест существует, чтобы предыдущий не был вечно-зелёным: он показывает, что
    проверяемое свойство действительно различает правильную и неправильную
    раскладку.
    """
    big = 2**32 + 12345
    wrong = struct.Struct("<IIIB3xQ")  # offset как u32
    with pytest.raises(struct.error):
        wrong.pack(big, 10, 1, 0, 0)


def test_offset_out_of_range_is_rejected_on_write():
    with pytest.raises(ValueError, match="offset"):
        OixEntry(offset=2**64, length=1, doc_id=1, flags=0, text_hash=0).pack()


# --------------------------------------------------------------------------- #
# Ловушка 2: length включает перевод строки
# --------------------------------------------------------------------------- #


def test_length_includes_newline_and_helper_strips_it():
    entry = OixEntry(offset=0, length=11, doc_id=1, flags=0, text_hash=0)
    assert entry.length == 11
    assert entry.length_without_newline == 10


def test_read_line_strips_newline_by_default(tmp_path):
    shard = tmp_path / "shard.jsonl"
    shard.write_bytes(b'{"a":1}\n{"b":22}\n')
    entries = make_entries([8, 9])

    with open(shard, "rb") as fh:
        assert read_line(fh, entries[0]) == b'{"a":1}'
        assert read_line(fh, entries[1]) == b'{"b":22}'
        assert read_line(fh, entries[1], keep_newline=True) == b'{"b":22}\n'


def test_read_line_detects_sidecar_shard_mismatch(tmp_path):
    """Если сайдкар не соответствует шарду, читатель обязан упасть, а не молчать."""
    shard = tmp_path / "shard.jsonl"
    shard.write_bytes(b'{"a":1}\n')
    off_by_one = OixEntry(offset=0, length=7, doc_id=1, flags=0, text_hash=0)
    with open(shard, "rb") as fh:
        with pytest.raises(ValueError, match=r"\\n"):
            read_line(fh, off_by_one)


# --------------------------------------------------------------------------- #
# Ловушка 3: doc_id не отсортирован
# --------------------------------------------------------------------------- #


def test_lookup_works_on_unsorted_doc_ids():
    """Замер на шарде 0000: 458 инверсий на 9 522 пары. bisect тут не годится."""
    entries = make_entries([10, 10, 10, 10], doc_ids=[61218, 61219, 55697, 138030])
    index = OixIndex(entries)

    assert index[55697].offset == 20
    assert index[138030].offset == 30
    assert index.get(999) is None
    assert 61218 in index
    # Порядок записей сохраняется — он же порядок строк в шарде.
    assert index.doc_ids() == [61218, 61219, 55697, 138030]


def test_duplicate_doc_id_is_rejected():
    entries = make_entries([10, 10], doc_ids=[7, 7])
    with pytest.raises(ValueError, match="дважды"):
        OixIndex(entries)


# --------------------------------------------------------------------------- #
# Инварианты
# --------------------------------------------------------------------------- #


def test_verify_accepts_a_well_formed_sidecar():
    entries = make_entries([12900, 4708, 25239])
    assert verify_oix(entries, shard_size=42847) == []


def test_verify_detects_gap():
    entries = make_entries([10, 10, 10])
    broken = list(entries)
    broken[1] = OixEntry(offset=11, length=10, doc_id=2, flags=0, text_hash=1)
    kinds = {v.kind for v in verify_oix(broken)}
    assert "contiguity" in kinds


def test_verify_detects_wrong_total_size():
    entries = make_entries([10, 10])
    problems = verify_oix(entries, shard_size=21)
    assert [p.kind for p in problems] == ["eof"]
    assert "+1" in str(problems[0])


def test_verify_detects_nonzero_start():
    entries = [OixEntry(offset=4, length=10, doc_id=1, flags=0, text_hash=0)]
    assert any(p.kind == "start" for p in verify_oix(entries))


# --------------------------------------------------------------------------- #
# Флаги
# --------------------------------------------------------------------------- #


def test_flag_bits():
    entry = OixEntry(
        offset=0, length=1, doc_id=1, flags=FLAG_IS_STUB | FLAG_IS_CURRENT, text_hash=0
    )
    assert entry.is_stub
    assert entry.is_current
    assert not entry.has_contents
    assert not entry.has_text  # text_hash == 0 означает «текста нет»

    other = OixEntry(offset=0, length=1, doc_id=2, flags=FLAG_HAS_CONTENTS, text_hash=5)
    assert other.has_contents
    assert other.has_text
    assert not other.is_current


def test_current_entries_selection():
    entries = [
        OixEntry(offset=0, length=10, doc_id=1, flags=FLAG_IS_CURRENT, text_hash=1),
        OixEntry(offset=10, length=10, doc_id=2, flags=0, text_hash=1),
        OixEntry(offset=20, length=10, doc_id=3, flags=FLAG_IS_CURRENT, text_hash=1),
    ]
    index = OixIndex(entries)
    assert [e.doc_id for e in index.current_entries()] == [1, 3]
    assert index.total_bytes() == 30


# --------------------------------------------------------------------------- #
# Запись
# --------------------------------------------------------------------------- #


def test_write_then_read_roundtrip(tmp_path):
    """Собственный сайдкар должен читаться тем же кодом, что и оригинальные."""
    entries = make_entries([100, 250, 33], doc_ids=[5, 3, 9])
    path = tmp_path / "slice.oix"

    assert write_oix(path, entries) == 3
    assert path.stat().st_size == 3 * ENTRY_SIZE

    loaded = read_oix(path)
    assert loaded == entries
    assert verify_oix(loaded, shard_size=383) == []


def test_truncated_sidecar_is_rejected(tmp_path):
    path = tmp_path / "broken.oix"
    path.write_bytes(b"\x00" * (ENTRY_SIZE + 7))
    with pytest.raises(ValueError, match="не кратна"):
        read_oix(path)
