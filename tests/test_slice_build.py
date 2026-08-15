"""Тесты построения среза.

Главное, что здесь проверяется, — что срез не теряет молча ничего: цепочки
попадают целиком, отсев всегда учтён и документами, и текстовой массой,
а потолок по длине акта не применяется сам собой.
"""

from __future__ import annotations

import json

import pytest

from mlaw.oix import read_oix, verify_oix
from mlaw.slice_build import _peek_doc_id, build, choose_acts, DocMeta
from mlaw.stream import oix_name, shard_name


def make_record(doc_id: int, act_id: int, *, index: int, n_editions: int, chars: int) -> dict:
    return {
        "doc_id": doc_id,
        "db": "MLAW",
        "text": "я" * chars,
        "text_hash": f"{doc_id:016x}",
        "status": "Действует",
        "edition": {
            "act_id": act_id,
            "n_editions": n_editions,
            "index": index,
            "is_current": index == n_editions - 1,
        },
    }


@pytest.fixture
def corpus(tmp_path):
    """Корпус, в котором цепочки намеренно разрезаны между шардами.

    Именно это свойство настоящего банка (92.2 % многоредакционных актов)
    и ломает срез по шардам, поэтому фикстура его воспроизводит.
    """
    root = tmp_path / "MLAW"
    root.mkdir()

    # act 10: три редакции, разложенные по двум шардам
    # act 20: одна редакция
    # act 30: две редакции, обе во втором шарде
    # act 40: гигант — одна редакция на 50 000 знаков
    shards = {
        0: [
            make_record(1, 10, index=0, n_editions=3, chars=100),
            make_record(2, 20, index=0, n_editions=1, chars=200),
            make_record(3, 10, index=1, n_editions=3, chars=100),
        ],
        1: [
            make_record(4, 10, index=2, n_editions=3, chars=100),
            make_record(5, 30, index=0, n_editions=2, chars=300),
            make_record(6, 30, index=1, n_editions=2, chars=300),
            make_record(7, 40, index=0, n_editions=1, chars=50_000),
        ],
    }

    for index, records in shards.items():
        payload = b"".join(
            json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in records
        )
        (root / shard_name(index)).write_bytes(payload)
        (root / oix_name(index)).write_bytes(b"")

    return root


# --------------------------------------------------------------------------- #
# Полнота цепочек
# --------------------------------------------------------------------------- #


def test_selected_acts_keep_their_whole_chain(corpus, tmp_path):
    """Взяли акт — взяли все его редакции, где бы они ни лежали.

    Акт 10 разложен по двум шардам; срез обязан собрать все три редакции.
    """
    manifest = build(str(corpus), tmp_path / "out", acts=4, seed=1, max_act_chars=None)

    assert manifest["checks"]["acts_with_truncated_chain"] == 0
    assert manifest["checks"]["acts_with_complete_chain"] == 4

    doc_ids = {
        json.loads(line)["doc_id"]
        for line in open(tmp_path / "out" / "slice.jsonl", encoding="utf-8")
    }
    assert {1, 3, 4} <= doc_ids, "все редакции акта 10 обязаны быть в срезе"


def test_every_act_has_exactly_one_current_edition(corpus, tmp_path):
    manifest = build(str(corpus), tmp_path / "out", acts=4, seed=1, max_act_chars=None)
    assert manifest["checks"]["acts_with_wrong_current_count"] == 0
    assert manifest["checks"]["acts_with_exactly_one_current"] == 4


# --------------------------------------------------------------------------- #
# Собственный сайдкар
# --------------------------------------------------------------------------- #


def test_slice_sidecar_passes_the_same_invariants_as_original_shards(corpus, tmp_path):
    out = tmp_path / "out"
    build(str(corpus), out, acts=4, seed=1, max_act_chars=None)

    entries = read_oix(out / "slice.oix")
    assert verify_oix(entries, shard_size=(out / "slice.jsonl").stat().st_size) == []


def test_sidecar_flags_match_the_records(corpus, tmp_path):
    out = tmp_path / "out"
    manifest = build(str(corpus), out, acts=4, seed=1, max_act_chars=None)

    entries = read_oix(out / "slice.oix")
    assert sum(1 for e in entries if e.is_current) == 4
    assert manifest["checks"]["current_in_sidecar"] == 4
    assert all(e.has_text for e in entries)


def test_sidecar_offsets_address_the_right_lines(corpus, tmp_path):
    """Сайдкар обязан адресовать те строки, которые рядом и записаны."""
    out = tmp_path / "out"
    build(str(corpus), out, acts=4, seed=1, max_act_chars=None)

    entries = read_oix(out / "slice.oix")
    with open(out / "slice.jsonl", "rb") as fh:
        for entry in entries:
            fh.seek(entry.offset)
            raw = fh.read(entry.length)
            assert raw.endswith(b"\n")
            assert json.loads(raw)["doc_id"] == entry.doc_id


# --------------------------------------------------------------------------- #
# Учёт отсева
# --------------------------------------------------------------------------- #


def test_accounting_reports_both_documents_and_mass(corpus, tmp_path):
    manifest = build(str(corpus), tmp_path / "out", acts=4, seed=1, max_act_chars=None)
    for step in manifest["accounting"]:
        assert "documents" in step and "chars" in step
        assert "documents_pct" in step and "chars_pct" in step


def test_length_cap_is_never_applied_silently(corpus, tmp_path):
    """Без явного потолка гигант остаётся в срезе, но о нём сообщено."""
    manifest = build(str(corpus), tmp_path / "out", acts=4, seed=1, max_act_chars=None)

    assert manifest["max_act_chars"] is None
    assert manifest["slice_profile"]["act_chars"]["max"] == 50_000
    # Предпросмотр потолка считается даже тогда, когда потолок не применён.
    assert manifest["slice_profile"]["cumulative_mass_under_cap"]


def test_length_cap_when_requested_is_accounted_for(corpus, tmp_path):
    manifest = build(str(corpus), tmp_path / "out", acts=4, seed=1, max_act_chars=1_000)

    dropped = [s for s in manifest["accounting"] if "длиннее" in s["step"]]
    assert len(dropped) == 1
    assert dropped[0]["documents"] == 1
    assert dropped[0]["chars"] == 50_000
    assert manifest["slice_profile"]["act_chars"]["max"] < 1_000


def test_mass_concentration_is_reported(corpus, tmp_path):
    """Гигант на 50 000 знаков против остальных на сотни — концентрация видна."""
    manifest = build(str(corpus), tmp_path / "out", acts=4, seed=1, max_act_chars=None)
    profile = manifest["slice_profile"]

    assert profile["mass_share_of_top_1_act"] > 90.0
    assert profile["max_to_median_ratio"] > 50
    assert profile["largest_acts"][0]["chars"] == 50_000


# --------------------------------------------------------------------------- #
# Воспроизводимость выборки
# --------------------------------------------------------------------------- #


def test_selection_is_deterministic_for_a_seed():
    by_act = {a: [] for a in [5, 3, 9, 1, 7]}
    assert choose_acts(by_act, 3, seed=42) == choose_acts(by_act, 3, seed=42)


def test_selection_does_not_depend_on_dict_insertion_order():
    """Порядок ключей словаря зависит от порядка чтения архива.

    Без сортировки перед выборкой тот же seed давал бы разные срезы —
    воспроизводимость сломалась бы незаметно.
    """
    forward = {a: [] for a in [1, 3, 5, 7, 9]}
    backward = {a: [] for a in [9, 7, 5, 3, 1]}
    assert choose_acts(forward, 3, seed=7) == choose_acts(backward, 3, seed=7)


def test_asking_for_more_acts_than_exist_returns_all():
    by_act = {a: [] for a in [1, 2, 3]}
    assert choose_acts(by_act, 99, seed=1) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Быстрый разбор doc_id
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line",
    [
        b'{"doc_id":61218,"db":"MLAW"}',
        b'{"doc_id": 61218, "db": "MLAW"}',
        b'{"db":"MLAW","doc_id":61218}',
    ],
)
def test_peek_doc_id_handles_spacing_and_position(line):
    assert _peek_doc_id(line) == 61218


def test_peek_doc_id_falls_back_to_full_parse():
    """Неожиданная раскладка обязана уронить разбор на честный json.loads."""
    assert _peek_doc_id(b'{"doc_id"\n:\n61218}') == 61218
