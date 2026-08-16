"""Тесты цитат: резолв, проверка и её способность провалиться."""

from __future__ import annotations

import json

import pytest

from mlaw.citations import Citation, CitationResolver, parse_citations
from mlaw.oix import OixEntry, write_oix


@pytest.fixture
def resolver(tmp_path):
    text = "Правительство Москвы постановляет. Пункт первый. Пункт второй."
    record = {"doc_id": 42, "text": text}
    payload = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
    (tmp_path / "slice.jsonl").write_bytes(payload)
    write_oix(
        tmp_path / "slice.oix",
        [OixEntry(offset=0, length=len(payload), doc_id=42, flags=0, text_hash=1)],
    )
    with CitationResolver(tmp_path / "slice.jsonl", tmp_path / "slice.oix") as r:
        yield r, text


def test_parse_finds_all_citations():
    found = parse_citations("Так [[1:0-10]] и вот [[22:5-9]].")
    assert [c.key for c in found] == ["1:0-10", "22:5-9"]


def test_parse_ignores_malformed():
    assert parse_citations("[[abc:0-1]] [[1:0]] [1:0-1]") == []


def test_resolve_returns_exact_slice(resolver):
    r, text = resolver
    assert r.resolve(Citation(42, 0, 34)) == text[0:34]


def test_unknown_doc_id_is_rejected(resolver):
    r, _ = resolver
    check = r.check(Citation(999, 0, 5))
    assert not check.ok and "doc_id" in check.reason


def test_range_beyond_text_is_rejected(resolver):
    r, text = resolver
    check = r.check(Citation(42, 0, len(text) + 100))
    assert not check.ok and "границей" in check.reason


def test_inverted_range_is_rejected(resolver):
    r, _ = resolver
    assert not r.check(Citation(42, 20, 5)).ok


def test_verbatim_quote_passes(resolver):
    r, text = resolver
    assert r.check(Citation(42, 0, 34), text[0:30]).ok


def test_paraphrased_quote_fails(resolver):
    """Ровно то, ради чего проверка существует: перефразирование не проходит."""
    r, _ = resolver
    check = r.check(Citation(42, 0, 34), "Мэрия Москвы решила")
    assert not check.ok and "не найден" in check.reason


def test_whitespace_differences_are_tolerated(resolver):
    """Перенос строки в цитате ломать сверку не должен — это не перефразирование."""
    r, _ = resolver
    assert r.check(Citation(42, 0, 34), "Правительство   Москвы\n постановляет").ok


def test_resolvable_and_verbatim_are_reported_separately(resolver):
    """Ссылка может быть корректной при недобросовестной цитате.

    Смешение двух свойств в одно число скрыло бы, что именно сломалось.
    """
    r, _ = resolver
    report = r.verify_answer("Вот [[42:0-34]].", quotes={"42:0-34": "выдуманный текст"})
    assert report["resolvable_share"] == 1.0
    assert report["quote_verbatim_share"] == 0.0


def test_repeated_citation_counted_once(resolver):
    r, text = resolver
    report = r.verify_answer(
        "Раз [[42:0-34]], два [[42:0-34]].", quotes={"42:0-34": text[0:20]}
    )
    assert report["citations"] == 2
    assert report["unique_citations"] == 1
    assert report["quote_verbatim_share"] == 1.0


def test_answer_without_citations_has_no_share(resolver):
    r, _ = resolver
    report = r.verify_answer("Ответ без единой ссылки.")
    assert report["unique_citations"] == 0
    assert report["valid_share"] is None
