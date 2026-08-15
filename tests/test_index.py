"""Тесты индексации: перевод дат в payload и кэш векторов."""

from __future__ import annotations

from mlaw.index import INDEFINITE, EmbeddingCache, date_ordinal, end_ordinal


def test_date_ordinal_is_comparable_as_integer():
    assert date_ordinal("2010-06-01") == 20100601
    assert date_ordinal("2026-08-14") == 20260814
    # Порядок сохраняется — на этом держится диапазонный фильтр.
    assert date_ordinal("2009-12-31") < date_ordinal("2010-01-01")


def test_date_ordinal_invents_nothing():
    assert date_ordinal(None) is None
    assert date_ordinal("") is None
    assert date_ordinal("2010") is None
    assert date_ordinal("не дата!!") is None


def test_indefinite_sentinel_becomes_an_open_end():
    """«Бессрочно» обязано участвовать в сравнении, а не выпадать из фильтра."""
    assert end_ordinal(None, "indefinite") == INDEFINITE
    assert end_ordinal(None, "indefinite") > date_ordinal("2999-12-31")


def test_null_sentinel_is_not_indefinite():
    """`null` — «неизвестно». Смешение с «бессрочно» меняет выдачу на десятки тысяч записей."""
    assert end_ordinal(None, "null") is None
    assert end_ordinal(None, None) is None


def test_real_end_date_wins_over_missing_sentinel():
    assert end_ordinal("2017-03-06", None) == 20170306


def test_cache_is_resumable(tmp_path):
    path = tmp_path / "vectors.jsonl"
    first = EmbeddingCache(path)
    first.append([("a:1-2", [0.1, 0.2]), ("b:3-4", [0.3, 0.4])])

    reopened = EmbeddingCache(path)
    assert "a:1-2" in reopened
    assert "b:3-4" in reopened
    assert "c:5-6" not in reopened
    assert len(reopened) == 2
    assert dict(reopened.iter_vectors())["a:1-2"] == [0.1, 0.2]


def test_cache_survives_a_truncated_line(tmp_path):
    """Прогон, убитый посреди записи, не должен ронять следующий запуск."""
    path = tmp_path / "vectors.jsonl"
    EmbeddingCache(path).append([("a:1-2", [0.1])])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"chunk_id": "broken"')  # оборванная строка

    assert "a:1-2" in EmbeddingCache(path)
