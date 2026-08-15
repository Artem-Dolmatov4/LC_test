"""Тесты инвентаризации.

Главное здесь — таблица кейсов по временной логике. Она проверяет не «сколько
получилось на банке», а то, что каждое прочтение фильтра делает ровно то, что
про него написано: сентинелы, отсутствующий `begin`, отменённый акт, редакция
в середине цепочки.
"""

from __future__ import annotations

import pytest

from mlaw.inventory import Inventory, pct, quantiles, summarise

X = "2020-06-01"  # одна из дат, на которые считает инвентаризация


def record(
    doc_id: int,
    *,
    act_id: int = 1,
    status: str = "Действует",
    begin: str | None = "2010-01-01",
    end: str | None = None,
    end_sentinel: str | None = "indefinite",
    begin_sentinel: str | None = None,
    is_current: bool = True,
    n_editions: int = 1,
    index: int = 0,
    date: str = "2010-01-01",
    text: str = "текст",
    contents: list | None = None,
) -> dict:
    out = {
        "doc_id": doc_id,
        "date": date,
        "status": status,
        "text": text,
        "text_hash": f"h{doc_id}",
        "edition": {
            "act_id": act_id,
            "n_editions": n_editions,
            "index": index,
            "is_current": is_current,
        },
        "effective_date_begin": begin,
        "effective_date_end": end,
    }
    if end_sentinel is not None:
        out["effective_date_end_sentinel"] = end_sentinel
    if begin_sentinel is not None:
        out["effective_date_begin_sentinel"] = begin_sentinel
    if contents is not None:
        out["contents"] = contents
    return out


def pit_for(rec: dict, date: str = X) -> dict:
    inv = Inventory()
    inv.add(rec, shard=0)
    return dict(inv.pit[date])


# --------------------------------------------------------------------------- #
# Таблица кейсов временной логики
# --------------------------------------------------------------------------- #


def test_indefinite_sentinel_is_an_open_interval():
    """Бессрочная редакция действует на любую дату после начала."""
    counts = pit_for(record(1, begin="2010-01-01", end=None, end_sentinel="indefinite"))
    assert counts["A_strict"] == 1
    assert counts["E_interval_any_edition"] == 1


def test_null_sentinel_is_not_an_open_interval_under_strict_reading():
    """`null` — это «неизвестно», а не «бессрочно». Строгое прочтение его не берёт."""
    counts = pit_for(record(1, begin="2010-01-01", end=None, end_sentinel="null"))
    assert counts.get("A_strict", 0) == 0
    assert counts.get("E_interval_any_edition", 0) == 0
    # Мягкое прочтение — берёт: в этом и разница между ними.
    assert counts["C_loose_end"] == 1


def test_expired_edition_is_excluded():
    counts = pit_for(record(1, begin="2005-01-01", end="2015-01-01", end_sentinel=None))
    assert counts.get("A_strict", 0) == 0


def test_edition_covering_the_date_is_included():
    counts = pit_for(record(1, begin="2018-01-01", end="2022-01-01", end_sentinel=None))
    assert counts["A_strict"] == 1


def test_edition_not_yet_started_is_excluded():
    counts = pit_for(record(1, begin="2024-01-01", end=None, end_sentinel="indefinite"))
    assert counts.get("A_strict", 0) == 0


def test_missing_begin_is_not_rescued_by_date_substitution():
    """Подстановка `date` вместо пустого `begin` не спасает.

    Такие записи отсекает условие по концу интервала, а не по началу — замер
    на банке даёт ровно ноль изменений (74 670 -> 74 670).
    """
    rec = record(1, begin=None, begin_sentinel="null", end=None, end_sentinel="null",
                 date="2010-01-01")
    counts = pit_for(rec)
    assert counts.get("A_strict", 0) == 0
    assert counts.get("B_date_substituted", 0) == 0


def test_repealed_act_is_dropped_by_status_but_kept_by_interval():
    """Ключевой кейс для исторических запросов.

    Акт, отменённый сегодня, на дату в прошлом действовал. `status` описывает
    акт сейчас, поэтому прочтение со статусом его теряет, а интервальное —
    сохраняет. Для `as_of` в прошлом правильно интервальное.
    """
    rec = record(
        1,
        status="Утратил силу или отменен",
        begin="2010-01-01",
        end="2023-01-01",
        end_sentinel=None,
    )
    counts = pit_for(rec)
    assert counts.get("A_strict", 0) == 0
    assert counts["E_interval_any_edition"] == 1


def test_historical_edition_of_a_live_act_is_found_by_interval():
    """Не текущая редакция — тоже правильный ответ, если спрашивают про прошлое."""
    rec = record(
        7,
        begin="2010-01-01",
        end="2021-01-01",
        end_sentinel=None,
        is_current=False,
        n_editions=3,
        index=0,
    )
    counts = pit_for(rec)
    assert counts["E_interval_any_edition"] == 1
    assert counts.get("A_strict_and_current", 0) == 0


def test_status_only_reading_ignores_dates_entirely():
    counts = pit_for(record(1, begin="2030-01-01", end=None, end_sentinel="indefinite"))
    assert counts["status_only"] == 1
    assert counts.get("A_strict", 0) == 0


# --------------------------------------------------------------------------- #
# Однозначность на уровне акта
# --------------------------------------------------------------------------- #


def test_non_overlapping_editions_give_exactly_one_in_force():
    """Замощение оси времени: на дату действует ровно одна редакция акта."""
    inv = Inventory()
    inv.add(record(1, act_id=5, begin="2005-01-01", end="2015-01-01",
                   end_sentinel=None, is_current=False, n_editions=2, index=0), shard=0)
    inv.add(record(2, act_id=5, begin="2015-01-02", end=None,
                   end_sentinel="indefinite", is_current=True, n_editions=2, index=1), shard=0)

    by_act = summarise(inv)["point_in_time_by_act"][X]["E"]
    assert by_act["acts_with_an_edition_in_force"] == 1
    assert by_act["acts_with_more_than_one"] == 0


def test_overlapping_editions_would_be_detected():
    """Контроль: если бы интервалы перекрывались, проверка обязана это увидеть.

    На банке таких актов ноль — но тест должен уметь их поймать, иначе он
    ничего не проверяет.
    """
    inv = Inventory()
    inv.add(record(1, act_id=5, begin="2010-01-01", end=None,
                   end_sentinel="indefinite", is_current=False, n_editions=2, index=0), shard=0)
    inv.add(record(2, act_id=5, begin="2011-01-01", end=None,
                   end_sentinel="indefinite", is_current=True, n_editions=2, index=1), shard=0)

    by_act = summarise(inv)["point_in_time_by_act"][X]["E"]
    assert by_act["acts_with_more_than_one"] == 1


# --------------------------------------------------------------------------- #
# Заполненность: ключ против значения
# --------------------------------------------------------------------------- #


def test_key_presence_and_value_presence_are_counted_separately():
    """Ключ с null-значением присутствует, но заполненным не считается.

    Документация датасета обещает, что пустые ключи опущены; для дат это
    неверно, и разрыв достигает 70 п.п.
    """
    inv = Inventory()
    inv.add(record(1, end=None, end_sentinel="indefinite"), shard=0)

    gap = summarise(inv)["date_keys_key_vs_value"]["effective_date_end"]
    assert gap["key_present_pct"] == 100.0
    assert gap["value_not_null_pct"] == 0.0
    assert gap["gap_pp"] == 100.0


def test_current_edition_share_of_records_and_of_mass_are_different():
    """Доля записей и доля текстовой массы — разные числа, и это суть находки."""
    inv = Inventory()
    inv.add(record(1, is_current=True, text="к" * 100), shard=0)
    inv.add(record(2, act_id=2, is_current=False, text="к" * 900), shard=0)

    cur = summarise(inv)["current_editions"]
    assert cur["records_pct"] == 50.0
    assert cur["chars_pct_of_mass"] == 10.0


# --------------------------------------------------------------------------- #
# Дерево contents
# --------------------------------------------------------------------------- #


def test_zero_span_nodes_are_counted_not_dropped():
    node = {"level": 1, "label": "Статья 1", "char_start": 10, "char_end": 10, "parent": None}
    inv = Inventory()
    inv.add(record(1, contents=[node]), shard=0)

    toc = summarise(inv)["contents"]
    assert toc["nodes"] == 1
    assert toc["zero_span_nodes"] == 1


def test_multiroot_document_is_detected():
    nodes = [
        {"level": 1, "label": "Раздел I", "char_start": 0, "char_end": 10, "parent": None},
        {"level": 1, "label": "Раздел II", "char_start": 10, "char_end": 20, "parent": None},
    ]
    inv = Inventory()
    inv.add(record(1, contents=nodes), shard=0)
    assert summarise(inv)["contents"]["multiroot_documents"] == 1


def test_label_first_word_is_normalised():
    nodes = [
        {"level": 1, "label": "Приложение 1", "char_start": 0, "char_end": 5, "parent": None},
        {"level": 1, "label": "Статья. 5", "char_start": 5, "char_end": 9, "parent": None},
    ]
    inv = Inventory()
    inv.add(record(1, contents=nodes), shard=0)
    heads = summarise(inv)["contents"]["label_first_word_top"]
    assert heads["приложение"] == 1
    assert heads["статья"] == 1


# --------------------------------------------------------------------------- #
# Помощники
# --------------------------------------------------------------------------- #


def test_quantiles():
    q = quantiles(list(range(1, 101)))
    assert q["min"] == 1
    assert q["max"] == 100
    assert q["p50"] == 51


def test_pct_handles_zero_denominator():
    assert pct(0, 0) == 0.0
    assert pct(1, 3) == 33.33


# --------------------------------------------------------------------------- #
# Регрессия на настоящих данных
# --------------------------------------------------------------------------- #

_ARCHIVE = __import__("pathlib").Path(__file__).resolve().parents[1] / "MLAW_dataset.tar.zst"


@pytest.mark.slow
@pytest.mark.skipif(not _ARCHIVE.exists(), reason="настоящий архив недоступен")
def test_inventory_reproduces_measured_numbers_on_shard_0000():
    """Контрольные точки по шарду 0000 — собственный замер, не документация.

    Если эти числа поехали, значит поехал код инвентаризации, а не корпус.
    """
    from mlaw.stream import open_corpus

    inv = Inventory()
    for rec in open_corpus(str(_ARCHIVE)).iter_lines(shards=[0]):
        inv.add(rec.json(), rec.shard)

    report = summarise(inv)

    assert report["records"] == 9523
    assert report["total_chars"] == 256_951_088
    assert report["stubs"] == 0
    assert report["current_editions"]["records"] == 7539
    assert report["contents"]["documents"] == 4287
    assert report["contents"]["labels_with_u0015"] == 0

    # Внутри одного шарда «ровно один is_current на акт» массово НЕ выполняется:
    # цепочки редакций разрезаны между шардами. Это свойство раскладки.
    by_count = report["editions"]["acts_by_current_count"]
    assert by_count.get(0, 0) > 1000, "ожидались акты без текущей редакции внутри шарда"
