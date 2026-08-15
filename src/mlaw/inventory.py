"""Шаг 1 — инвентаризация банка сплошным проходом.

Считается по всем 266 657 записям, а не по выборке: полный проход с разбором
JSON стоит ~40 с, и экономить тут не на чем. Результат — `reports/inventory.json`
плюс читаемая сводка на stdout.

Три вещи, которые этот модуль делает не так, как сделал бы наивный счётчик,
и каждая из них — следствие измеренного свойства данных:

1. **Заполненность считается по не-null значениям, а не по наличию ключа.**
   Документация датасета обещает «ключ с пустым значением опущен», но для дат
   это неверно: ``effective_date_end`` присутствует у 99.96 % записей, а
   не-null — примерно у 30 %. Меряется и то и другое, расхождение показывается.

2. **Точка отсчёта времени — не одна.** «Действует» по статусу, `is_current`
   и попадание в интервал действия — три разных признака, расходящихся на
   заметной доле записей. Считаются все прочтения, а выбор объясняется в отчёте.

3. **Цепочка редакций живёт поперёк шардов.** Проверка «ровно один is_current
   на акт» имеет смысл только на банке целиком; внутри шарда она массово
   не выполняется, и это свойство раскладки, а не дефект данных.

    python -m mlaw.inventory --archive MLAW_dataset.tar.zst
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlaw.stream import open_corpus

# Даты, на которые считается point-in-time. Историческая глубина нужна, чтобы
# проверить фильтр не только на «сегодня»: срез датирован июлем 2026.
PIT_DATES = ("2010-06-01", "2015-06-01", "2020-06-01", "2026-08-14")

# Ключи-даты, для которых важно различать «ключа нет» и «ключ есть, значение null».
DATE_KEYS = ("date", "edition_date", "effective_date_begin", "effective_date_end")


def quantiles(values: list[int], points=(0.1, 0.5, 0.9, 0.99)) -> dict[str, int]:
    ordered = sorted(values)
    out = {}
    for p in points:
        idx = min(len(ordered) - 1, int(p * len(ordered)))
        out[f"p{int(p * 100)}"] = ordered[idx]
    out["max"] = ordered[-1]
    out["min"] = ordered[0]
    return out


def pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


@dataclass
class Inventory:
    """Аккумулятор одного прохода."""

    records: int = 0
    stubs: int = 0
    total_chars: int = 0

    key_present: Counter = field(default_factory=Counter)
    value_not_null: Counter = field(default_factory=Counter)
    extra_present: Counter = field(default_factory=Counter)

    lengths_all: list[int] = field(default_factory=list)
    lengths_current: list[int] = field(default_factory=list)
    chars_current: int = 0

    status: Counter = field(default_factory=Counter)
    doc_type: Counter = field(default_factory=Counter)
    issuing_body: Counter = field(default_factory=Counter)
    type_inf: Counter = field(default_factory=Counter)
    multivalued: Counter = field(default_factory=Counter)

    sentinel_end: Counter = field(default_factory=Counter)
    sentinel_begin: Counter = field(default_factory=Counter)
    sentinel_edition: Counter = field(default_factory=Counter)

    # act_id -> список (doc_id, is_current, n_editions, index, shard)
    acts: dict[int, list[tuple]] = field(default_factory=lambda: defaultdict(list))

    text_hashes: Counter = field(default_factory=Counter)

    docs_with_toc: int = 0
    toc_nodes: int = 0
    toc_zero_span: int = 0
    toc_multiroot: int = 0
    toc_levels: Counter = field(default_factory=Counter)
    toc_label_head: Counter = field(default_factory=Counter)
    toc_u0015: int = 0

    # PIT: дата -> прочтение -> счётчик
    pit: dict = field(default_factory=lambda: defaultdict(Counter))
    # PIT на уровне актов: (дата, прочтение) -> act_id -> сколько редакций в силе.
    # Нужен для проверки, ради которой весь темпоральный фильтр и существует:
    # на любую дату у акта должна быть в силе не больше одной редакции.
    pit_acts: dict = field(default_factory=lambda: defaultdict(Counter))

    def add(self, record: dict, shard: int) -> None:
        self.records += 1
        text = record.get("text") or ""
        length = len(text)
        self.total_chars += length
        self.lengths_all.append(length)

        if record.get("is_stub"):
            self.stubs += 1

        for key, value in record.items():
            self.key_present[key] += 1
            if value is not None and value != "" and value != []:
                self.value_not_null[key] += 1
        for key in record.get("extra") or {}:
            self.extra_present[key] += 1

        edition = record.get("edition") or {}
        is_current = bool(edition.get("is_current"))
        if is_current:
            self.lengths_current.append(length)
            self.chars_current += length
        self.acts[edition.get("act_id")].append(
            (
                record.get("doc_id"),
                is_current,
                edition.get("n_editions"),
                edition.get("index"),
                shard,
            )
        )

        self.status[record.get("status")] += 1
        self.doc_type[record.get("doc_type")] += 1
        self.issuing_body[record.get("issuing_body_short")] += 1
        self.type_inf[(record.get("extra") or {}).get("COMMON_TYPE_INF")] += 1
        for key in ("doc_type_all", "rubrics_code_all", "source_all", "number_all"):
            if key in record:
                self.multivalued[key] += 1

        self.sentinel_end[record.get("effective_date_end_sentinel")] += 1
        self.sentinel_begin[record.get("effective_date_begin_sentinel")] += 1
        self.sentinel_edition[record.get("edition_date_sentinel")] += 1

        self.text_hashes[record.get("text_hash")] += 1

        self._add_pit(record, is_current, edition.get("act_id"))
        self._add_toc(record)

    def _add_pit(self, record: dict, is_current: bool, act_id: int | None) -> None:
        """Считает прочтения фильтра «действовало на дату X».

        Считается по **всем** редакциям, а не только по текущим: именно
        историческая редакция и есть правильный ответ на запрос с датой
        в прошлом.
        """
        status = record.get("status")
        begin = record.get("effective_date_begin")
        end = record.get("effective_date_end")
        end_sentinel = record.get("effective_date_end_sentinel")

        acting = status == "Действует"
        begin_substituted = begin or record.get("date")
        open_strict = end_sentinel == "indefinite"
        open_loose = open_strict or end_sentinel == "null" or end is None

        for date in PIT_DATES:
            begins_ok = bool(begin and begin <= date)
            begins_subst = bool(begin_substituted and begin_substituted <= date)
            ends_strict = bool(open_strict or (end and end >= date))
            ends_loose = bool(open_loose or (end and end >= date))

            bucket = self.pit[date]
            if acting and begins_ok and ends_strict:
                bucket["A_strict"] += 1
                if is_current:
                    bucket["A_strict_and_current"] += 1
            if acting and begins_subst and ends_strict:
                bucket["B_date_substituted"] += 1
            if acting and begins_ok and ends_loose:
                bucket["C_loose_end"] += 1
            if begins_ok and ends_loose:
                bucket["D_no_status_condition"] += 1
            if acting:
                bucket["status_only"] += 1
            # Интервальное прочтение по всем редакциям, без учёта статуса акта:
            # именно оно отвечает на «какая редакция действовала на дату X».
            if begins_ok and ends_strict:
                bucket["E_interval_any_edition"] += 1
                self.pit_acts[(date, "E")][act_id] += 1
            if acting and begins_ok and ends_strict:
                self.pit_acts[(date, "A")][act_id] += 1

    def _add_toc(self, record: dict) -> None:
        contents = record.get("contents")
        if not contents:
            return
        self.docs_with_toc += 1
        self.toc_nodes += len(contents)
        if sum(1 for node in contents if node.get("parent") is None) > 1:
            self.toc_multiroot += 1
        for node in contents:
            if node["char_start"] == node["char_end"]:
                self.toc_zero_span += 1
            self.toc_levels[node["level"]] += 1
            label = node.get("label") or ""
            if "\x15" in label:
                self.toc_u0015 += 1
            head = label.strip().split()
            if head:
                self.toc_label_head[head[0].rstrip(".").lower()] += 1


def summarise(inv: Inventory) -> dict[str, Any]:
    """Сводит аккумулятор в отчёт."""
    n = inv.records

    # --- цепочки редакций ---------------------------------------------- #
    chain_lengths = [len(v) for v in inv.acts.values()]
    multi = {a: v for a, v in inv.acts.items() if len(v) > 1}
    current_per_act = Counter(sum(1 for x in v if x[1]) for v in inv.acts.values())
    max_docid_wrong = sum(1 for v in multi.values() if not max(v, key=lambda x: x[0])[1])
    index_wrong = sum(1 for v in multi.values() for x in v if x[1] and x[3] != x[2] - 1)
    n_editions_mismatch = sum(1 for v in inv.acts.values() if v[0][2] != len(v))
    split_across_shards = sum(1 for v in multi.values() if len({x[4] for x in v}) > 1)
    shards_per_act = Counter(len({x[4] for x in v}) for v in multi.values())

    multi_lengths = sorted(len(v) for v in multi.values())

    # --- заполненность: ключ против значения ---------------------------- #
    presence = {}
    for key in sorted(inv.key_present, key=lambda k: -inv.key_present[k]):
        presence[key] = {
            "key_present_pct": pct(inv.key_present[key], n),
            "value_not_null_pct": pct(inv.value_not_null[key], n),
        }
    date_key_gap = {
        key: {
            "key_present_pct": pct(inv.key_present.get(key, 0), n),
            "value_not_null_pct": pct(inv.value_not_null.get(key, 0), n),
            "gap_pp": round(
                pct(inv.key_present.get(key, 0), n) - pct(inv.value_not_null.get(key, 0), n), 2
            ),
        }
        for key in DATE_KEYS
    }

    duplicates = n - len(inv.text_hashes)

    return {
        "records": n,
        "stubs": inv.stubs,
        "total_chars": inv.total_chars,
        "current_editions": {
            "records": len(inv.lengths_current),
            "records_pct": pct(len(inv.lengths_current), n),
            "chars": inv.chars_current,
            "chars_pct_of_mass": pct(inv.chars_current, inv.total_chars),
        },
        "text_length": {
            "all": quantiles(inv.lengths_all),
            "current_only": quantiles(inv.lengths_current),
        },
        "field_presence": presence,
        "date_keys_key_vs_value": date_key_gap,
        "extra_keys": {
            k: pct(v, n) for k, v in sorted(inv.extra_present.items(), key=lambda kv: -kv[1])
        },
        "multivalued_pct": {k: pct(v, n) for k, v in inv.multivalued.items()},
        "status": dict(inv.status.most_common()),
        "doc_type_top": dict(inv.doc_type.most_common(12)),
        "issuing_body_top": dict(inv.issuing_body.most_common(10)),
        "type_inf": dict(inv.type_inf.most_common(8)),
        "sentinels": {
            "effective_date_end": dict(inv.sentinel_end.most_common()),
            "effective_date_begin": dict(inv.sentinel_begin.most_common()),
            "edition_date": dict(inv.sentinel_edition.most_common()),
        },
        "editions": {
            "acts": len(inv.acts),
            "multi_edition_acts": len(multi),
            "records_in_multi": sum(len(v) for v in multi.values()),
            "acts_by_current_count": dict(sorted(current_per_act.items())),
            "max_docid_is_not_current": max_docid_wrong,
            "current_index_ne_n_editions_minus_1": index_wrong,
            "n_editions_ne_chain_length": n_editions_mismatch,
            "chain_length_all": quantiles(chain_lengths),
            # Многоредакционных актов может не быть вовсе — например, на срезе
            # из одного акта. Пустой список тут законный вход, а не аномалия.
            "multi_chain_median": multi_lengths[len(multi_lengths) // 2] if multi_lengths else None,
            "multi_chain_p90": (
                multi_lengths[int(0.9 * len(multi_lengths))] if multi_lengths else None
            ),
            "multi_chain_max": multi_lengths[-1] if multi_lengths else None,
            "split_across_shards": split_across_shards,
            "split_across_shards_pct": pct(split_across_shards, len(multi)),
            "shards_per_multi_act": dict(sorted(shards_per_act.items())),
        },
        "duplicates_by_text_hash": duplicates,
        "contents": {
            "documents": inv.docs_with_toc,
            "documents_pct": pct(inv.docs_with_toc, n),
            "nodes": inv.toc_nodes,
            "zero_span_nodes": inv.toc_zero_span,
            "zero_span_pct": pct(inv.toc_zero_span, inv.toc_nodes),
            "multiroot_documents": inv.toc_multiroot,
            "multiroot_pct": pct(inv.toc_multiroot, inv.docs_with_toc),
            "levels": dict(sorted(inv.toc_levels.items())),
            "label_first_word_top": dict(inv.toc_label_head.most_common(15)),
            "labels_with_u0015": inv.toc_u0015,
        },
        "point_in_time": {date: dict(counts) for date, counts in sorted(inv.pit.items())},
        "point_in_time_by_act": _pit_by_act(inv),
    }


def _pit_by_act(inv: Inventory) -> dict:
    """Сколько редакций одного акта оказываются в силе одновременно.

    Это и есть проверка пригодности темпорального фильтра: если на какую-то
    дату у акта в силе две редакции, выдача становится неоднозначной, и дедуп
    по акту будет выбирать произвольно.
    """
    out: dict = {}
    for (date, reading), counter in sorted(inv.pit_acts.items()):
        distribution = Counter(counter.values())
        out.setdefault(date, {})[reading] = {
            "acts_with_an_edition_in_force": len(counter),
            "editions_in_force_per_act": dict(sorted(distribution.items())),
            "acts_with_more_than_one": sum(v for k, v in distribution.items() if k > 1),
        }
    return out


def run(archive: str, out: Path) -> dict:
    inv = Inventory()
    started = time.time()
    print(f"Сплошной проход по банку: {archive}")
    for record in open_corpus(archive).iter_lines():
        inv.add(record.json(), record.shard)
        if inv.records % 50_000 == 0:
            print(f"  {inv.records:>7} записей, {time.time() - started:5.1f} с")

    elapsed = time.time() - started
    report = summarise(inv)
    report["_meta"] = {
        "archive": archive,
        "seconds": round(elapsed, 1),
        "records_per_sec": round(inv.records / elapsed),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_summary(r: dict) -> None:
    cur = r["current_editions"]
    ed = r["editions"]
    toc = r["contents"]

    print(f"\n{'=' * 66}")
    print(f"  Записей {r['records']:,} · знаков {r['total_chars']:,} · стабов {r['stubs']}"
          .replace(",", " "))
    print(f"  Действующих редакций: {cur['records']:,} ({cur['records_pct']} %) — "
          f"но лишь {cur['chars_pct_of_mass']} % текстовой массы".replace(",", " "))
    print()
    print("  Длина текста, знаков:")
    for label, key in (("все записи", "all"), ("только is_current", "current_only")):
        q = r["text_length"][key]
        print(f"    {label:<18} p10 {q['p10']:>7} · med {q['p50']:>7} · "
              f"p90 {q['p90']:>8} · p99 {q['p99']:>9} · max {q['max']:>10}")
    print()
    print(f"  Актов {ed['acts']:,}, многоредакционных {ed['multi_edition_acts']:,}"
          .replace(",", " "))
    print(f"    is_current на акт: {ed['acts_by_current_count']}")
    print(f"    max(doc_id) не текущая редакция: {ed['max_docid_is_not_current']}")
    print(f"    цепочка разрезана между шардами: {ed['split_across_shards']:,} "
          f"({ed['split_across_shards_pct']} %)".replace(",", " "))
    print()
    print(f"  contents: {toc['documents']:,} док. ({toc['documents_pct']} %), "
          f"узлов {toc['nodes']:,}, нулевого размаха {toc['zero_span_nodes']:,} "
          f"({toc['zero_span_pct']} %)".replace(",", " "))
    print(f"    многокорневых: {toc['multiroot_pct']} % · "
          f"меток с U+0015: {toc['labels_with_u0015']}")
    print()
    print("  Ключ есть против значение не-null:")
    for key, gap in r["date_keys_key_vs_value"].items():
        print(f"    {key:<24} ключ {gap['key_present_pct']:>6} % · "
              f"значение {gap['value_not_null_pct']:>6} % · разрыв {gap['gap_pp']:>6} п.п.")
    print()
    print("  Point-in-time, записей по прочтениям:")
    for date, counts in r["point_in_time"].items():
        print(f"    {date}: A(со статусом) {counts.get('A_strict', 0):>7} · "
              f"E(только интервал) {counts.get('E_interval_any_edition', 0):>7}")
    print()
    print("  Point-in-time на уровне актов (редакций в силе на акт):")
    for date, readings in r["point_in_time_by_act"].items():
        for reading, data in readings.items():
            print(f"    {date} [{reading}]: актов {data['acts_with_an_edition_in_force']:>7}, "
                  f"из них с двумя и более редакциями сразу: "
                  f"{data['acts_with_more_than_one']}")
    print(f"{'=' * 66}")
    print(f"  {r['_meta']['seconds']} с, {r['_meta']['records_per_sec']:,} записей/с"
          .replace(",", " "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Инвентаризация банка MLAW")
    parser.add_argument("--archive", default="MLAW_dataset.tar.zst")
    parser.add_argument("--out", type=Path, default=Path("reports/inventory.json"))
    args = parser.parse_args()

    report = run(args.archive, args.out)
    print_summary(report)
    print(f"Записано в {args.out}")


if __name__ == "__main__":
    main()
