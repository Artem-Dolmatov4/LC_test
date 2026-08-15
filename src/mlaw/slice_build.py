"""Шаг 2 — построение среза по актам.

Срез строится **по актам, а не по шардам**, и в него идёт **вся цепочка
редакций** выбранного акта. Оба решения продиктованы замерами:

* 92.2 % многоредакционных актов имеют цепочку, разрезанную между шардами
  (до всех 28 сразу). Срез по шардам молча теряет актуальный текст 14–17 %
  затронутых актов — это ровно тот случай «молча выброшенной части корпуса»,
  который задание называет красным флагом.
* Без исторических редакций point-in-time поиск не на чем проверять: на запрос
  с датой в прошлом правильный ответ — историческая редакция, и если её нет
  в индексе, фильтр нечего фильтровать.

Цена решения известна заранее: средний акт со всеми редакциями — 42 264 знака
против 14 544 у одной действующей, то есть в 2.9 раза дороже.

Результат — `data/slice.jsonl`, собственный сайдкар `data/slice.oix` и
`data/slice_manifest.json` с полным учётом отсева.

    python -m mlaw.slice_build --acts 800 --seed 20260815
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mlaw.oix import (
    FLAG_HAS_CONTENTS,
    FLAG_IS_CURRENT,
    FLAG_IS_STUB,
    OixEntry,
    read_oix,
    verify_oix,
    write_oix,
)
from mlaw.stream import open_corpus

# Пороги, на которых показывается, во что обошёлся бы отсев длинных актов.
CAP_PROBES = (500_000, 2_000_000, 8_000_000)


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Лёгкая карточка записи — всё, что нужно для отбора, без текста."""

    doc_id: int
    act_id: int
    is_current: bool
    is_stub: bool
    has_contents: bool
    chars: int
    shard: int


@dataclass
class Accounting:
    """Учёт отсева: и документами, и текстовой массой.

    Задание требует на каждом отсеве сообщать обе величины. Считать только
    документы недостаточно — 22 % записей держат 66 % текста.
    """

    steps: list[dict] = field(default_factory=list)

    def add(self, name: str, docs: int, chars: int, total_docs: int, total_chars: int) -> None:
        self.steps.append(
            {
                "step": name,
                "documents": docs,
                "documents_pct": round(100 * docs / total_docs, 3) if total_docs else 0.0,
                "chars": chars,
                "chars_pct": round(100 * chars / total_chars, 3) if total_chars else 0.0,
            }
        )


def scan(archive: str) -> tuple[list[DocMeta], dict[int, list[DocMeta]]]:
    """Первый проход: карточки всех записей и группировка по актам."""
    metas: list[DocMeta] = []
    by_act: dict[int, list[DocMeta]] = defaultdict(list)
    started = time.time()
    for rec in open_corpus(archive).iter_lines():
        d = rec.json()
        edition = d.get("edition") or {}
        meta = DocMeta(
            doc_id=d["doc_id"],
            act_id=edition.get("act_id"),
            is_current=bool(edition.get("is_current")),
            is_stub=bool(d.get("is_stub")),
            has_contents=bool(d.get("contents")),
            chars=len(d.get("text") or ""),
            shard=rec.shard,
        )
        metas.append(meta)
        by_act[meta.act_id].append(meta)
        if len(metas) % 100_000 == 0:
            print(f"  просмотрено {len(metas):>7}, {time.time() - started:5.1f} с")
    print(f"  проход 1: {len(metas)} записей, {len(by_act)} актов, {time.time() - started:.1f} с")
    return metas, by_act


def choose_acts(by_act: dict[int, list[DocMeta]], n: int, seed: int) -> list[int]:
    """Случайная выборка актов с фиксированным зерном.

    Сортировка перед выборкой обязательна: порядок ключей словаря зависит от
    порядка вставки, а значит от порядка чтения архива. Без сортировки та же
    команда с тем же зерном дала бы другой срез.
    """
    act_ids = sorted(by_act)
    if n >= len(act_ids):
        return act_ids
    return sorted(random.Random(seed).sample(act_ids, n))


def build(
    archive: str,
    out_dir: Path,
    acts: int,
    seed: int,
    max_act_chars: int | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Проход 1: карточки всех записей ({archive})")
    metas, by_act = scan(archive)

    total_docs = len(metas)
    total_chars = sum(m.chars for m in metas)
    accounting = Accounting()
    accounting.add("банк целиком", total_docs, total_chars, total_docs, total_chars)

    # --- отсев стабов --------------------------------------------------- #
    stub_docs = [m for m in metas if m.is_stub]
    stub_acts = {m.act_id for m in stub_docs}
    accounting.add(
        "исключены стабы (is_stub)",
        len(stub_docs),
        sum(m.chars for m in stub_docs),
        total_docs,
        total_chars,
    )

    eligible = {a: v for a, v in by_act.items() if a not in stub_acts}

    # --- необязательный потолок по длине акта --------------------------- #
    # Никогда не применяется молча: даже когда потолка нет, показывается,
    # во что он обошёлся бы.
    act_chars = {a: sum(m.chars for m in v) for a, v in eligible.items()}
    cap_preview = []
    for probe in CAP_PROBES:
        over = [a for a, c in act_chars.items() if c > probe]
        cap_preview.append(
            {
                "threshold_chars": probe,
                "acts_over": len(over),
                "acts_over_pct": round(100 * len(over) / len(eligible), 3),
                "chars_in_them_pct": round(
                    100 * sum(act_chars[a] for a in over) / sum(act_chars.values()), 2
                ),
            }
        )

    if max_act_chars is not None:
        dropped = [a for a, c in act_chars.items() if c > max_act_chars]
        dropped_docs = sum(len(eligible[a]) for a in dropped)
        dropped_chars = sum(act_chars[a] for a in dropped)
        accounting.add(
            f"исключены акты длиннее {max_act_chars} знаков",
            dropped_docs,
            dropped_chars,
            total_docs,
            total_chars,
        )
        eligible = {a: v for a, v in eligible.items() if a not in set(dropped)}

    # --- выборка актов --------------------------------------------------- #
    chosen = choose_acts(eligible, acts, seed)
    allow: dict[int, DocMeta] = {}
    for act_id in chosen:
        for meta in eligible[act_id]:
            allow[meta.doc_id] = meta

    slice_chars = sum(m.chars for m in allow.values())
    accounting.add(
        f"выборка {len(chosen)} актов с полными цепочками",
        len(allow),
        slice_chars,
        total_docs,
        total_chars,
    )

    slice_profile = _act_size_profile(chosen, eligible)

    # --- проход 2: запись среза ------------------------------------------ #
    slice_path = out_dir / "slice.jsonl"
    oix_path = out_dir / "slice.oix"
    print(f"Проход 2: пишу {len(allow)} записей в {slice_path}")

    entries: list[OixEntry] = []
    offset = 0
    written = 0
    started = time.time()
    with open(slice_path, "wb") as sink:
        for rec in open_corpus(archive).iter_lines():
            meta = allow.get(_peek_doc_id(rec.line))
            if meta is None:
                continue
            line = rec.line + b"\n"
            sink.write(line)
            flags = 0
            if meta.is_stub:
                flags |= FLAG_IS_STUB
            if meta.has_contents:
                flags |= FLAG_HAS_CONTENTS
            if meta.is_current:
                flags |= FLAG_IS_CURRENT
            entries.append(
                OixEntry(
                    offset=offset,
                    length=len(line),
                    doc_id=meta.doc_id,
                    flags=flags,
                    text_hash=_text_hash(rec.line),
                )
            )
            offset += len(line)
            written += 1
    write_oix(oix_path, entries)
    print(f"  проход 2: {written} записей, {time.time() - started:.1f} с")

    # --- проверки --------------------------------------------------------- #
    checks = verify(slice_path, oix_path, chosen, by_act)

    manifest = {
        "seed": seed,
        "acts_requested": acts,
        "acts_selected": len(chosen),
        "documents": written,
        "chars": slice_chars,
        "max_act_chars": max_act_chars,
        "mean_chars_per_act": round(slice_chars / len(chosen)) if chosen else 0,
        "accounting": accounting.steps,
        "length_cap_preview": cap_preview,
        "slice_profile": slice_profile,
        "checks": checks,
        "files": {"slice": str(slice_path), "sidecar": str(oix_path)},
    }
    (out_dir / "slice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _act_size_profile(chosen: list[int], eligible: dict[int, list[DocMeta]]) -> dict:
    """Распределение массы по актам среза и её концентрация.

    Существует потому, что средняя длина акта здесь бессмысленна: масса
    сосредоточена в единицах актов, и знать это надо до того, как считать
    эмбеддинги, а не после.
    """
    sizes = sorted((sum(m.chars for m in eligible[a]) for a in chosen))
    total = sum(sizes)
    ranked = sorted(
        ((a, sum(m.chars for m in eligible[a]), len(eligible[a])) for a in chosen),
        key=lambda t: -t[1],
    )

    def at(p: float) -> int:
        return sizes[min(len(sizes) - 1, int(p * len(sizes)))]

    def head_share(k: int) -> float:
        return round(100 * sum(s for _, s, _ in ranked[:k]) / total, 2) if total else 0.0

    return {
        "act_chars": {
            "p50": at(0.5),
            "p90": at(0.9),
            "p99": at(0.99),
            "max": sizes[-1] if sizes else 0,
            "mean": round(total / len(sizes)) if sizes else 0,
        },
        "max_to_median_ratio": round(sizes[-1] / at(0.5)) if sizes and at(0.5) else None,
        "mass_share_of_top_1_act": head_share(1),
        "mass_share_of_top_8_acts": head_share(8),
        "largest_acts": [
            {"act_id": a, "chars": c, "editions": n, "mass_pct": round(100 * c / total, 2)}
            for a, c, n in ranked[:5]
        ],
        "cumulative_mass_under_cap": [
            {
                "cap_chars": cap,
                "acts_kept": sum(1 for s in sizes if s <= cap),
                "acts_kept_pct": round(100 * sum(1 for s in sizes if s <= cap) / len(sizes), 1),
                "mass_kept_pct": round(
                    100 * sum(s for s in sizes if s <= cap) / total, 1
                ),
            }
            for cap in (200_000, 500_000, 1_000_000)
        ],
    }


def _peek_doc_id(line: bytes) -> int:
    """Достаёт doc_id без полного разбора JSON — второй проход этим живёт.

    Разбор всей записи ради одного числа стоит около трети времени прохода,
    поэтому здесь дешёвый поиск по подстроке. Любая неожиданность в раскладке
    (пробел после двоеточия, перенос ключа вглубь) откатывает на честный
    `json.loads`, а не портит выборку молча.
    """
    start = line.find(b'"doc_id":')
    if start >= 0:
        cursor = start + len(b'"doc_id":')
        while cursor < len(line) and line[cursor : cursor + 1] == b" ":
            cursor += 1
        end = cursor
        while end < len(line) and line[end : end + 1].isdigit():
            end += 1
        if end > cursor:
            return int(line[cursor:end])
    return json.loads(line)["doc_id"]


def _text_hash(line: bytes) -> int:
    """Берёт text_hash записи как есть — он уже посчитан издателем."""
    try:
        value = json.loads(line).get("text_hash")
        return int(value, 16) if value else 0
    except (ValueError, TypeError):
        return 0


def verify(
    slice_path: Path,
    oix_path: Path,
    chosen: list[int],
    by_act: dict[int, list[DocMeta]],
) -> dict:
    """Проверяет срез теми же инвариантами, что и оригинальные шарды."""
    entries = read_oix(oix_path)
    problems = verify_oix(entries, shard_size=slice_path.stat().st_size)

    chosen_set = set(chosen)
    in_slice = defaultdict(list)
    for meta in (m for act in chosen_set for m in by_act[act]):
        in_slice[meta.act_id].append(meta)

    current_counts = {a: sum(1 for m in v if m.is_current) for a, v in in_slice.items()}
    chain_complete = sum(1 for a, v in in_slice.items() if len(v) == len(by_act[a]))

    return {
        "sidecar_violations": [str(p) for p in problems],
        "acts_with_exactly_one_current": sum(1 for c in current_counts.values() if c == 1),
        "acts_with_wrong_current_count": sum(1 for c in current_counts.values() if c != 1),
        "acts_with_complete_chain": chain_complete,
        "acts_with_truncated_chain": len(in_slice) - chain_complete,
        "records_in_sidecar": len(entries),
        "current_in_sidecar": sum(1 for e in entries if e.is_current),
        "with_contents_in_sidecar": sum(1 for e in entries if e.has_contents),
        "stubs_in_sidecar": sum(1 for e in entries if e.is_stub),
    }


def print_summary(m: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Срез: {m['acts_selected']} актов, {m['documents']} записей, "
          f"{m['chars']:,} знаков".replace(",", " "))
    print(f"  Зерно {m['seed']} · в среднем {m['mean_chars_per_act']:,} знаков на акт"
          .replace(",", " "))
    print()
    print("  Учёт отсева (документы и текстовая масса):")
    for step in m["accounting"]:
        print(f"    {step['step']:<46} {step['documents']:>8} док "
              f"({step['documents_pct']:>6} %) · {step['chars_pct']:>6} % массы")
    print()
    p = m["slice_profile"]
    a = p["act_chars"]
    print("  Масса по актам среза, знаков:")
    print(f"    p50 {a['p50']:,} · p90 {a['p90']:,} · p99 {a['p99']:,} · max {a['max']:,}"
          .replace(",", " "))
    print(f"    max/median = {p['max_to_median_ratio']:,}x — почти пять порядков"
          .replace(",", " "))
    print(f"    крупнейший акт держит {p['mass_share_of_top_1_act']} % массы среза, "
          f"восемь крупнейших — {p['mass_share_of_top_8_acts']} %")
    for row in p["largest_acts"][:3]:
        print(f"      act {row['act_id']:>7}: {row['chars']:>12,} знаков, "
              f"{row['editions']:>2} ред. — {row['mass_pct']:>5} %".replace(",", " "))
    print()
    print("  Сколько осталось бы под потолком по длине акта (потолок НЕ применён):")
    for row in p["cumulative_mass_under_cap"]:
        print(f"    <= {row['cap_chars']:>9,} знаков: актов {row['acts_kept']:>4} "
              f"({row['acts_kept_pct']:>5} %), массы {row['mass_kept_pct']:>5} %"
              .replace(",", " "))
    print()
    print("  Для сравнения, по банку целиком:")
    for probe in m["length_cap_preview"]:
        print(f"    длиннее {probe['threshold_chars']:>9,} знаков: "
              f"{probe['acts_over']:>6} актов ({probe['acts_over_pct']:>5} %), "
              f"в них {probe['chars_in_them_pct']:>5} % массы БАНКА".replace(",", " "))
    print()
    c = m["checks"]
    print("  Проверки:")
    print(f"    нарушений сайдкара:              {len(c['sidecar_violations'])}")
    print(f"    актов с ровно одной is_current:  {c['acts_with_exactly_one_current']}")
    print(f"    актов с неверным числом current: {c['acts_with_wrong_current_count']}")
    print(f"    актов с полной цепочкой:         {c['acts_with_complete_chain']}")
    print(f"    актов с обрезанной цепочкой:     {c['acts_with_truncated_chain']}")
    print(f"    в сайдкаре: {c['records_in_sidecar']} записей, "
          f"{c['current_in_sidecar']} current, {c['with_contents_in_sidecar']} с contents, "
          f"{c['stubs_in_sidecar']} стабов")
    print(f"{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Срез корпуса по актам")
    parser.add_argument("--archive", default="MLAW_dataset.tar.zst")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--acts", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--max-act-chars",
        type=int,
        default=None,
        help="потолок суммарной длины акта; по умолчанию нет — отсев только явный",
    )
    args = parser.parse_args()

    manifest = build(args.archive, args.out, args.acts, args.seed, args.max_act_chars)
    print_summary(manifest)
    print(f"Манифест: {args.out / 'slice_manifest.json'}")


if __name__ == "__main__":
    main()
