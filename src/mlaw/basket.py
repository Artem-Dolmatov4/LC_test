"""Шаг 5 (часть 1) — сборка корзины запросов с эталонами.

Корзина состоит из четырёх типов, и они намеренно разные: каждый меряет то,
на что остальные слепы. Тип пишется в запись, и метрика считается по типам
раздельно — иначе лёгкая синтетика замаскирует провал на сложном.

**metadata** — реквизиты акта: «125-ПП», «приказ Мосгорнаследия N 63».
Эталон известен точно и без разметки. На этих запросах обязана выигрывать
лексическая нога; если не выигрывает — она не нужна, и это тоже результат.

**temporal** — пары «один акт, две даты». Строятся из 73 многоредакционных
актов среза, у которых интервалы действия заполнены. Эталон — та редакция,
что действовала на дату запроса. Это единственный тип, который ловит
сломанный point-in-time фильтр: обычная метрика на него слепа, потому что
акт-то найден правильный.

**synthetic** — вопрос, сочинённый моделью по известному фрагменту. Эталон —
сам фрагмент, то есть единственный тип с разметкой на уровне чанка. Смещён
лексически: вопрос неизбежно переиспользует слова исходника, и задача
оказывается легче настоящей. Это идёт в раздел ограничений, а не
замалчивается.

**manual** — написанные руками правовые вопросы. Дорого и мало, зато без
методологического смещения.

    python -m mlaw.basket --synthetic 40 --temporal-acts 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

__all__ = ["Query", "Gold", "build_metadata", "build_temporal", "build_synthetic"]

# Из титула вида «Постановление Правительства Москвы от 09.02.2010 N 125-ПП»
_TITLE = re.compile(
    r"^(?P<type>[А-ЯЁ][а-яё]+)\s+(?P<body>.+?)\s+от\s+(?P<date>\d{2}\.\d{2}\.\d{4})"
    r"(?:\s+N\s+(?P<number>\S+))?"
)


@dataclass(slots=True)
class Gold:
    """Эталон на трёх уровнях. Чем ниже уровень, тем строже проверка."""

    act_id: int
    doc_id: int | None = None
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Query:
    query_id: str
    type: str
    query: str
    gold: Gold
    as_of: str | None = None
    note: str = ""

    def to_json(self) -> dict:
        out = asdict(self)
        out["gold"] = asdict(self.gold)
        return out


# --------------------------------------------------------------------------- #
# Загрузка среза
# --------------------------------------------------------------------------- #


def load_slice(path: Path) -> dict[int, list[dict]]:
    """act_id -> редакции, отсортированные по началу действия."""
    acts: dict[int, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            edition = row.get("edition") or {}
            acts[edition.get("act_id")].append(
                {
                    "doc_id": row["doc_id"],
                    "title": row.get("title") or "",
                    "doc_type": row.get("doc_type"),
                    "status": row.get("status"),
                    "is_current": bool(edition.get("is_current")),
                    "begin": row.get("effective_date_begin"),
                    "end": row.get("effective_date_end"),
                    "end_sentinel": row.get("effective_date_end_sentinel"),
                    "chars": len(row.get("text") or ""),
                }
            )
    for editions in acts.values():
        editions.sort(key=lambda e: e["begin"] or "")
    return acts


# --------------------------------------------------------------------------- #
# Тип 1: реквизиты
# --------------------------------------------------------------------------- #


def build_metadata(acts: dict[int, list[dict]], count: int, seed: int) -> list[Query]:
    """Запросы по номеру и реквизитам акта.

    Формулируются так, как их набирают в поиске: коротко и с номером.
    Эталон — акт целиком; уровня чанка тут нет, потому что вопрос не про
    содержание, а про то, найдётся ли нужный документ вообще.
    """
    candidates = []
    for act_id, editions in acts.items():
        current = next((e for e in editions if e["is_current"]), editions[-1])
        match = _TITLE.match(current["title"])
        if not match or not match.group("number"):
            continue
        candidates.append((act_id, current, match))

    rng = random.Random(seed)
    rng.shuffle(candidates)

    queries: list[Query] = []
    for index, (act_id, current, match) in enumerate(candidates[:count]):
        number = match.group("number")
        doc_type = match.group("type").lower()
        body = match.group("body")
        issued = match.group("date")

        # Три формы одного запроса — короткая, средняя и полная. Так корзина
        # проверяет не одну формулировку, а устойчивость к ним.
        form = index % 3
        if form == 0:
            text = number
        elif form == 1:
            text = f"{doc_type} {number}"
        else:
            text = f"{doc_type} {body} от {issued} N {number}"

        queries.append(
            Query(
                query_id=f"meta-{act_id}-{form}",
                type="metadata",
                query=text,
                gold=Gold(act_id=act_id, doc_id=current["doc_id"]),
                note=current["title"][:120],
            )
        )
    return queries


# --------------------------------------------------------------------------- #
# Тип 2: темпоральные пары
# --------------------------------------------------------------------------- #


def _midpoint(begin: str, end: str | None) -> str | None:
    """Дата внутри интервала действия редакции — заведомо однозначная."""
    if not begin:
        return None
    start = date.fromisoformat(begin)
    if end:
        finish = date.fromisoformat(end)
        if finish < start:
            return None
        return (start + (finish - start) / 2).isoformat()
    # Открытый интервал: берём дату заведомо после начала, но не в будущем.
    return (start + timedelta(days=30)).isoformat()


def build_temporal(
    acts: dict[int, list[dict]], count: int, seed: int
) -> list[Query]:
    """Пары «один акт, две даты, разные редакции».

    Запрос намеренно содержит реквизиты акта: измеряется не способность
    угадать акт по смыслу, а способность выбрать ПРАВИЛЬНУЮ РЕДАКЦИЮ.
    Если система вернёт тот же акт, но не ту редакцию, метрика на уровне
    act_id этого не заметит, а на уровне doc_id — заметит.
    """
    usable = []
    for act_id, editions in acts.items():
        if len(editions) < 2:
            continue
        dated = [e for e in editions if e["begin"] and _midpoint(e["begin"], e["end"])]
        if len(dated) >= 2:
            usable.append((act_id, dated))

    rng = random.Random(seed)
    rng.shuffle(usable)

    queries: list[Query] = []
    for act_id, editions in usable[:count]:
        # Первая и последняя редакции — между ними разница максимальна.
        chosen = [editions[0], editions[-1]]
        match = _TITLE.match(editions[-1]["title"])
        number = match.group("number") if match and match.group("number") else None
        subject = f"{editions[-1]['doc_type'] or 'акт'} {number}" if number else editions[-1]["title"][:60]

        for position, edition in enumerate(chosen):
            as_of = _midpoint(edition["begin"], edition["end"])
            queries.append(
                Query(
                    query_id=f"temp-{act_id}-{position}",
                    type="temporal",
                    query=f"{subject} — редакция, действовавшая на {as_of}",
                    gold=Gold(act_id=act_id, doc_id=edition["doc_id"]),
                    as_of=as_of,
                    note=f"{edition['begin']}..{edition['end'] or edition['end_sentinel']}",
                )
            )
    return queries


# --------------------------------------------------------------------------- #
# Тип 2b: темпоральные запросы без номера акта
# --------------------------------------------------------------------------- #

SEMANTIC_TEMPORAL_SYSTEM = (
    "Ты составляешь тестовую корзину для поиска по правовым актам города Москвы. "
    "Отвечай строго в формате JSON, без пояснений."
)

SEMANTIC_TEMPORAL_PROMPT = """Вот фрагмент правового акта города Москвы — одна из его редакций.

Сформулируй ОДИН вопрос о предмете этого фрагмента: что регулируется, кому
и на каких условиях положено, каков порядок. Спроси так, как спросил бы
человек, который не знает и не называет номер акта, а спрашивает по существу.

Требования к вопросу:
- НЕ упоминай номер акта, реквизиты вида "постановление N ...", дату издания;
- НЕ переписывай формулировки фрагмента дословно, перефразируй;
- если фрагмент не даёт материала для содержательного вопроса (оглавление,
  таблица без контекста, обрывок), поставь answerable=false.

Фрагмент:
{text}

Верни JSON: {{"question": "...", "answerable": true|false}}"""


def build_temporal_semantic(
    acts: dict[int, list[dict]],
    chunks_path: Path,
    count: int,
    seed: int,
    *,
    model: str,
) -> tuple[list[Query], dict]:
    """Темпоральные запросы БЕЗ номера акта в тексте.

    `build_temporal` меряет способность выбрать правильную редакцию, но
    запрос там называет акт по номеру — а с названным актом темпоральный
    фильтр не может ошибиться редакцией: он оставляет ровно одну, чей
    интервал накрывает `as_of` (непересекаемость интервалов доказана
    инвентаризацией). «Нашёлся акт» и «нашлась нужная редакция» становятся
    одним и тем же событием ещё до ранжирования — это измеренное свойство
    пайплайна (см. REPORT §4), но оно же делает `temporal` слепым к одной
    вещи: находит ли система акт по смыслу, если по номеру его не назвать.

    Здесь акт называется по содержанию конкретной редакции, а не по
    реквизитам: систему нужно сначала найти по смыслу вопроса, а не по
    номеру, и только затем фильтр выберет редакцию. Цена та же, что
    у `synthetic`: вопрос сочинён по тексту самой редакции, значит
    лексически смещён в её пользу — обе редакции акта почти наверняка
    используют разные формулировки, так что смещение слабее, чем
    у `synthetic`, но оно есть, и это идёт в раздел ограничений, а не
    замалчивается.
    """
    from mlaw.llm import DeepSeek

    # Тот же отбор актов, что у build_temporal: минимум две датированные
    # редакции, иначе выбирать не из чего.
    usable = []
    for act_id, editions in acts.items():
        if len(editions) < 2:
            continue
        dated = [e for e in editions if e["begin"] and _midpoint(e["begin"], e["end"])]
        if len(dated) >= 2:
            usable.append((act_id, dated))

    rng = random.Random(seed)
    rng.shuffle(usable)

    # doc_id -> первый встреченный чанк редакции с достаточным содержанием.
    # Читаем chunks.jsonl один раз, а не по одному doc_id — так же, как
    # build_synthetic читает пул целиком.
    wanted_docs: set[int] = set()
    for act_id, editions in usable:
        wanted_docs.add(editions[0]["doc_id"])
        wanted_docs.add(editions[-1]["doc_id"])
    text_by_doc: dict[int, str] = {}
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            doc_id = row["doc_id"]
            if doc_id in wanted_docs and doc_id not in text_by_doc and len(row["text"]) >= 400:
                text_by_doc[doc_id] = row["text"]

    llm = DeepSeek(model=model)
    queries: list[Query] = []
    rejected = 0
    started = time.time()

    for act_id, editions in usable:
        if len(queries) >= count:
            break
        chosen = [editions[0], editions[-1]]
        pair = [(edition, text_by_doc.get(edition["doc_id"])) for edition in chosen]
        if any(text is None for _, text in pair):
            continue  # ни одна из редакций не пошла в индекс без чанка

        for position, (edition, text) in enumerate(pair):
            if len(queries) >= count:
                break
            as_of = _midpoint(edition["begin"], edition["end"])
            try:
                result = llm.complete(
                    SEMANTIC_TEMPORAL_SYSTEM,
                    SEMANTIC_TEMPORAL_PROMPT.format(text=text[:3000]),
                    json_mode=True, max_tokens=3000,
                )
                parsed = result.json()
            except Exception:
                rejected += 1
                continue
            if not parsed.get("answerable") or not parsed.get("question"):
                rejected += 1
                continue
            queries.append(
                Query(
                    query_id=f"tsem-{act_id}-{position}",
                    type="temporal_semantic",
                    query=f"{parsed['question'].strip()} (по состоянию на {as_of})",
                    gold=Gold(act_id=act_id, doc_id=edition["doc_id"]),
                    as_of=as_of,
                    note=f"{edition['begin']}..{edition['end'] or edition['end_sentinel']}",
                )
            )

    stats = {
        "requested": count,
        "produced": len(queries),
        "rejected_as_unanswerable": rejected,
        "prompt_tokens": llm.prompt_tokens,
        "completion_tokens": llm.completion_tokens,
        "seconds": round(time.time() - started, 1),
    }
    return queries[:count], stats


# --------------------------------------------------------------------------- #
# Тип 3: синтетика по фрагменту
# --------------------------------------------------------------------------- #

SYSTEM = (
    "Ты составляешь тестовую корзину для поиска по правовым актам города Москвы. "
    "Отвечай строго в формате JSON, без пояснений."
)

PROMPT = """Вот фрагмент правового акта города Москвы.

Сформулируй ОДИН вопрос, ответ на который содержится именно в этом фрагменте.

Требования к вопросу:
- так, как его задал бы юрист или сотрудник органа власти своими словами;
- НЕ переписывай формулировки фрагмента дословно, перефразируй;
- не упоминай номер акта и не ссылайся на "данный фрагмент";
- если фрагмент бессодержателен (оглавление, таблица без контекста, обрывок),
  поставь answerable=false.

Фрагмент:
{text}

Верни JSON: {{"question": "...", "answerable": true|false}}"""


def build_synthetic(
    chunks_path: Path,
    count: int,
    seed: int,
    *,
    model: str,
    concurrency: int = 8,
    min_chars: int = 900,
    max_per_act: int = 2,
) -> tuple[list[Query], dict]:
    """Вопрос сочиняется по фрагменту, эталон — сам фрагмент.

    Это единственный тип с разметкой на уровне чанка, поэтому без него
    строгую метрику посчитать не на чем. Цена — лексическое смещение:
    как бы модель ни перефразировала, она видит именно этот текст.
    """
    from mlaw.llm import DeepSeek

    # Только действующие редакции. Вопрос без даты означает «как сейчас»,
    # и поиск фильтрует по is_current — эталон в исторической редакции был бы
    # недостижим в принципе, и метрика мерила бы этот дефект, а не качество.
    pool: list[dict] = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if len(row["text"]) >= min_chars and row.get("is_current"):
                pool.append(row)

    rng = random.Random(seed)
    rng.shuffle(pool)

    # Не больше max_per_act вопросов на акт. Иначе один акт, занимающий 81 %
    # индекса, забирает четверть корзины, и метрика описывает его, а не корпус.
    per_act: Counter = Counter()
    sample: list[dict] = []
    for row in pool:
        if per_act[row["act_id"]] >= max_per_act:
            continue
        per_act[row["act_id"]] += 1
        sample.append(row)
        if len(sample) >= int(count * 1.6):
            break

    llm = DeepSeek(model=model)
    queries: list[Query] = []
    rejected = 0
    started = time.time()

    def ask(row: dict):
        return row, llm.complete(
            SYSTEM, PROMPT.format(text=row["text"][:3000]), json_mode=True, max_tokens=3000
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(ask, row) for row in sample]
        for future in as_completed(futures):
            if len(queries) >= count:
                continue
            try:
                row, result = future.result()
                parsed = result.json()
            except Exception:
                rejected += 1
                continue
            if not parsed.get("answerable") or not parsed.get("question"):
                rejected += 1
                continue
            queries.append(
                Query(
                    query_id=f"syn-{row['chunk_id']}",
                    type="synthetic",
                    query=parsed["question"].strip(),
                    gold=Gold(
                        act_id=row["act_id"],
                        doc_id=row["doc_id"],
                        chunk_ids=[row["chunk_id"]],
                    ),
                    note=row.get("breadcrumb", "")[:120],
                )
            )

    stats = {
        "requested": count,
        "produced": len(queries),
        "rejected_as_unanswerable": rejected,
        "prompt_tokens": llm.prompt_tokens,
        "completion_tokens": llm.completion_tokens,
        "seconds": round(time.time() - started, 1),
    }
    return queries[:count], stats


# --------------------------------------------------------------------------- #
# Сборка и разделение
# --------------------------------------------------------------------------- #


def split_dev_test(queries: list[Query], seed: int, dev_share: float = 0.4):
    """Стратифицированное разделение: доли типов в dev и test одинаковы.

    Все настройки подбираются на dev, итог считается один раз на test.
    Без этого метрика окажется подогнанной под собственную корзину.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[Query]] = defaultdict(list)
    for query in queries:
        by_type[query.type].append(query)

    dev: list[Query] = []
    test: list[Query] = []
    for group in by_type.values():
        shuffled = sorted(group, key=lambda q: q.query_id)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * dev_share)
        dev.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return dev, test


def write(path: Path, queries: list[Query]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for query in sorted(queries, key=lambda q: (q.type, q.query_id)):
            fh.write(json.dumps(query.to_json(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Корзина запросов с эталонами")
    parser.add_argument("--slice", type=Path, default=Path("data/slice.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("queries"))
    parser.add_argument("--metadata", type=int, default=30)
    parser.add_argument("--temporal-acts", type=int, default=20)
    parser.add_argument("--temporal-semantic", type=int, default=8)
    parser.add_argument("--synthetic", type=int, default=40)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--regenerate", action="store_true",
                        help="пересчитать синтетику заново, игнорируя кэш")
    args = parser.parse_args()

    acts = load_slice(args.slice)
    print(f"Срез: {len(acts)} актов")

    metadata = build_metadata(acts, args.metadata, args.seed)
    print(f"  реквизиты: {len(metadata)}")

    temporal = build_temporal(acts, args.temporal_acts, args.seed)
    print(f"  темпоральные: {len(temporal)} ({len(temporal) // 2} актов x 2 даты)")

    # Кэшируется тем же приёмом, что синтетика: LLM-вызовы недёшевы,
    # пересборка корзины ради добавленных ручных запросов не должна
    # заново дёргать модель.
    temporal_semantic_path = args.out / "temporal_semantic.jsonl"
    if temporal_semantic_path.exists() and not args.regenerate:
        temporal_semantic = [
            Query(query_id=r["query_id"], type="temporal_semantic", query=r["query"],
                  gold=Gold(**r["gold"]), as_of=r.get("as_of"), note=r.get("note", ""))
            for r in (json.loads(l) for l in open(temporal_semantic_path, encoding="utf-8"))
        ]
        temporal_semantic_stats = {"reused_from_cache": len(temporal_semantic)}
        print(f"  темпоральные (без номера): {len(temporal_semantic)} (из кэша)")
    else:
        temporal_semantic, temporal_semantic_stats = build_temporal_semantic(
            acts, args.chunks, args.temporal_semantic, args.seed, model=args.model,
        )
        write(temporal_semantic_path, temporal_semantic)
        print(f"  темпоральные (без номера): {len(temporal_semantic)} "
              f"(отбраковано {temporal_semantic_stats['rejected_as_unanswerable']}, "
              f"{temporal_semantic_stats['seconds']} с)")

    # Синтетика кэшируется: пересборка корзины из-за добавленных ручных
    # запросов не должна заново дёргать модель и жечь токены.
    cache_path = args.out / "synthetic.jsonl"
    if cache_path.exists() and not args.regenerate:
        synthetic = [
            Query(query_id=r["query_id"], type="synthetic", query=r["query"],
                  gold=Gold(**r["gold"]), as_of=r.get("as_of"), note=r.get("note", ""))
            for r in (json.loads(l) for l in open(cache_path, encoding="utf-8"))
        ]
        stats = {"reused_from_cache": len(synthetic)}
        print(f"  синтетика: {len(synthetic)} (из кэша {cache_path})")
    else:
        synthetic, stats = build_synthetic(
            args.chunks, args.synthetic, args.seed,
            model=args.model, concurrency=args.concurrency,
        )
        write(cache_path, synthetic)
        print(f"  синтетика: {len(synthetic)} (отбраковано {stats['rejected_as_unanswerable']}, "
              f"{stats['seconds']} с, токенов {stats['prompt_tokens']}+{stats['completion_tokens']})")

    manual_path = args.out / "manual.jsonl"
    manual: list[Query] = []
    if manual_path.exists():
        with open(manual_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                manual.append(
                    Query(
                        query_id=row["query_id"], type="manual", query=row["query"],
                        gold=Gold(**row["gold"]), as_of=row.get("as_of"),
                        note=row.get("note", ""),
                    )
                )
    print(f"  ручные: {len(manual)}")

    # temporal_semantic — новый тип, добавлен ПОСЛЕДНИМ в списке намеренно:
    # split_dev_test группирует запросы по типу в порядке первого появления
    # и расходует общий random.Random последовательно по группам. Добавление
    # нового типа в конец не меняет число вызовов rng для существующих групп,
    # значит dev/test-разбиение metadata/temporal/synthetic/manual остаётся
    # побайтово тем же, что было заморожено для чисел в REPORT.md.
    everything = metadata + temporal + synthetic + manual + temporal_semantic
    dev, test = split_dev_test(everything, args.seed)

    write(args.out / "dev.jsonl", dev)
    write(args.out / "test.jsonl", test)
    write(args.out / "all.jsonl", everything)

    print(f"\nВсего {len(everything)} запросов: dev {len(dev)}, test {len(test)}")
    print("  по типам:", dict(Counter(q.type for q in everything)))
    print(f"  записано в {args.out}/")

    (Path("reports") / "basket.json").write_text(
        json.dumps(
            {
                "total": len(everything),
                "dev": len(dev),
                "test": len(test),
                "by_type": dict(Counter(q.type for q in everything)),
                "synthetic_stats": stats,
                "temporal_semantic_stats": temporal_semantic_stats,
                "seed": args.seed,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
