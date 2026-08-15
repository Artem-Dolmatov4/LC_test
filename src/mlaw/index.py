"""Шаг 4 — эмбеддинги и индекс.

Устройство продиктовано двумя вещами, которые уже измерены.

**Эмбеддинги дорогие, индексация дешёвая — значит их надо развести.**
Векторы складываются в кэш (`data/vectors_<модель>.jsonl`) по одному
чанку на строку. Прогон, упавший на 40-тысячном чанке, продолжается
с 40-тысячного, а не с нуля. Загрузка в Qdrant читает кэш и стоит секунды,
поэтому переиндексировать можно сколько угодно раз бесплатно.

**Контекстная модель требует, чтобы чанки документа шли вместе.**
У `voyage-context-4` вектор чанка зависит от соседей по документу, поэтому
единица работы здесь — документ, а не чанк. Поштучные модели группировку
просто игнорируют.

Даты в payload кладутся числом `YYYYMMDD`, а не строкой: диапазонный фильтр
по целому работает в Qdrant без оговорок, а «бессрочно» получает явное
значение 99999999 вместо отсутствия поля. Это прямая реализация того
прочтения интервала, которое проверено на банке.

    python -m mlaw.index --model voyage --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mlaw.embed import Embedder

__all__ = ["EmbeddingCache", "date_ordinal", "end_ordinal", "make_embedder", "load_chunks"]

# «Бессрочно» как конкретное число: фильтр по диапазону не умеет работать
# с отсутствующим полем, а сентинел обязан участвовать в сравнении.
INDEFINITE = 99_999_999


def date_ordinal(value: str | None) -> int | None:
    """``"2010-06-01"`` -> ``20100601``. Ничего не выдумывает: нет даты — None."""
    if not value or len(value) < 10:
        return None
    try:
        return int(value[0:4] + value[5:7] + value[8:10])
    except ValueError:
        return None


def end_ordinal(end: str | None, sentinel: str | None) -> int | None:
    """Конец интервала действия числом.

    ``indefinite`` — бессрочно, это открытый интервал. ``null`` — «неизвестно»,
    и это **не** то же самое: замер на банке показал, что смешивать их нельзя,
    разница между прочтениями достигает 60 тысяч записей.
    """
    if sentinel == "indefinite":
        return INDEFINITE
    return date_ordinal(end)


# --------------------------------------------------------------------------- #
# Кэш векторов
# --------------------------------------------------------------------------- #


class EmbeddingCache:
    """Дописываемый кэш «chunk_id -> вектор», переживающий обрыв прогона."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        self._done.add(json.loads(line)["chunk_id"])
                    except (ValueError, KeyError):
                        continue

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._done

    def __len__(self) -> int:
        return len(self._done)

    def append(self, rows: list[tuple[str, list[float]]]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for chunk_id, vector in rows:
                fh.write(json.dumps({"chunk_id": chunk_id, "v": vector}) + "\n")
                self._done.add(chunk_id)

    def iter_vectors(self) -> Iterator[tuple[str, list[float]]]:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                yield row["chunk_id"], row["v"]


# --------------------------------------------------------------------------- #
# Загрузка чанков
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ChunkRow:
    chunk_id: str
    doc_id: int
    text: str
    breadcrumb: str
    payload: dict


def load_chunks(path: Path, limit: int | None = None) -> list[ChunkRow]:
    rows: list[ChunkRow] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            rows.append(
                ChunkRow(
                    chunk_id=raw["chunk_id"],
                    doc_id=raw["doc_id"],
                    text=raw["text"],
                    breadcrumb=raw.get("breadcrumb") or "",
                    payload=raw,
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def group_by_document(rows: list[ChunkRow]) -> list[list[ChunkRow]]:
    """Чанки одного документа — вместе и в исходном порядке.

    Для контекстной модели это обязательно: перемешав чанки между документами,
    получишь векторы, посчитанные в чужом контексте.
    """
    groups: dict[int, list[ChunkRow]] = defaultdict(list)
    for row in rows:
        groups[row.doc_id].append(row)
    return list(groups.values())


def embed_text(row: ChunkRow, with_breadcrumb: bool) -> str:
    if with_breadcrumb and row.breadcrumb:
        return f"{row.breadcrumb}\n\n{row.text}"
    return row.text


# --------------------------------------------------------------------------- #
# Эмбеддеры
# --------------------------------------------------------------------------- #


def make_embedder(name: str, dim: int) -> Embedder:
    if name == "voyage":
        from mlaw.embed import VoyageEmbedder

        return VoyageEmbedder(dim=dim)
    if name == "dashscope":
        from mlaw.embed import DashScopeEmbedder

        return DashScopeEmbedder(dim=dim)
    if name == "ollama":
        from mlaw.embed import OllamaEmbedder

        return OllamaEmbedder(dim=dim)
    raise ValueError(f"неизвестная модель: {name}")


def run_embedding(
    rows: list[ChunkRow],
    embedder: Embedder,
    cache: EmbeddingCache,
    *,
    with_breadcrumb: bool,
    docs_per_request: int,
) -> dict:
    groups = [g for g in group_by_document(rows) if any(r.chunk_id not in cache for r in g)]
    todo = sum(1 for g in groups for r in g if r.chunk_id not in cache)
    print(f"  к расчёту: {todo} чанков в {len(groups)} документах "
          f"(в кэше уже {len(cache)})")

    tokens = 0
    done = 0
    started = time.time()
    for start in range(0, len(groups), docs_per_request):
        batch = groups[start : start + docs_per_request]
        # Документ отправляется целиком, даже если часть его чанков уже в кэше:
        # контекст обязан быть тем же, иначе векторы окажутся несравнимыми.
        payload = [[embed_text(r, with_breadcrumb) for r in g] for g in batch]
        result = embedder.embed_documents(payload)
        tokens += result.prompt_tokens

        flat = [r for g in batch for r in g]
        if len(flat) != len(result.vectors):
            raise RuntimeError(
                f"модель вернула {len(result.vectors)} векторов на {len(flat)} чанков"
            )
        fresh = [
            (r.chunk_id, v)
            for r, v in zip(flat, result.vectors, strict=True)
            if r.chunk_id not in cache
        ]
        cache.append(fresh)
        done += len(fresh)

        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        remaining = (todo - done) / rate / 60 if rate else 0
        print(f"    {done:>7}/{todo} чанков · {rate:5.2f} чанк/с · "
              f"осталось ~{remaining:5.1f} мин · токенов {tokens:,}".replace(",", " "))

    return {
        "chunks_embedded": done,
        "tokens": tokens,
        "seconds": round(time.time() - started, 1),
        "chunks_per_sec": round(done / max(1e-9, time.time() - started), 3),
    }


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #


def load_into_qdrant(
    rows: list[ChunkRow], cache: EmbeddingCache, collection: str, dim: int, url: str
) -> dict:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=url)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    # Индексы под фильтры, которые реально используются в шаге 6.
    for field, schema in (
        ("act_id", models.PayloadSchemaType.INTEGER),
        ("doc_id", models.PayloadSchemaType.INTEGER),
        ("is_current", models.PayloadSchemaType.BOOL),
        ("status", models.PayloadSchemaType.KEYWORD),
        ("doc_type", models.PayloadSchemaType.KEYWORD),
        ("begin_ord", models.PayloadSchemaType.INTEGER),
        ("end_ord", models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(collection, field_name=field, field_schema=schema)

    by_id = {row.chunk_id: row for row in rows}
    points: list = []
    loaded = 0
    skipped = 0

    for index, (chunk_id, vector) in enumerate(cache.iter_vectors()):
        row = by_id.get(chunk_id)
        if row is None:
            skipped += 1
            continue
        raw = row.payload
        payload = {
            "chunk_id": chunk_id,
            "doc_id": raw["doc_id"],
            "act_id": raw["act_id"],
            "char_start": raw["char_start"],
            "char_end": raw["char_end"],
            "text": raw["text"],
            "breadcrumb": raw.get("breadcrumb") or "",
            "title": raw.get("title"),
            "doc_type": raw.get("doc_type"),
            "status": raw.get("status"),
            "is_current": bool(raw.get("is_current")),
            "boundary_kind": raw.get("boundary_kind"),
            "begin_ord": date_ordinal(raw.get("effective_date_begin")),
            "end_ord": end_ordinal(
                raw.get("effective_date_end"), raw.get("effective_date_end_sentinel")
            ),
        }
        points.append(models.PointStruct(id=index, vector=vector, payload=payload))
        if len(points) >= 512:
            client.upsert(collection, points=points)
            loaded += len(points)
            points = []
    if points:
        client.upsert(collection, points=points)
        loaded += len(points)

    return {
        "collection": collection,
        "points": loaded,
        "skipped_not_in_slice": skipped,
        "count_reported_by_qdrant": client.count(collection).count,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Эмбеддинги и индекс")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--model", default="voyage", choices=("voyage", "dashscope", "ollama"))
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="сколько чанков взять")
    parser.add_argument("--breadcrumb", action="store_true", help="приписывать путь меток")
    parser.add_argument("--docs-per-request", type=int, default=8)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--qdrant", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--skip-load", action="store_true", help="только эмбеддинги")
    args = parser.parse_args()

    suffix = "bc" if args.breadcrumb else "raw"
    tag = f"{args.model}_{args.dim}_{suffix}"
    collection = args.collection or f"mlaw_{tag}"
    cache = EmbeddingCache(Path(f"data/vectors_{tag}.jsonl"))

    rows = load_chunks(args.chunks, args.limit)
    print(f"Чанков к обработке: {len(rows)} · модель {args.model} · dim {args.dim} · "
          f"крошка {'да' if args.breadcrumb else 'нет'}")

    stats = run_embedding(
        rows,
        make_embedder(args.model, args.dim),
        cache,
        with_breadcrumb=args.breadcrumb,
        docs_per_request=args.docs_per_request,
    )
    print(f"\nЭмбеддинги: {stats}")

    if args.skip_load:
        return

    loaded = load_into_qdrant(rows, cache, collection, args.dim, args.qdrant)
    print(f"Qdrant: {loaded}")

    report = Path(f"reports/index_{tag}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"embedding": stats, "qdrant": loaded}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Отчёт: {report}")


if __name__ == "__main__":
    main()
