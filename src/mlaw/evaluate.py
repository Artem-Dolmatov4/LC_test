"""Шаг 5 (часть 2) — измерение качества поиска.

Три принципа, каждый следует из уже сделанных замеров.

**Метрика считается на трёх уровнях сразу.** `act_id` — нашёлся ли нужный
акт. `doc_id` — та ли редакция. Чанк — то ли место внутри документа.
Гейт шага 0.5 показал, что контекстуализация стягивает чанки одного
документа (косинус 0.92 между разными чанками против 0.74 между одним и тем
же чанком в контексте и вне его), поэтому уровень акта заведомо будет лёгким,
а уровень чанка — строгим. Одно число вместо трёх скрыло бы ровно то, где
система ломается.

**Контроли обязаны обваливать метрику.** Если подмена эталона не роняет
Recall, значит меряется не то, что предполагалось. Контроль, который не
срабатывает, — это не хорошая новость, а сломанная проверка.

**Темпоральный фильтр применяется до слияния, а не после.** Постфильтрация
обычного top-k молча теряет правильную редакцию. Для лексической ноги, где
фильтра в индексе нет, используется overfetch, и достаточность его глубины
измеряется, а не назначается.

    python -m mlaw.evaluate --split dev --mode hybrid
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mlaw.index import date_ordinal

__all__ = ["Hit", "evaluate", "metrics_for", "RETRIEVAL_MODES"]

RETRIEVAL_MODES = ("dense", "lexical", "hybrid", "random")

# Глубина, на которой считаются метрики.
K_VALUES = (1, 5, 10, 20)
NDCG_AT = 10
# Во сколько раз лексическая нога берёт кандидатов сверх нужного, чтобы
# темпоральный фильтр не отсёк всё полезное. Достаточность проверяется
# отдельным замером, см. `overfetch_probe`.
LEXICAL_OVERFETCH = 8


@dataclass(slots=True)
class Hit:
    chunk_id: str
    doc_id: int
    act_id: int
    score: float


@dataclass(slots=True)
class QueryResult:
    query_id: str
    type: str
    hits: list[Hit]
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Ретриверы
# --------------------------------------------------------------------------- #


class DenseRetriever:
    """Плотный вектор из Qdrant с темпоральным фильтром в самом запросе."""

    def __init__(self, collection: str, embedder, url: str = "http://localhost:6333"):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embedder = embedder
        self._cache: dict[str, list[float]] = {}

    def encode(self, queries: list[str]) -> None:
        """Считает векторы запросов пачкой — по одному это дороже на порядок."""
        fresh = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if not fresh:
            return
        result = self.embedder.embed_queries(fresh)
        for query, vector in zip(fresh, result.vectors, strict=True):
            self._cache[query] = vector

    def _filter(self, as_of: str | None):
        from qdrant_client import models

        if as_of is None:
            # Без даты вопрос про сегодняшнее состояние — берём текущие редакции.
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_current", match=models.MatchValue(value=True)
                    )
                ]
            )
        # С датой — интервал действия, БЕЗ условия по статусу: статус описывает
        # акт сегодня, а не на дату запроса. Замер на банке: условие по статусу
        # на 2010 год выбрасывает 30 тысяч записей, отменённых уже после.
        ordinal = date_ordinal(as_of)
        return models.Filter(
            must=[
                models.FieldCondition(key="begin_ord", range=models.Range(lte=ordinal)),
                models.FieldCondition(key="end_ord", range=models.Range(gte=ordinal)),
            ]
        )

    def search(self, query: str, k: int, as_of: str | None = None) -> list[Hit]:
        self.encode([query])
        points = self.client.query_points(
            self.collection,
            query=self._cache[query],
            limit=k,
            query_filter=self._filter(as_of),
            with_payload=True,
        ).points
        return [
            Hit(
                chunk_id=p.payload["chunk_id"],
                doc_id=p.payload["doc_id"],
                act_id=p.payload["act_id"],
                score=float(p.score),
            )
            for p in points
        ]


class LexicalRetriever:
    """BM25 с темпоральным фильтром через overfetch.

    В индексе BM25 фильтра нет, поэтому кандидаты берутся с запасом и
    фильтруются до усечения до top-k. Постфильтрация обычного top-k здесь
    была бы прямой ошибкой: правильная редакция часто стоит не в первой
    десятке именно потому, что редакции похожи друг на друга.
    """

    def __init__(self, index, overfetch: int = LEXICAL_OVERFETCH):
        self.index = index
        self.overfetch = overfetch
        self.by_chunk = {p["chunk_id"]: p for p in index.payloads}

    def _in_force(self, payload: dict, as_of: str | None) -> bool:
        if as_of is None:
            return payload["is_current"]
        ordinal = date_ordinal(as_of)
        begin = date_ordinal(payload.get("effective_date_begin"))
        if payload.get("effective_date_end_sentinel") == "indefinite":
            end = 99_999_999
        else:
            end = date_ordinal(payload.get("effective_date_end"))
        return begin is not None and end is not None and begin <= ordinal <= end

    def search(self, query: str, k: int, as_of: str | None = None) -> list[Hit]:
        raw = self.index.search(query, k=k * self.overfetch)
        hits: list[Hit] = []
        for item in raw:
            payload = self.index.payloads[item.index]
            if not self._in_force(payload, as_of):
                continue
            hits.append(
                Hit(
                    chunk_id=payload["chunk_id"],
                    doc_id=payload["doc_id"],
                    act_id=payload["act_id"],
                    score=item.score,
                )
            )
            if len(hits) >= k:
                break
        return hits


class HybridRetriever:
    """Слияние Reciprocal Rank Fusion.

    RRF выбран потому, что шкалы несопоставимы: у плотной ноги косинус
    в диапазоне около нуля до единицы, у BM25 — неограниченный вес. Слияние
    по рангам не требует калибровки и не даёт одной ноге задавить другую
    просто масштабом.
    """

    def __init__(self, dense: DenseRetriever, lexical: LexicalRetriever, rrf_k: int = 60):
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k

    def search(self, query: str, k: int, as_of: str | None = None) -> list[Hit]:
        pools = [
            self.dense.search(query, k * 3, as_of),
            self.lexical.search(query, k * 3, as_of),
        ]
        scores: dict[str, float] = defaultdict(float)
        seen: dict[str, Hit] = {}
        for pool in pools:
            for rank, hit in enumerate(pool, start=1):
                scores[hit.chunk_id] += 1.0 / (self.rrf_k + rank)
                seen.setdefault(hit.chunk_id, hit)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(
                chunk_id=chunk_id,
                doc_id=seen[chunk_id].doc_id,
                act_id=seen[chunk_id].act_id,
                score=score,
            )
            for chunk_id, score in ordered
        ]


class RandomRetriever:
    """Контроль-основание: случайные чанки.

    Даёт нижнюю границу. Если настоящий ретривер её не превосходит
    с большим отрывом, измерять нечего.
    """

    def __init__(self, payloads: list[dict], seed: int = 20260815):
        self.payloads = payloads
        self.rng = random.Random(seed)

    def search(self, query: str, k: int, as_of: str | None = None) -> list[Hit]:
        sample = self.rng.sample(self.payloads, min(k, len(self.payloads)))
        return [
            Hit(
                chunk_id=p["chunk_id"], doc_id=p["doc_id"], act_id=p["act_id"], score=0.0
            )
            for p in sample
        ]


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #


def _rank_of_first_match(hits: list[Hit], level: str, gold: dict) -> int | None:
    """Позиция первого правильного попадания, считая с единицы."""
    if level == "act":
        target = {gold["act_id"]}
        values = [h.act_id for h in hits]
    elif level == "doc":
        if gold.get("doc_id") is None:
            return None
        target = {gold["doc_id"]}
        values = [h.doc_id for h in hits]
    else:
        if not gold.get("chunk_ids"):
            return None
        target = set(gold["chunk_ids"])
        values = [h.chunk_id for h in hits]

    for position, value in enumerate(values, start=1):
        if value in target:
            return position
    return None


def metrics_for(results: list[QueryResult], golds: dict[str, dict]) -> dict:
    """Recall@k, MRR и nDCG@10 на трёх уровнях релевантности."""
    out: dict = {}
    for level in ("act", "doc", "chunk"):
        applicable = 0
        recall = {k: 0 for k in K_VALUES}
        reciprocal = 0.0
        ndcg = 0.0

        for result in results:
            gold = golds[result.query_id]
            rank = _rank_of_first_match(result.hits, level, gold)
            has_gold = (
                level == "act"
                or (level == "doc" and gold.get("doc_id") is not None)
                or (level == "chunk" and bool(gold.get("chunk_ids")))
            )
            if not has_gold:
                continue
            applicable += 1
            if rank is None:
                continue
            for k in K_VALUES:
                if rank <= k:
                    recall[k] += 1
            reciprocal += 1.0 / rank
            if rank <= NDCG_AT:
                # Бинарная релевантность: идеальный DCG равен единице,
                # поэтому nDCG сводится к 1/log2(rank+1).
                ndcg += 1.0 / math.log2(rank + 1)

        if applicable == 0:
            continue
        out[level] = {
            "queries": applicable,
            **{f"recall@{k}": round(recall[k] / applicable, 4) for k in K_VALUES},
            "mrr": round(reciprocal / applicable, 4),
            f"ndcg@{NDCG_AT}": round(ndcg / applicable, 4),
        }
    return out


def by_type(results: list[QueryResult], golds: dict[str, dict]) -> dict:
    grouped: dict[str, list[QueryResult]] = defaultdict(list)
    for result in results:
        grouped[result.type].append(result)
    return {name: metrics_for(group, golds) for name, group in sorted(grouped.items())}


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def load_basket(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def run_retrieval(retriever, queries: list[dict], k: int) -> list[QueryResult]:
    results: list[QueryResult] = []
    for query in queries:
        started = time.time()
        hits = retriever.search(query["query"], k, query.get("as_of"))
        results.append(
            QueryResult(
                query_id=query["query_id"],
                type=query["type"],
                hits=hits,
                seconds=time.time() - started,
            )
        )
    return results


def shuffled_gold(queries: list[dict], seed: int) -> dict[str, dict]:
    """Контроль: эталон подменяется на чужой.

    Метрика обязана обвалиться почти до нуля. Если не обвалилась — значит
    попадания засчитываются не тем механизмом, каким кажется.
    """
    rng = random.Random(seed)
    acts = [q["gold"]["act_id"] for q in queries]
    docs = [q["gold"].get("doc_id") for q in queries]
    chunks = [q["gold"].get("chunk_ids") or [] for q in queries]
    order = list(range(len(queries)))
    rng.shuffle(order)

    out: dict[str, dict] = {}
    for position, query in enumerate(queries):
        source = order[position]
        # Один сдвиг на +1 не гарантирует несовпадение: типы temporal
        # и temporal_semantic дают по несколько запросов на один акт,
        # и после единственного шага источник может снова оказаться
        # тем же актом. Сдвигаем, пока не найдём чужой — а не проверяем
        # один раз и считаем дело сделанным.
        seen = 0
        while acts[source] == query["gold"]["act_id"] and seen < len(queries):
            source = (source + 1) % len(queries)
            seen += 1
        out[query["query_id"]] = {
            "act_id": acts[source],
            "doc_id": docs[source],
            "chunk_ids": chunks[source],
        }
    return out


def overfetch_probe(lexical: LexicalRetriever, queries: list[dict], k: int) -> dict:
    """Проверяет, хватает ли глубины лексического overfetch.

    Мерится доля темпоральных запросов, у которых правильная редакция
    находится в отфильтрованной выдаче. Если она растёт с глубиной, значит
    выбранного запаса мало и постфильтрация теряла бы ответы.
    """
    temporal = [q for q in queries if q["type"] == "temporal"]
    if not temporal:
        return {}
    original = lexical.overfetch
    probe: dict[str, float] = {}
    for depth in (1, 2, 4, 8, 16):
        lexical.overfetch = depth
        found = 0
        for query in temporal:
            hits = lexical.search(query["query"], k, query.get("as_of"))
            if query["gold"]["doc_id"] in {h.doc_id for h in hits}:
                found += 1
        probe[f"x{depth}"] = round(found / len(temporal), 4)
    lexical.overfetch = original
    return probe


def build_retrievers(collection: str, dim: int, model: str, qdrant_url: str, bm25_dir: Path):
    from mlaw.index import make_embedder
    from mlaw.lexical import LexicalIndex

    dense = DenseRetriever(collection, make_embedder(model, dim), qdrant_url)
    lexical = LexicalRetriever(LexicalIndex.load(bm25_dir))
    return {
        "dense": dense,
        "lexical": lexical,
        "hybrid": HybridRetriever(dense, lexical),
        "random": RandomRetriever(lexical.index.payloads),
    }


def evaluate(
    queries: list[dict],
    retrievers: dict,
    modes: tuple[str, ...],
    k: int,
    seed: int,
) -> dict:
    golds = {q["query_id"]: q["gold"] for q in queries}
    report: dict = {"queries": len(queries), "k": k, "modes": {}}

    # Векторы запросов считаются один раз на все режимы.
    dense = retrievers.get("dense")
    if dense is not None:
        dense.encode([q["query"] for q in queries])

    for mode in modes:
        retriever = retrievers[mode]
        started = time.time()
        results = run_retrieval(retriever, queries, k)
        report["modes"][mode] = {
            "overall": metrics_for(results, golds),
            "by_type": by_type(results, golds),
            "seconds": round(time.time() - started, 1),
        }

    # --- контроли ------------------------------------------------------- #
    primary = modes[0]
    results = run_retrieval(retrievers[primary], queries, k)

    controls: dict = {}
    controls["shuffled_gold"] = metrics_for(results, shuffled_gold(queries, seed))
    controls["random_retriever"] = metrics_for(
        run_retrieval(retrievers["random"], queries, k), golds
    )
    if "lexical" in retrievers:
        controls["lexical_overfetch_probe"] = overfetch_probe(
            retrievers["lexical"], queries, k
        )
    report["controls"] = controls
    report["control_reference"] = {
        "mode": primary,
        "act_recall@10": report["modes"][primary]["overall"].get("act", {}).get("recall@10"),
    }
    return report


def print_report(report: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Запросов: {report['queries']} · глубина k={report['k']}")
    for mode, data in report["modes"].items():
        print(f"\n  --- {mode} ({data['seconds']} с) ---")
        for level, m in data["overall"].items():
            print(f"    {level:<6} n={m['queries']:<4} "
                  f"R@1 {m['recall@1']:.3f} · R@5 {m['recall@5']:.3f} · "
                  f"R@10 {m['recall@10']:.3f} · MRR {m['mrr']:.3f} · nDCG@10 {m['ndcg@10']:.3f}")
        print("    по типам (act / doc / chunk, recall@10):")
        for qtype, levels in data["by_type"].items():
            cells = " · ".join(
                f"{lv}: {levels[lv]['recall@10']:.3f}" for lv in ("act", "doc", "chunk")
                if lv in levels
            )
            print(f"      {qtype:<10} {cells}")

    print(f"\n  --- контроли ---")
    ref = report["control_reference"]
    print(f"    опорное значение ({ref['mode']}, act recall@10): {ref['act_recall@10']}")
    for name, data in report["controls"].items():
        if name == "lexical_overfetch_probe":
            print(f"    overfetch лексики (доля найденных правильных редакций):")
            for depth, share in data.items():
                print(f"      {depth:>4}: {share:.3f}")
            continue
        act = data.get("act", {})
        print(f"    {name:<18} act recall@10 = {act.get('recall@10')} "
              f"· MRR {act.get('mrr')}")
    print(f"{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества поиска")
    parser.add_argument("--basket", type=Path, default=None)
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument("--collection", default="mlaw_voyage_1024_raw")
    parser.add_argument("--model", default="voyage")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--qdrant", default="http://localhost:6333")
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--modes", default="hybrid,dense,lexical")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    basket = args.basket or Path(f"queries/{args.split}.jsonl")
    queries = load_basket(basket)
    print(f"Корзина: {basket} — {len(queries)} запросов")
    print(f"  по типам: {dict(Counter(q['type'] for q in queries))}")

    retrievers = build_retrievers(
        args.collection, args.dim, args.model, args.qdrant, args.bm25
    )
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    report = evaluate(queries, retrievers, modes, args.k, args.seed)
    report["meta"] = {
        "basket": str(basket),
        "split": args.split,
        "collection": args.collection,
        "model": args.model,
    }
    print_report(report)

    out = args.out or Path(f"reports/eval_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
