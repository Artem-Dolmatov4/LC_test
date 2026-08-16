"""Шаг 6 (часть 2) — конвейер поиска.

Порядок стадий здесь не декоративный, каждая стоит там, где стоит, по причине.

    retrieve (обе ноги, темпоральный фильтр ДО слияния)
      -> RRF-слияние
      -> top-m фрагментов на акт
      -> переранжирование кросс-энкодером
      -> схлопывание по act_id
      -> выдача

**Темпоральный фильтр — до слияния.** Постфильтрация обычного top-k молча
теряет правильную редакцию: редакции одного акта похожи друг на друга, и
нужная часто стоит не в первой десятке. Для лексической ноги, где фильтра
в индексе нет, глубина overfetch измерена (правильная редакция находится
полностью начиная с четырёхкратного запаса).

**top-m на акт — до реранка, а не после.** Замер показал, что в топ-10 без
ограничения попадает всего 2 различных акта из 10 позиций: контекстуализация
стягивает чанки документа. Без этого шага реранкер потратил бы весь бюджет
на соседние куски одного и того же акта.

**Схлопывание по акту — после реранка, а не до.** Схлопни раньше — и от акта
останется случайный фрагмент, выбранный по грубому score слияния, а не лучший
доказательный, выбранный кросс-энкодером.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from mlaw.evaluate import Hit

__all__ = ["SearchConfig", "SearchPipeline", "Reranker", "load_chunk_texts"]


@dataclass(slots=True)
class SearchConfig:
    # Сколько кандидатов берёт каждая нога до слияния.
    k_retrieve: int = 60
    # Сколько фрагментов одного акта доходит до реранка.
    per_act_before_rerank: int = 3
    # Сколько кандидатов уходит в кросс-энкодер.
    rerank_top: int = 50
    # Размер итоговой выдачи.
    final_k: int = 10
    dedup_by_act: bool = True
    use_reranker: bool = True


@dataclass(slots=True)
class Stage:
    """След одной стадии — чтобы вклад каждой был виден числом."""

    name: str
    hits: int
    acts: int
    seconds: float


@dataclass(slots=True)
class SearchResult:
    hits: list[Hit]
    stages: list[Stage] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.stages)


def load_chunk_texts(path: Path) -> dict[str, str]:
    """chunk_id -> текст. Нужен реранкеру: лексическая нога текста не хранит."""
    texts: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            texts[row["chunk_id"]] = row["text"]
    return texts


class Reranker:
    """Кросс-энкодер `bge-reranker-v2-m3`.

    Замер на настоящих чанках: 6.5 пары в секунду, то есть около 8 секунд
    на запрос при 50 кандидатах. Батчинг не помогает — 6.1 пары/с при
    batch=32 против 6.5 при batch=8.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "mps"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.name = model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=8)]


class SearchPipeline:
    def __init__(
        self,
        dense,
        lexical,
        texts: dict[str, str],
        reranker: Reranker | None = None,
        config: SearchConfig | None = None,
    ):
        self.dense = dense
        self.lexical = lexical
        self.texts = texts
        self.reranker = reranker
        self.config = config or SearchConfig()

    # -- стадии ------------------------------------------------------------ #

    @staticmethod
    def fuse(pools: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
        """RRF: шкалы косинуса и BM25 несопоставимы, ранги — сопоставимы."""
        scores: dict[str, float] = {}
        seen: dict[str, Hit] = {}
        for pool in pools:
            for rank, hit in enumerate(pool, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                seen.setdefault(hit.chunk_id, hit)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return [
            Hit(chunk_id=cid, doc_id=seen[cid].doc_id, act_id=seen[cid].act_id, score=score)
            for cid, score in ordered
        ]

    @staticmethod
    def cap_per_act(hits: list[Hit], limit: int) -> list[Hit]:
        """Не больше limit фрагментов одного акта, порядок сохраняется."""
        kept: list[Hit] = []
        seen: dict[int, int] = {}
        for hit in hits:
            if seen.get(hit.act_id, 0) >= limit:
                continue
            seen[hit.act_id] = seen.get(hit.act_id, 0) + 1
            kept.append(hit)
        return kept

    @staticmethod
    def collapse_by_act(hits: list[Hit]) -> list[Hit]:
        """По одному лучшему фрагменту на акт, порядок — как пришёл."""
        best: list[Hit] = []
        seen: set[int] = set()
        for hit in hits:
            if hit.act_id in seen:
                continue
            seen.add(hit.act_id)
            best.append(hit)
        return best

    # -- конвейер ---------------------------------------------------------- #

    def search(self, query: str, as_of: str | None = None) -> SearchResult:
        cfg = self.config
        stages: list[Stage] = []

        def mark(name: str, hits: list[Hit], started: float) -> None:
            stages.append(
                Stage(name, len(hits), len({h.act_id for h in hits}), time.time() - started)
            )

        started = time.time()
        dense_hits = self.dense.search(query, cfg.k_retrieve, as_of)
        lexical_hits = self.lexical.search(query, cfg.k_retrieve, as_of)
        mark("retrieve", dense_hits + lexical_hits, started)

        started = time.time()
        fused = self.fuse([dense_hits, lexical_hits])
        mark("fusion", fused, started)

        started = time.time()
        capped = self.cap_per_act(fused, cfg.per_act_before_rerank)[: cfg.rerank_top]
        mark("cap_per_act", capped, started)

        started = time.time()
        if self.reranker is not None and cfg.use_reranker and capped:
            texts = [self.texts.get(h.chunk_id, "") for h in capped]
            scores = self.reranker.score(query, texts)
            ranked = [
                Hit(chunk_id=h.chunk_id, doc_id=h.doc_id, act_id=h.act_id, score=score)
                for h, score in sorted(
                    zip(capped, scores, strict=True), key=lambda pair: -pair[1]
                )
            ]
        else:
            ranked = capped
        mark("rerank", ranked, started)

        started = time.time()
        final = self.collapse_by_act(ranked) if cfg.dedup_by_act else ranked
        final = final[: cfg.final_k]
        mark("collapse", final, started)

        return SearchResult(hits=final, stages=stages)


# --------------------------------------------------------------------------- #
# Абляция стадий
# --------------------------------------------------------------------------- #


def ablate(pipeline: SearchPipeline, queries: list[dict], k: int) -> dict:
    """Считает вклад каждой стадии на одной и той же выдаче.

    Реранк считается ОДИН раз на запрос, а варианты «с дедупом» и «без»
    выводятся из него же: они отличаются только последним шагом, и гонять
    кросс-энкодер дважды было бы просто тратой восьми секунд на запрос.
    """
    from mlaw.evaluate import QueryResult, by_type, metrics_for

    cfg = pipeline.config
    variants: dict[str, list[QueryResult]] = {
        "1_fusion": [],
        "2_cap_per_act": [],
        "3_rerank": [],
        "4_rerank_dedup": [],
    }
    rerank_seconds = 0.0

    for query in queries:
        text, as_of = query["query"], query.get("as_of")

        dense_hits = pipeline.dense.search(text, cfg.k_retrieve, as_of)
        lexical_hits = pipeline.lexical.search(text, cfg.k_retrieve, as_of)
        fused = pipeline.fuse([dense_hits, lexical_hits])
        capped = pipeline.cap_per_act(fused, cfg.per_act_before_rerank)[: cfg.rerank_top]

        started = time.time()
        if pipeline.reranker is not None and capped:
            scores = pipeline.reranker.score(
                text, [pipeline.texts.get(h.chunk_id, "") for h in capped]
            )
            ranked = [
                Hit(chunk_id=h.chunk_id, doc_id=h.doc_id, act_id=h.act_id, score=s)
                for h, s in sorted(zip(capped, scores, strict=True), key=lambda p: -p[1])
            ]
        else:
            ranked = capped
        rerank_seconds += time.time() - started

        for name, hits in (
            ("1_fusion", fused[:k]),
            ("2_cap_per_act", capped[:k]),
            ("3_rerank", ranked[:k]),
            ("4_rerank_dedup", pipeline.collapse_by_act(ranked)[:k]),
        ):
            variants[name].append(
                QueryResult(query_id=query["query_id"], type=query["type"], hits=hits)
            )

    golds = {q["query_id"]: q["gold"] for q in queries}
    return {
        "k": k,
        "queries": len(queries),
        "rerank_seconds_total": round(rerank_seconds, 1),
        "rerank_seconds_per_query": round(rerank_seconds / max(1, len(queries)), 2),
        "variants": {
            name: {"overall": metrics_for(res, golds), "by_type": by_type(res, golds)}
            for name, res in variants.items()
        },
    }


def print_ablation(report: dict) -> None:
    print(f"\n{'=' * 76}")
    print(f"  Абляция стадий · {report['queries']} запросов · k={report['k']}")
    print(f"  Реранк: {report['rerank_seconds_per_query']} с на запрос")
    print()
    header = f"  {'вариант':<18} {'act R@10':>9} {'act MRR':>9} {'chunk R@10':>11} {'chunk MRR':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, data in report["variants"].items():
        act = data["overall"].get("act", {})
        chunk = data["overall"].get("chunk", {})
        print(f"  {name:<18} {act.get('recall@10', 0):>9.3f} {act.get('mrr', 0):>9.3f} "
              f"{chunk.get('recall@10', 0):>11.3f} {chunk.get('mrr', 0):>10.3f}")
    print(f"{'=' * 76}")


def main() -> None:
    import argparse

    from mlaw.evaluate import build_retrievers, load_basket

    parser = argparse.ArgumentParser(description="Конвейер поиска и абляция стадий")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--collection", default="mlaw_voyage_1024_raw")
    parser.add_argument("--model", default="voyage")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--qdrant", default="http://localhost:6333")
    parser.add_argument("--bm25", type=Path, default=Path("data/bm25"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--per-act", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    queries = load_basket(Path(f"queries/{args.split}.jsonl"))
    print(f"Корзина: {args.split} — {len(queries)} запросов")

    retrievers = build_retrievers(
        args.collection, args.dim, args.model, args.qdrant, args.bm25
    )
    retrievers["dense"].encode([q["query"] for q in queries])

    print("Загружаю тексты чанков…")
    texts = load_chunk_texts(args.chunks)

    reranker = None
    if not args.no_rerank:
        print("Поднимаю кросс-энкодер…")
        reranker = Reranker()

    pipeline = SearchPipeline(
        retrievers["dense"], retrievers["lexical"], texts, reranker,
        SearchConfig(per_act_before_rerank=args.per_act, final_k=args.k),
    )

    report = ablate(pipeline, queries, args.k)
    report["config"] = {
        "per_act_before_rerank": args.per_act,
        "reranker": reranker.name if reranker else None,
        "collection": args.collection,
    }
    print_ablation(report)

    out = args.out or Path(f"reports/ablation_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
