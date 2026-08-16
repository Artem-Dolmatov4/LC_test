"""Шаг 6 (часть 3) — ответ с программно проверяемыми цитатами.

Модель здесь не источник знания, а сборщик: она обязана отвечать только по
переданным фрагментам и подкреплять каждое утверждение ссылкой вида
``[[doc_id:char_start-char_end]]``. Ссылки не принимаются на веру — каждая
резолвится обратно в текст среза, и процитированный кусок сверяется
дословно (см. :mod:`mlaw.citations`).

Отказ — штатный исход, а не сбой. Если во фрагментах ответа нет, модель
обязана это сказать, и такой ответ считается правильным поведением.
Проверяется отдельным контролем: запрос про предмет, которого в корпусе
заведомо нет, должен получать отказ, а не правдоподобную выдумку.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from mlaw.citations import CitationResolver

__all__ = ["AnswerRequest", "Answer", "answer_query", "SYSTEM"]

SYSTEM = (
    "Ты — помощник юриста по правовым актам города Москвы. "
    "Ты отвечаешь ИСКЛЮЧИТЕЛЬНО по переданным фрагментам актов. "
    "Отвечай строго в формате JSON."
)

PROMPT = """Вопрос: {question}
{as_of_note}
Ниже — фрагменты правовых актов. У каждого есть идентификатор в квадратных скобках.

{fragments}

Правила:
1. Отвечай ТОЛЬКО по этим фрагментам. Никаких сведений из собственных знаний.
2. Каждое утверждение подкрепляй ссылкой [[идентификатор]] сразу после него.
   Ответ без единой ссылки недопустим: если ссылаться не на что — это отказ.
3. В поле quotes для КАЖДОЙ использованной ссылки приведи выдержку, скопированную
   из фрагмента ПОБУКВЕННО. Её будут искать в тексте фрагмента автоматическим
   поиском подстроки. Любое перефразирование, сокращение, замена кавычек,
   исправление опечатки или склейка двух мест через многоточие — считаются
   ошибкой. Копируй одно непрерывное предложение целиком, как оно напечатано.
4. Если во фрагментах нет ответа на вопрос — поставь refused=true и объясни,
   чего именно не хватает. Не додумывай и не отвечай по общим знаниям.

Верни JSON:
{{"refused": true|false,
  "answer": "текст ответа со ссылками [[...]] или пустая строка при отказе",
  "reason": "чего не хватает — только при refused=true",
  "quotes": {{"идентификатор": "дословная выдержка", ...}}}}"""


@dataclass(slots=True)
class Fragment:
    citation: str
    breadcrumb: str
    text: str


@dataclass(slots=True)
class Answer:
    question: str
    refused: bool
    text: str
    reason: str = ""
    # Ошибка разбора — отдельный исход. Записывать её в отказы нельзя:
    # отказ это осмысленное поведение модели, а обрыв — сбой, и смешение
    # двух делает метрику отказов бессмысленной.
    error: str = ""
    quotes: dict[str, str] = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    seconds: float = 0.0
    tokens: tuple[int, int] = (0, 0)

    @property
    def citations_valid(self) -> bool:
        """Ответ пригоден, только если все ссылки в нём разрешились.

        Ответ без единой ссылки пригодным не считается: утверждение без
        опоры на текст — ровно то, чего задание требует не допускать.
        """
        if self.error:
            return False
        if self.refused:
            return True
        return self.verification.get("valid_share") == 1.0


def format_fragments(fragments: list[Fragment], max_chars: int = 2200) -> str:
    blocks = []
    for fragment in fragments:
        body = fragment.text[:max_chars]
        header = f"[{fragment.citation}]"
        if fragment.breadcrumb:
            header += f" {fragment.breadcrumb}"
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)


def answer_query(
    question: str,
    fragments: list[Fragment],
    resolver: CitationResolver,
    llm,
    *,
    as_of: str | None = None,
) -> Answer:
    """Собирает ответ и тут же проверяет каждую его ссылку."""
    note = (
        f"\nОтвет должен описывать состояние на дату {as_of}.\n" if as_of else "\n"
    )
    prompt = PROMPT.format(
        question=question, as_of_note=note, fragments=format_fragments(fragments)
    )
    started = time.time()
    parsed = None
    result = None
    # Первая попытка укладывается почти всегда; вторая — для длинных ответов.
    # Замер показал, что рассуждающая модель тратит в среднем 2 233 токена
    # на ответ, но хвост распределения уходит заметно дальше.
    for budget in (8000, 16000):
        result = llm.complete(SYSTEM, prompt, json_mode=True, max_tokens=budget)
        try:
            parsed = result.json()
            break
        except ValueError:
            continue

    if parsed is None:
        return Answer(
            question=question,
            refused=False,
            text="",
            error="модель вернула неразбираемый JSON даже при удвоенном бюджете",
            seconds=time.time() - started,
            tokens=(result.prompt_tokens, result.completion_tokens),
        )

    quotes = parsed.get("quotes") or {}
    if not isinstance(quotes, dict):
        quotes = {}
    text = parsed.get("answer") or ""
    refused = bool(parsed.get("refused"))

    answer = Answer(
        question=question,
        refused=refused,
        text=text,
        reason=parsed.get("reason") or "",
        quotes=quotes,
        seconds=time.time() - started,
        tokens=(result.prompt_tokens, result.completion_tokens),
    )
    if not refused:
        answer.verification = resolver.verify_answer(text, quotes)
    return answer


# --------------------------------------------------------------------------- #
# Прогон по корзине
# --------------------------------------------------------------------------- #


def build_fragments(hits, texts: dict[str, str], breadcrumbs: dict[str, str]) -> list[Fragment]:
    return [
        Fragment(
            citation=hit.chunk_id,
            breadcrumb=breadcrumbs.get(hit.chunk_id, ""),
            text=texts.get(hit.chunk_id, ""),
        )
        for hit in hits
    ]


def load_breadcrumbs(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["chunk_id"]] = row.get("breadcrumb") or ""
    return out


def main() -> None:
    from mlaw.evaluate import build_retrievers, load_basket
    from mlaw.llm import DeepSeek
    from mlaw.search import SearchConfig, SearchPipeline, Reranker, load_chunk_texts

    parser = argparse.ArgumentParser(description="Ответы с проверяемыми цитатами")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--fragments", type=int, default=6)
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--collection", default="mlaw_voyage_1024_raw")
    parser.add_argument("--model", default="voyage")
    parser.add_argument("--llm", default="deepseek-v4-pro")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    queries = load_basket(Path(f"queries/{args.split}.jsonl"))[: args.limit]

    # Контроль отказа: предмета заведомо нет в корпусе московских актов.
    queries.append(
        {
            "query_id": "control-refusal",
            "type": "control",
            "query": "Каков размер пошлины за регистрацию патента на изобретение в Японии?",
            "gold": {"act_id": -1, "doc_id": None, "chunk_ids": []},
        }
    )

    retrievers = build_retrievers(
        args.collection, 1024, args.model, "http://localhost:6333", Path("data/bm25")
    )
    retrievers["dense"].encode([q["query"] for q in queries])
    texts = load_chunk_texts(args.chunks)
    breadcrumbs = load_breadcrumbs(args.chunks)
    reranker = None if args.no_rerank else Reranker()
    pipeline = SearchPipeline(
        retrievers["dense"], retrievers["lexical"], texts, reranker,
        # Для ответа полное схлопывание по акту вредно: абляция показала,
        # что оно поднимает поиск акта (0.941 -> 0.961), но роняет поиск
        # нужного фрагмента (0.562 -> 0.438). Ответу нужны доказательные
        # фрагменты, поэтому здесь только потолок 2 на акт.
        SearchConfig(final_k=args.fragments, dedup_by_act=False,
                     per_act_before_rerank=2),
    )
    llm = DeepSeek(model=args.llm)

    rows = []
    with CitationResolver(Path("data/slice.jsonl"), Path("data/slice.oix")) as resolver:
        for query in queries:
            found = pipeline.search(query["query"], query.get("as_of"))
            fragments = build_fragments(found.hits, texts, breadcrumbs)
            answer = answer_query(
                query["query"], fragments, resolver, llm, as_of=query.get("as_of")
            )
            rows.append(
                {
                    "query_id": query["query_id"],
                    "type": query["type"],
                    "question": query["query"],
                    "refused": answer.refused,
                    "error": answer.error,
                    "reason": answer.reason,
                    "answer": answer.text,
                    "verification": answer.verification,
                    "gold_act_id": query["gold"]["act_id"],
                    "retrieved_acts": [h.act_id for h in found.hits],
                    "seconds": round(answer.seconds, 1),
                }
            )
            mark = (
                "СБОЙ" if answer.error
                else "ОТКАЗ" if answer.refused
                else "ok" if answer.citations_valid
                else "ЦИТАТЫ НЕ СХОДЯТСЯ"
            )
            v = answer.verification
            # Считаем по УНИКАЛЬНЫМ ссылкам: одна и та же ссылка, повторённая
            # трижды в тексте ответа, — это одна проверка, а не три.
            print(f"  [{mark:>18}] {query['query_id']:<22} "
                  f"ссылок {v.get('unique_citations', 0)}, "
                  f"резолвятся {v.get('resolvable', 0)}, "
                  f"дословны {v.get('quote_verbatim', 0)}")

    errors = [r for r in rows if r.get("error")]
    answered = [r for r in rows if not r["refused"] and not r.get("error")]
    total_citations = sum(r["verification"].get("unique_citations", 0) for r in answered)
    resolvable = sum(r["verification"].get("resolvable", 0) for r in answered)
    valid_citations = sum(r["verification"].get("quote_verbatim", 0) for r in answered)
    no_citations = sum(
        1 for r in answered if r["verification"].get("unique_citations", 0) == 0
    )
    control = next((r for r in rows if r["query_id"] == "control-refusal"), None)

    summary = {
        "queries": len(rows),
        "answered": len(answered),
        "refused": sum(1 for r in rows if r["refused"]),
        "errors": len(errors),
        "citations_unique": total_citations,
        "citations_resolvable": resolvable,
        "citations_resolvable_share": round(resolvable / total_citations, 4)
        if total_citations else None,
        "citations_verbatim": valid_citations,
        "citations_verbatim_share": round(valid_citations / total_citations, 4)
        if total_citations else None,
        "answers_without_citations": no_citations,
        "answers_with_all_citations_valid": sum(
            1 for r in answered
            if r["verification"].get("valid_share") == 1.0
        ),
        "control_refusal_worked": bool(control and control["refused"]),
        "llm_tokens": [llm.prompt_tokens, llm.completion_tokens],
        "rows": rows,
    }

    print(f"\n{'=' * 66}")
    print(f"  Ответов {summary['answered']}, отказов {summary['refused']}, "
          f"сбоев разбора {summary['errors']}")
    print(f"  Уникальных ссылок {total_citations}: "
          f"резолвятся {resolvable} ({summary['citations_resolvable_share']}), "
          f"дословны {valid_citations} ({summary['citations_verbatim_share']})")
    print(f"  Ответов без единой ссылки: {no_citations}")
    print(f"  Ответов, где ВСЕ ссылки валидны: "
          f"{summary['answers_with_all_citations_valid']}/{len(answered)}")
    print(f"  Контроль отказа сработал: {summary['control_refusal_worked']}")
    print(f"{'=' * 66}")

    out = args.out or Path(f"reports/answers_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
