"""Шаг 0.5 — гейт по производительности эмбеддеров.

Смысл шага: не начинать индексацию, пока не измерено, сколько она стоит.
Размер среза выводится из пропускной способности, а не назначается.

Проверяется:

* сколько знаков приходится на токен на настоящем русском правовом тексте;
* реальная пропускная способность в чанках и токенах в секунду;
* размерность ответа и корректность MRL-усечения;
* что переполнение ``num_ctx`` даёт **явную ошибку**, а не молчаливое обрезание.

Результат пишется в ``reports/gate.json`` и попадает в отчёт: выбор
инфраструктуры — такое же решение, обоснованное числом, как и все остальные.

    python -m mlaw.gate --archive MLAW_dataset.tar.zst --chunks 8
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from mlaw.embed import OllamaEmbedder, OllamaOverflow
from mlaw.stream import open_corpus

CHUNK_CHARS = 2000


def sample_chunks(archive: str, count: int, *, shard: int = 0) -> list[str]:
    """Набирает различные фрагменты настоящего текста корпуса.

    Настоящий чанкер появится на шаге 3; для замера скорости достаточно
    суррогатных кусков фиксированной длины — важно лишь, чтобы это был
    реальный русский правовой текст, а не синтетика.
    """
    chunks: list[str] = []
    seen: set[str] = set()
    for record in open_corpus(archive).iter_lines(shards=[shard]):
        text = record.json().get("text") or ""
        if len(text) < 6 * CHUNK_CHARS:
            continue
        for start in range(0, len(text) - CHUNK_CHARS, CHUNK_CHARS):
            piece = text[start : start + CHUNK_CHARS]
            if len(piece) == CHUNK_CHARS and piece not in seen:
                seen.add(piece)
                chunks.append(piece)
            if len(chunks) >= count:
                return chunks
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def measure_overflow(model: str, host: str | None) -> dict:
    """Проверяет, что переполнение контекста — это ошибка, а не тишина.

    Контрольная половина теста обязательна: если бы ``truncate=True`` тоже
    падал, проверка ничего бы не доказывала.
    """
    small = OllamaEmbedder(model, host=host, num_ctx=512, dim=None)
    long_text = "Правительство Москвы постановляет. " * 400  # ~14 000 знаков

    result: dict = {"num_ctx": 512}
    try:
        small._call(long_text)
        result["truncate_false"] = "ПРОШЛО МОЛЧА — переполнение не детектируется"
        result["ok"] = False
    except OllamaOverflow as exc:
        result["truncate_false"] = f"явная ошибка: {str(exc)[:120]}"
        result["ok"] = True

    # Контроль: с truncate=true тот же вход обязан пройти и обрезаться.
    import urllib.request

    payload = {
        "model": model,
        "input": long_text,
        "truncate": True,
        "options": {"num_ctx": 512},
    }
    request = urllib.request.Request(
        f"{small.host}/api/embed",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read())
    result["truncate_true_tokens"] = body.get("prompt_eval_count")
    result["truncate_true"] = "прошло молча — ровно то, чего нельзя допускать"
    return result


def run(archive: str, model: str, host: str | None, count: int, out: Path) -> dict:
    # На один больше: первый уходит на прогрев и в замер не попадает.
    # Прогревать тем же фрагментом, что потом меряешь, нельзя — попадание
    # в кэш даёт 0.14 с вместо 3.3 с и завышает пропускную способность.
    print(f"Набираю {count + 1} фрагментов по {CHUNK_CHARS} знаков из шарда 0…")
    sampled = sample_chunks(archive, count + 1)
    warmup, chunks = sampled[0], sampled[1:]
    print(f"  набрано: {len(chunks)} измеряемых + 1 на прогрев")

    native = OllamaEmbedder(model, host=host, num_ctx=8192, dim=None)

    print("Прогрев (первый вызов платит за загрузку модели)…")
    warm_started = time.time()
    native._call(warmup)
    print(f"  {time.time() - warm_started:.1f} с")

    print("Замер по одному фрагменту…")
    per_chunk: list[float] = []
    tokens: list[int] = []
    vectors: list[list[float]] = []
    for i, chunk in enumerate(chunks, 1):
        started = time.time()
        vector, used = native._call(chunk)
        elapsed = time.time() - started
        per_chunk.append(elapsed)
        tokens.append(used)
        vectors.append(vector)
        print(f"  #{i}: {elapsed:5.2f} с, {used} токенов")

    total_time = sum(per_chunk)
    total_tokens = sum(tokens)
    chunks_per_sec = len(chunks) / total_time
    chars_per_token = (CHUNK_CHARS * len(chunks)) / total_tokens

    # MRL: усечение считается на уже полученных векторах, лишних вызовов нет.
    truncated = [_unit(v[:1024]) for v in vectors]
    full = [_unit(v) for v in vectors]
    pairs_full, pairs_cut = [], []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pairs_full.append(_cosine(full[i], full[j]))
            pairs_cut.append(_cosine(truncated[i], truncated[j]))
    mrl = {
        "native_dim": len(vectors[0]),
        "truncated_dim": 1024,
        "pearson_on_pairwise_cosine": round(_pearson(pairs_full, pairs_cut), 4),
        "mean_abs_shift": round(
            statistics.fmean(abs(a - b) for a, b in zip(pairs_full, pairs_cut, strict=True)), 4
        ),
        "note": (
            "усечение с перенормировкой; замер на суррогатных чанках и без "
            "эталонов релевантности — показывает сохранность геометрии, а не качество поиска"
        ),
    }

    print("Проверяю поведение при переполнении контекста…")
    overflow = measure_overflow(model, host)

    report = {
        "model": model,
        "chunk_chars": CHUNK_CHARS,
        "chunks_measured": len(chunks),
        "throughput": {
            "chunks_per_sec": round(chunks_per_sec, 3),
            "tokens_per_sec": round(total_tokens / total_time, 1),
            "sec_per_chunk_median": round(statistics.median(per_chunk), 2),
            "sec_per_chunk_min": round(min(per_chunk), 2),
            "sec_per_chunk_max": round(max(per_chunk), 2),
        },
        "tokenization": {
            "tokens_per_chunk_median": statistics.median(tokens),
            "chars_per_token": round(chars_per_token, 2),
        },
        "mrl": mrl,
        "overflow": overflow,
        "projection": {
            f"{n:_} чанков, часов": round(n / chunks_per_sec / 3600, 1)
            for n in (1_000, 10_000, 50_000, 100_000)
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_voyage(archive: str, count: int, out: Path) -> dict:
    """Гейт API-ноги.

    Главная проверка — что контекстуализация не декларативная: один и тот же
    чанк, посланный в одиночку и в составе документа, обязан получить **разные**
    векторы. Если они совпадут, вся ось эксперимента «нативный контекст против
    хлебной крошки» лишена смысла, и это надо знать до, а не после индексации.
    """
    from mlaw.embed import VoyageEmbedder

    print(f"Voyage: набираю {count} фрагментов…")
    chunks = sample_chunks(archive, count)
    embedder = VoyageEmbedder(dim=1024)

    # Документ целиком — чанки идут одной группой, как и требует модель.
    started = time.time()
    as_document = embedder.embed_documents([chunks])
    elapsed = time.time() - started

    # Тот же средний чанк, но в одиночку — контекста нет.
    middle = len(chunks) // 2
    alone = embedder.embed_documents([[chunks[middle]]])

    contextual = _unit(as_document.vectors[middle])
    isolated = _unit(alone.vectors[0])
    similarity = _cosine(contextual, isolated)

    # Контроль: два разных чанка должны отличаться сильнее, чем один и тот же
    # чанк в двух режимах. Иначе «разница» — просто шум.
    other = _unit(as_document.vectors[(middle + 1) % len(chunks)])
    baseline = _cosine(contextual, other)

    report = {
        "model": embedder.model,
        "chunks": len(chunks),
        "dim": as_document.dim,
        "throughput": {
            "chunks_per_sec": round(len(chunks) / elapsed, 1),
            "seconds": round(elapsed, 2),
        },
        "usage": {
            "tokens_this_run": as_document.prompt_tokens + alone.prompt_tokens,
            "note": "200M токенов на аккаунт выдаются один раз и не возобновляются",
        },
        "contextualization": {
            "same_chunk_in_document_vs_alone": round(similarity, 4),
            "different_chunks_baseline": round(baseline, 4),
            "verdict": (
                "контекст влияет" if similarity < 0.999 else "ВЛИЯНИЯ НЕ ВИДНО"
            ),
        },
        "windowed_documents": embedder.windowed_documents,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def _pearson(a: list[float], b: list[float]) -> float:
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Гейт по производительности эмбеддера")
    parser.add_argument("--archive", default="MLAW_dataset.tar.zst")
    parser.add_argument("--model", default="qwen3-embedding:8b")
    parser.add_argument("--host", default=None)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("reports/gate.json"))
    parser.add_argument(
        "--voyage", action="store_true", help="гейт API-ноги вместо локальной"
    )
    args = parser.parse_args()

    if args.voyage:
        report = run_voyage(args.archive, args.chunks, Path("reports/gate_voyage.json"))
        c = report["contextualization"]
        print(f"\n{'=' * 58}")
        print(f"  {report['model']} · dim {report['dim']}")
        print(f"  {report['throughput']['chunks_per_sec']} чанк/с")
        print(f"  токенов истрачено: {report['usage']['tokens_this_run']}")
        print(f"  тот же чанк в документе и в одиночку: cos {c['same_chunk_in_document_vs_alone']}")
        print(f"  контроль, разные чанки:               cos {c['different_chunks_baseline']}")
        print(f"  вердикт: {c['verdict']}")
        print(f"{'=' * 58}\nЗаписано в reports/gate_voyage.json")
        return

    report = run(args.archive, args.model, args.host, args.chunks, args.out)

    t = report["throughput"]
    print(f"\n{'=' * 58}")
    print(f"  {t['chunks_per_sec']} чанк/с · {t['tokens_per_sec']} токен/с")
    print(f"  медиана {t['sec_per_chunk_median']} с на чанк")
    print(f"  {report['tokenization']['chars_per_token']} знака на токен")
    print(f"  MRL 4096->1024: pearson {report['mrl']['pearson_on_pairwise_cosine']}")
    print(f"  переполнение детектируется: {report['overflow']['ok']}")
    print("  прогноз:")
    for key, hours in report["projection"].items():
        print(f"    {key}: {hours}")
    print(f"{'=' * 58}\nЗаписано в {args.out}")


if __name__ == "__main__":
    main()
