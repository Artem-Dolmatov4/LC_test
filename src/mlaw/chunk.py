"""Шаг 3 — нарезка.

Политика выведена из замеров по действующим редакциям, а не из привычки:

* дерево `contents` есть у 41.4 % документов, но они держат **85.26 %**
  текстовой массы — значит структурная нарезка покрывает почти весь текст,
  который реально идёт в индекс;
* у документов без дерева абзацы крошечные (p50 = 55, p99 = 852 знака),
  упаковка абзацев там не рвёт ничего;
* а вот сегменты, порождённые границами `contents`, крупные: p50 = 1 077,
  p90 = 5 953, p99 = 32 682, max = 15 699 600. При окне 2 000 знаков
  доразбивать приходится сегменты, в которых лежит **86.1 % массы**.

Отсюда три уровня, и третий — не формальность, а то, что решает судьбу
большей части текста.

1. **Границы сегментов `contents` — жёсткие.** Чанк не пересекает границу.
   Склеивать разрешено только соседей с общим родителем: два приложения
   рядом не сливаются, а пункты одной статьи — сливаются.
2. **Внутри сегмента — упаковка абзацев** до целевого окна, каждому чанку
   приписывается «хлебная крошка» из пути меток.
3. **Абзац крупнее окна** (редко, но бывает — max 33 646 знаков) разбивается
   по границам предложений, а если и предложение не влезает — по окну
   с перекрытием. Доля таких разрезов измеряется и сообщается, а не
   объявляется нулевой.

Две ловушки дерева, каждая проверена тестом:

* `char_end` — конец **поддерева**, а не узла: «Статья» простирается до конца
  всех своих пунктов. Собственные границы восстанавливаются из соседних
  `char_start`, а не берутся из `char_end`.
* узлы нулевого размаха (0.92 % узлов) не выбрасываются: они адресованы верно,
  и отбрасывание выпотрошило бы оглавление у части документов.

    python -m mlaw.chunk --slice data/slice.jsonl --target 2000
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "ChunkConfig",
    "Chunk",
    "Segment",
    "derive_segments",
    "chunk_document",
]

# Управляющий символ, оставшийся в метках оглавления вместо «§». На этой
# выгрузке не встретился ни разу (0 из 2 287 787 узлов) — нормализация
# оставлена как безвредная страховка, но чинить ей нечего.
LABEL_CONTROL = "\x15"

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    target_chars: int = 2000
    # Жёсткий потолок: чанк длиннее не выпускается никогда.
    max_chars: int = 3000
    # Куски короче склеиваются с соседом — но только внутри своего сегмента.
    min_chars: int = 200
    # Перекрытие только на аварийном пути, внутри одного абзаца.
    overlap_chars: int = 200

    def __post_init__(self) -> None:
        if self.max_chars < self.target_chars:
            raise ValueError("max_chars не может быть меньше target_chars")
        if self.overlap_chars >= self.target_chars:
            raise ValueError("перекрытие должно быть меньше окна")


@dataclass(frozen=True, slots=True)
class Segment:
    """Атомарный кусок текста, порождённый границами оглавления."""

    start: int
    end: int
    breadcrumb: tuple[str, ...]
    parent_key: int
    node_index: int | None

    @property
    def chars(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class Chunk:
    doc_id: int
    act_id: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    breadcrumb: str
    boundary_kind: str  # structural | paragraph | mid_paragraph
    segments_merged: int
    has_structure: bool

    def payload(self, record: dict) -> dict:
        """Чанк вместе с полями, по которым потом фильтруется выдача."""
        out = asdict(self)
        out["chunk_id"] = f"{self.doc_id}:{self.char_start}-{self.char_end}"
        out["n_chars"] = self.char_end - self.char_start
        edition = record.get("edition") or {}
        out.update(
            {
                "title": record.get("title"),
                "doc_type": record.get("doc_type"),
                "status": record.get("status"),
                "date": record.get("date"),
                "is_current": bool(edition.get("is_current")),
                "edition_index": edition.get("index"),
                "n_editions": edition.get("n_editions"),
                "effective_date_begin": record.get("effective_date_begin"),
                "effective_date_end": record.get("effective_date_end"),
                "effective_date_end_sentinel": record.get("effective_date_end_sentinel"),
                "rubrics_code": record.get("rubrics_code"),
            }
        )
        return out


# --------------------------------------------------------------------------- #
# Сегментация по дереву
# --------------------------------------------------------------------------- #


def _clean_label(label: str | None) -> str:
    return (label or "").replace(LABEL_CONTROL, "§").strip()


def _breadcrumb_paths(contents: list[dict]) -> list[tuple[str, ...]]:
    """Путь меток от корня до каждого узла.

    `parent` — индекс в этом же списке (ближайший предыдущий узел меньшего
    уровня), а не идентификатор. Многокорневых документов 100 %, поэтому
    `parent is None` — норма, а не признак единственного корня.
    """
    paths: list[tuple[str, ...]] = []
    for node in contents:
        parent = node.get("parent")
        label = _clean_label(node.get("label"))
        if parent is None or not (0 <= parent < len(paths)):
            paths.append((label,) if label else ())
        else:
            paths.append(paths[parent] + ((label,) if label else ()))
    return paths


def derive_segments(text: str, contents: list[dict] | None) -> list[Segment]:
    """Разбивает текст на атомарные сегменты по границам оглавления.

    Границы берутся из `char_start` соседних узлов, а НЕ из `char_end`:
    `char_end` — конец поддерева, и по нему «Статья» перекрыла бы все свои
    пункты, породив вложенные, а не соседние сегменты.
    """
    if not contents:
        return [Segment(0, len(text), (), -1, None)]

    paths = _breadcrumb_paths(contents)

    # Для каждой позиции — самый глубокий узел, который с неё начинается:
    # он даёт самую точную метку. Узлы нулевого размаха тут и участвуют.
    deepest: dict[int, int] = {}
    for index, node in enumerate(contents):
        start = max(0, min(int(node["char_start"]), len(text)))
        current = deepest.get(start)
        if current is None or node["level"] >= contents[current]["level"]:
            deepest[start] = index

    boundaries = sorted({0, len(text)} | set(deepest))
    segments: list[Segment] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end <= start:
            continue
        node_index = deepest.get(start)
        if node_index is None:
            # Текст до первого узла оглавления — преамбула, реквизиты акта.
            segments.append(Segment(start, end, (), -1, None))
            continue
        parent = contents[node_index].get("parent")
        segments.append(
            Segment(
                start=start,
                end=end,
                breadcrumb=paths[node_index],
                parent_key=parent if parent is not None else -(node_index + 2),
                node_index=node_index,
            )
        )
    return segments


# --------------------------------------------------------------------------- #
# Упаковка
# --------------------------------------------------------------------------- #


def _split_paragraphs(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Границы абзацев внутри сегмента, в абсолютных офсетах документа."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for piece in text[start:end].split("\n"):
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece) + 1
    return [(a, b) for a, b in spans if b > a]


def _split_oversized(text: str, start: int, end: int, cfg: ChunkConfig) -> list[tuple[int, int]]:
    """Аварийный путь: абзац крупнее окна.

    Сначала по границам предложений; если и предложение не влезает — по окну
    с перекрытием. Только здесь разрез попадает внутрь абзаца, и только эти
    чанки помечаются как `mid_paragraph`.
    """
    # Границы предложений ищутся по позициям, а не через re.split: split
    # выбрасывает разделители, и офсеты кусков уехали бы влево на сумму
    # съеденных пробелов — а на них держится резолв цитат.
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        if match.start() > cursor:
            pieces.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        pieces.append((cursor, end))
    if not pieces:
        pieces = [(start, end)]

    out: list[tuple[int, int]] = []
    buffer_start: int | None = None
    buffer_end = 0
    for piece_start, piece_end in pieces:
        if piece_end - piece_start > cfg.max_chars:
            if buffer_start is not None:
                out.append((buffer_start, buffer_end))
                buffer_start = None
            step = cfg.target_chars - cfg.overlap_chars
            position = piece_start
            while position < piece_end:
                out.append((position, min(position + cfg.target_chars, piece_end)))
                position += step
            continue
        if buffer_start is None:
            buffer_start, buffer_end = piece_start, piece_end
        elif piece_end - buffer_start <= cfg.target_chars:
            buffer_end = piece_end
        else:
            out.append((buffer_start, buffer_end))
            buffer_start, buffer_end = piece_start, piece_end
    if buffer_start is not None:
        out.append((buffer_start, buffer_end))
    return out


def _pack_segment(
    text: str, segment: Segment, cfg: ChunkConfig
) -> list[tuple[int, int, str]]:
    """Упаковывает один сегмент в куски, не выходя за его границы."""
    if segment.chars <= cfg.max_chars:
        return [(segment.start, segment.end, "structural")]

    out: list[tuple[int, int, str]] = []
    buffer_start: int | None = None
    buffer_end = 0

    for para_start, para_end in _split_paragraphs(text, segment.start, segment.end):
        if para_end - para_start > cfg.max_chars:
            if buffer_start is not None:
                out.append((buffer_start, buffer_end, "paragraph"))
                buffer_start = None
            for piece_start, piece_end in _split_oversized(text, para_start, para_end, cfg):
                out.append((piece_start, piece_end, "mid_paragraph"))
            continue
        if buffer_start is None:
            buffer_start, buffer_end = para_start, para_end
        elif para_end - buffer_start <= cfg.target_chars:
            buffer_end = para_end
        else:
            out.append((buffer_start, buffer_end, "paragraph"))
            buffer_start, buffer_end = para_start, para_end

    if buffer_start is not None:
        out.append((buffer_start, buffer_end, "paragraph"))

    # Последний кусок сегмента заканчивается на настоящей структурной границе.
    if out and out[-1][1] >= segment.end - 1 and out[-1][2] == "paragraph":
        last = out[-1]
        out[-1] = (last[0], last[1], "structural")
    return out


def _merge_siblings(segments: list[Segment], cfg: ChunkConfig) -> list[list[Segment]]:
    """Склеивает соседние мелкие сегменты — но только с общим родителем.

    Без этого каждый мелкий узел стал бы отдельным чанком: медиана сегмента
    1 077 знаков, и таких огрызков было бы большинство. Ограничение по общему
    родителю сохраняет исходный смысл жёсткой границы — два соседних
    приложения не сливаются, а пункты одной статьи сливаются.
    """
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    current_chars = 0

    for segment in segments:
        same_parent = bool(current) and current[0].parent_key == segment.parent_key
        fits = current_chars + segment.chars <= cfg.target_chars
        if current and same_parent and fits and segment.chars < cfg.target_chars:
            current.append(segment)
            current_chars += segment.chars
        else:
            if current:
                groups.append(current)
            current = [segment]
            current_chars = segment.chars
    if current:
        groups.append(current)
    return groups


def chunk_document(record: dict, cfg: ChunkConfig) -> list[Chunk]:
    text = record.get("text") or ""
    if not text:
        return []
    contents = record.get("contents")
    edition = record.get("edition") or {}
    title = (record.get("title") or "").strip()

    segments = derive_segments(text, contents)
    chunks: list[Chunk] = []

    for group in _merge_siblings(segments, cfg):
        breadcrumb_parts = group[0].breadcrumb
        merged = len(group)
        if merged > 1:
            span_start, span_end = group[0].start, group[-1].end
            pieces = [(span_start, span_end, "structural")]
        else:
            pieces = _pack_segment(text, group[0], cfg)

        for piece_start, piece_end, kind in pieces:
            raw = text[piece_start:piece_end]
            body = raw.strip()
            if not body:
                continue
            # Офсеты подтягиваются к очищенному тексту, чтобы выполнялось
            # тождество text[char_start:char_end] == chunk.text. Без этого
            # проверка цитат в шаге 6 сравнивала бы разные строки.
            lead = len(raw) - len(raw.lstrip())
            tail = len(raw) - len(raw.rstrip())
            chunks.append(
                Chunk(
                    doc_id=record["doc_id"],
                    act_id=edition.get("act_id"),
                    chunk_index=len(chunks),
                    char_start=piece_start + lead,
                    char_end=piece_end - tail,
                    text=body,
                    breadcrumb=" › ".join(p for p in (title, *breadcrumb_parts) if p),
                    boundary_kind=kind,
                    segments_merged=merged,
                    has_structure=bool(contents),
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


@dataclass
class Stats:
    documents: int = 0
    chunks: int = 0
    chars_in: int = 0
    chars_out: int = 0
    boundary: Counter = field(default_factory=Counter)
    sizes: list[int] = field(default_factory=list)
    per_document: list[int] = field(default_factory=list)
    structured_docs: int = 0
    merged_groups: int = 0


def run(slice_path: Path, out_path: Path, report_path: Path, cfg: ChunkConfig) -> dict:
    stats = Stats()
    started = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(slice_path, encoding="utf-8") as source, open(
        out_path, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            record = json.loads(line)
            chunks = chunk_document(record, cfg)

            stats.documents += 1
            stats.chars_in += len(record.get("text") or "")
            stats.per_document.append(len(chunks))
            if record.get("contents"):
                stats.structured_docs += 1

            for chunk in chunks:
                stats.chunks += 1
                stats.chars_out += len(chunk.text)
                stats.boundary[chunk.boundary_kind] += 1
                stats.sizes.append(len(chunk.text))
                if chunk.segments_merged > 1:
                    stats.merged_groups += 1
                sink.write(
                    json.dumps(chunk.payload(record), ensure_ascii=False) + "\n"
                )

            if stats.documents % 200 == 0:
                print(f"  {stats.documents:>5} док, {stats.chunks:>7} чанков, "
                      f"{time.time() - started:5.1f} с")

    report = _summarise(stats, cfg, time.time() - started)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _summarise(stats: Stats, cfg: ChunkConfig, seconds: float) -> dict:
    sizes = sorted(stats.sizes)
    per_doc = sorted(stats.per_document)

    def at(values: list[int], p: float) -> int:
        return values[min(len(values) - 1, int(p * len(values)))] if values else 0

    total = stats.chunks or 1
    return {
        "config": asdict(cfg),
        "documents": stats.documents,
        "documents_with_structure": stats.structured_docs,
        "chunks": stats.chunks,
        "chars_in": stats.chars_in,
        "chars_in_chunks": stats.chars_out,
        "coverage_pct": round(100 * stats.chars_out / stats.chars_in, 2)
        if stats.chars_in
        else 0.0,
        "chunk_chars": {
            "p10": at(sizes, 0.1),
            "p50": at(sizes, 0.5),
            "p90": at(sizes, 0.9),
            "p99": at(sizes, 0.99),
            "max": sizes[-1] if sizes else 0,
            "mean": round(stats.chars_out / total),
        },
        "chunks_per_document": {
            "p50": at(per_doc, 0.5),
            "p90": at(per_doc, 0.9),
            "p99": at(per_doc, 0.99),
            "max": per_doc[-1] if per_doc else 0,
        },
        "boundary_kind": {
            kind: {"chunks": count, "pct": round(100 * count / total, 2)}
            for kind, count in stats.boundary.most_common()
        },
        "merged_sibling_groups": stats.merged_groups,
        "seconds": round(seconds, 1),
    }


def print_summary(r: dict) -> None:
    c = r["chunk_chars"]
    d = r["chunks_per_document"]
    print(f"\n{'=' * 68}")
    print(f"  Документов {r['documents']:,} (со структурой {r['documents_with_structure']:,}) "
          f"-> чанков {r['chunks']:,}".replace(",", " "))
    print(f"  Знаков на входе {r['chars_in']:,}, в чанках {r['chars_in_chunks']:,} "
          f"({r['coverage_pct']} % покрытия)".replace(",", " "))
    print()
    print(f"  Размер чанка: p10 {c['p10']} · p50 {c['p50']} · p90 {c['p90']} · "
          f"p99 {c['p99']} · max {c['max']} · среднее {c['mean']}")
    print(f"  Чанков на документ: p50 {d['p50']} · p90 {d['p90']} · p99 {d['p99']} · "
          f"max {d['max']:,}".replace(",", " "))
    print()
    print("  Где заканчивается чанк:")
    for kind, data in r["boundary_kind"].items():
        label = {
            "structural": "на границе сегмента оглавления",
            "paragraph": "на границе абзаца внутри сегмента",
            "mid_paragraph": "ВНУТРИ абзаца (аварийный путь)",
        }.get(kind, kind)
        print(f"    {label:<38} {data['chunks']:>8,} ({data['pct']:>5} %)".replace(",", " "))
    print(f"\n  Склеено групп соседних сегментов: {r['merged_sibling_groups']:,}"
          .replace(",", " "))
    print(f"{'=' * 68}\n  {r['seconds']} с")


def main() -> None:
    parser = argparse.ArgumentParser(description="Нарезка среза на чанки")
    parser.add_argument("--slice", type=Path, default=Path("data/slice.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/chunking.json"))
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    cfg = ChunkConfig(
        target_chars=args.target, max_chars=args.max_chars, overlap_chars=args.overlap
    )
    report = run(args.slice, args.out, args.report, cfg)
    print_summary(report)
    print(f"Чанки: {args.out} · отчёт: {args.report}")


if __name__ == "__main__":
    main()
