"""Шаг 6 (часть 1) — цитаты, разрешимые обратно в текст.

Ссылка имеет вид ``doc_id:char_start-char_end`` и резолвится в исходный текст
редакции через собственный сайдкар среза: `slice.oix` даёт офсет строки
в `slice.jsonl`, дальше seek и срез по символам. Тот же формат, что у
оригинальных шардов, и тот же код чтения — сайдкар писался ради этого.

Проверка ссылки — программная, а не на доверии к модели. Ссылка считается
валидной, только если выполняются все пять условий:

1. ссылка разобралась в тройку ``doc_id:char_start-char_end``;
2. `doc_id` есть в срезе;
3. диапазон лежит в границах текста;
4. диапазон непустой;
5. модель привела выдержку, и она **действительно совпадает** с текстом
   по этому диапазону.

Пятое условие возможно потому, что нарезка гарантирует тождество
``text[char_start:char_end] == chunk.text`` — оно проверено на всех
64 031 чанке среза без единого исключения.

Первое и пятое условия добавлены после того, как прогон показал два способа
пройти проверку, ничего не подтвердив: ссылка вида ``[[103271]]`` (без
диапазона) не попадала в подсчёт вовсе, а ссылка без записи в ``quotes``
засчитывалась дословной, потому что сверять было нечего. Оба случая теперь
считаются провалом и стоят в знаменателе.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from mlaw.oix import OixIndex, read_line

__all__ = [
    "Citation",
    "CitationResolver",
    "parse_citations",
    "parse_malformed",
    "CITATION_RE",
    "BRACKET_RE",
]

# Ссылка в тексте ответа: [[123456:1000-2500]]
CITATION_RE = re.compile(r"\[\[(\d+):(\d+)-(\d+)\]\]")
# Любая пара двойных скобок, включая испорченные. Нужна, чтобы отличить
# «ссылки нет» от «ссылка есть, но её формат сломан»: без этого ссылка
# вида [[103271]] (только doc_id, без диапазона) не резолвится и не
# проваливается — она просто не попадает в подсчёт, и ответ проходит
# проверку с пустым списком ошибок.
BRACKET_RE = re.compile(r"\[\[([^\[\]]*)\]\]")


@dataclass(slots=True)
class Citation:
    doc_id: int
    char_start: int
    char_end: int

    @property
    def key(self) -> str:
        return f"{self.doc_id}:{self.char_start}-{self.char_end}"

    def __str__(self) -> str:
        return f"[[{self.key}]]"


@dataclass(slots=True)
class CitationCheck:
    citation: Citation
    resolved: bool
    reason: str = ""
    text: str = ""

    @property
    def ok(self) -> bool:
        return self.resolved


def parse_citations(text: str) -> list[Citation]:
    """Достаёт все правильно оформленные ссылки из ответа модели."""
    return [
        Citation(int(doc_id), int(start), int(end))
        for doc_id, start, end in CITATION_RE.findall(text)
    ]


def parse_malformed(text: str) -> list[str]:
    """Содержимое скобок, не разобравшееся в `doc_id:start-end`.

    Испорченная ссылка — это не «ссылки нет». Модель на что-то сослалась,
    но проверить это «что-то» нельзя, и засчитывать такой ответ как
    подкреплённый нельзя тем более. Возвращается сырое содержимое скобок
    в порядке появления, без повторов.
    """
    seen: dict[str, None] = {}
    for raw in BRACKET_RE.findall(text):
        if not CITATION_RE.fullmatch(f"[[{raw}]]"):
            seen.setdefault(raw.strip(), None)
    return list(seen)


class CitationResolver:
    """Произвольный доступ к тексту среза по `doc_id` и символьному диапазону.

    Тексты не держатся в памяти: срез — 202 МБ, и грузить его целиком ради
    нескольких цитат незачем. Читается ровно та строка, которая нужна,
    по офсету из сайдкара.
    """

    def __init__(self, slice_path: Path, oix_path: Path):
        self.slice_path = Path(slice_path)
        self.index = OixIndex.load(oix_path)
        self._handle = None
        self._text_cache: dict[int, str] = {}

    def __enter__(self) -> CitationResolver:
        self._handle = open(self.slice_path, "rb")
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _text(self, doc_id: int) -> str | None:
        if doc_id in self._text_cache:
            return self._text_cache[doc_id]
        entry = self.index.get(doc_id)
        if entry is None or self._handle is None:
            return None
        record = json.loads(read_line(self._handle, entry))
        text = record.get("text") or ""
        # Кэш маленький намеренно: цитаты в одном ответе обычно приходят
        # из двух-трёх документов, а тексты бывают по 12 млн знаков.
        if len(self._text_cache) > 8:
            self._text_cache.clear()
        self._text_cache[doc_id] = text
        return text

    def resolve(self, citation: Citation) -> str | None:
        """Текст по ссылке или None, если ссылка не разрешается."""
        text = self._text(citation.doc_id)
        if text is None:
            return None
        if not (0 <= citation.char_start < citation.char_end <= len(text)):
            return None
        return text[citation.char_start : citation.char_end]

    def check(self, citation: Citation, quoted: str | None = None) -> CitationCheck:
        """Проверяет ссылку, и при наличии цитаты — её дословность."""
        text = self._text(citation.doc_id)
        if text is None:
            return CitationCheck(citation, False, "нет такого doc_id в срезе")
        if citation.char_start >= citation.char_end:
            return CitationCheck(citation, False, "пустой или вывернутый диапазон")
        if citation.char_end > len(text):
            return CitationCheck(
                citation, False, f"диапазон за границей текста ({len(text)} знаков)"
            )
        fragment = text[citation.char_start : citation.char_end]
        if quoted is not None:
            if _normalise(quoted) not in _normalise(fragment):
                return CitationCheck(
                    citation, False, "процитированный текст не найден в диапазоне", fragment
                )
        return CitationCheck(citation, True, "", fragment)

    def verify_answer(self, answer: str, quotes: dict[str, str] | None = None) -> dict:
        """Проверяет все ссылки ответа разом.

        Возвращает и сводку, и разбор по каждой ссылке: доля валидных ссылок —
        отчётное число, а причины отказов нужны, чтобы понимать, что именно
        ломается.

        Знаменатель — **все** ссылки ответа, включая испорченные по формату.
        Считать долю только от разобравшихся значило бы вычитать собственные
        промахи из знаменателя: ответ с единственной ссылкой `[[103271]]`
        показывал бы «ошибок нет» вместо «ни одно утверждение не подкреплено».
        """
        citations = parse_citations(answer)
        unique = list({c.key: c for c in citations}.values())
        malformed = parse_malformed(answer)

        # Два разных свойства, и смешивать их в одно число нельзя.
        # «Резолвится» — ссылка указывает на существующий диапазон реального
        # документа; это свойство самой ссылки. «Дословна» — модель привела
        # выдержку, буквально совпадающую с текстом по этому диапазону;
        # это свойство добросовестности цитирования. Первое может быть
        # стопроцентным при провальном втором, и наоборот.
        resolvable = [self.check(c, None) for c in unique]

        # Ключи выдержек приводятся к общей форме: модель ключует их так,
        # как видит заголовок фрагмента, то есть в одинарных скобках.
        by_key = {_normalise_key(k): v for k, v in (quotes or {}).items()}

        verbatim: list[CitationCheck] = []
        for citation in unique:
            quoted = by_key.get(_normalise_key(citation.key))
            if quoted is None:
                # Ссылка без выдержки — это не «проверять нечего», а провал:
                # правило промпта требует выдержку для КАЖДОЙ ссылки, и без
                # неё дословность не подтверждена ничем.
                verbatim.append(
                    CitationCheck(citation, False, "модель не привела выдержку")
                )
                continue
            verbatim.append(self.check(citation, quoted))

        ok_resolve = sum(1 for c in resolvable if c.ok)
        ok_quote = sum(1 for c in verbatim if c.ok)
        # Испорченные ссылки не резолвятся и не дословны по определению,
        # но в знаменателе стоят наравне с остальными.
        total = len(unique) + len(malformed)

        failures = [
            {"citation": c.citation.key, "reason": c.reason}
            for c in verbatim
            if not c.ok
        ]
        failures += [
            {"citation": raw, "reason": "ссылка не разобралась в doc_id:start-end"}
            for raw in malformed
        ]

        return {
            "citations": len(citations),
            "unique_citations": len(unique),
            "malformed_citations": len(malformed),
            "references_total": total,
            "resolvable": ok_resolve,
            "resolvable_share": round(ok_resolve / total, 4) if total else None,
            "quote_verbatim": ok_quote,
            "quote_verbatim_share": round(ok_quote / total, 4) if total else None,
            "valid": ok_quote,
            "valid_share": round(ok_quote / total, 4) if total else None,
            "failures": failures,
        }


def _normalise(text: str) -> str:
    """Схлопывает пробелы: перенос строки в цитате не должен ломать сверку."""
    return re.sub(r"\s+", " ", text).strip()


def _normalise_key(key: str) -> str:
    """Ключ выдержки без обрамляющих скобок и пробелов.

    Фрагменты показываются модели заголовком ``[13880:2701-4455]``, а ссылка
    ставится как ``[[13880:2701-4455]]`` — и модель, вполне разумно, ключует
    выдержки той формой, которую видит в заголовке. Скобки здесь разделитель,
    а не часть идентификатора, поэтому сверять надо очищенные ключи.

    Это не послабление проверки: дословность по-прежнему требует, чтобы текст
    выдержки буквально нашёлся в диапазоне. Без нормализации провалом
    засчитывалось бы расхождение форматов ключа, то есть аккуратность нашего
    сопоставления, а не добросовестность модели.
    """
    return key.strip().strip("[]").strip()
