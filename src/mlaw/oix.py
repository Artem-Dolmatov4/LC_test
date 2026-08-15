"""Чтение и запись сайдкара `.oix` — бинарного индекса офсетов рядом с шардом JSONL.

Формат: 28 байт на запись, в порядке следования строк в шарде,
``struct "<QIIB3xQ"`` (little-endian, выравнивание отключено):

===========  ======  =====================================================
поле         тип     смысл
===========  ======  =====================================================
offset       u64     байтовый офсет строки внутри своего шарда
length       u32     длина строки, **включая** завершающий ``\\n``
doc_id       u32     идентификатор документа
flags        u8      бит0 is_stub · бит1 contents · бит2 is_current
                     бит3 duplicate_of · бит4 digest (не использовать)
(padding)    3B      следствие выравнивания struct — не данные
text_hash    u64     первые 64 бита sha256 очищенного текста; 0 = текста нет
===========  ======  =====================================================

Три ловушки, ради которых этот модуль существует отдельно и покрыт тестами:

1. **offset обязан быть u64.** На этой выгрузке максимальный ``offset + length``
   шарда 0000 равен 501 970 395 — то есть сегодня влезает в u32. Это не повод
   писать u32: запас до 2**32 у соседних банков того же корпуса почти исчерпан.
2. **``length`` включает перевод строки**, а почти всякий читатель ожидает длину
   без него. Здесь это закреплено в двух отдельных свойствах — ``length``
   (сырое поле) и ``length_without_newline`` — и проверено тестами с обеих сторон.
3. **``doc_id`` в сайдкаре НЕ отсортирован.** Замер на шарде 0000: 458 инверсий
   на 9 522 пары. Поиск по ``doc_id`` — только через хеш-таблицу
   (:class:`OixIndex`), никакого ``bisect``.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

__all__ = [
    "ENTRY_STRUCT",
    "ENTRY_SIZE",
    "OixEntry",
    "OixIndex",
    "OixViolation",
    "iter_oix",
    "read_oix",
    "write_oix",
    "verify_oix",
    "read_line",
]

ENTRY_STRUCT = struct.Struct("<QIIB3xQ")
ENTRY_SIZE = ENTRY_STRUCT.size  # 28

# Границы полей — используются при записи, чтобы переполнение падало явно,
# а не сворачивалось по модулю где-то в недрах struct.
_U64_MAX = 2**64 - 1
_U32_MAX = 2**32 - 1
_U8_MAX = 2**8 - 1

# Маски битов флагов.
FLAG_IS_STUB = 1 << 0
FLAG_HAS_CONTENTS = 1 << 1
FLAG_IS_CURRENT = 1 << 2
FLAG_DUPLICATE_OF = 1 << 3
FLAG_DIGEST = 1 << 4  # документацией помечен как «не использовать»


@dataclass(frozen=True, slots=True)
class OixEntry:
    """Одна запись сайдкара."""

    offset: int
    length: int
    doc_id: int
    flags: int
    text_hash: int

    @property
    def end(self) -> int:
        """Офсет первого байта за строкой (он же offset следующей записи)."""
        return self.offset + self.length

    @property
    def length_without_newline(self) -> int:
        """Длина строки без завершающего ``\\n`` — то, что обычно и нужно."""
        return self.length - 1

    @property
    def is_stub(self) -> bool:
        return bool(self.flags & FLAG_IS_STUB)

    @property
    def has_contents(self) -> bool:
        return bool(self.flags & FLAG_HAS_CONTENTS)

    @property
    def is_current(self) -> bool:
        """Помечена ли редакция как текущая в своей цепочке.

        Внимание: это НЕ «акт действует». У отменённого акта последняя редакция
        тоже несёт этот флаг — правовой статус живёт в поле ``status``.
        """
        return bool(self.flags & FLAG_IS_CURRENT)

    @property
    def has_text(self) -> bool:
        return self.text_hash != 0

    def pack(self) -> bytes:
        _check_field("offset", self.offset, _U64_MAX)
        _check_field("length", self.length, _U32_MAX)
        _check_field("doc_id", self.doc_id, _U32_MAX)
        _check_field("flags", self.flags, _U8_MAX)
        _check_field("text_hash", self.text_hash, _U64_MAX)
        return ENTRY_STRUCT.pack(
            self.offset, self.length, self.doc_id, self.flags, self.text_hash
        )

    @classmethod
    def unpack(cls, buf: bytes | bytearray | memoryview) -> OixEntry:
        return cls(*ENTRY_STRUCT.unpack(buf))


def _check_field(name: str, value: int, limit: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} должен быть int, получен {type(value).__name__}")
    if value < 0 or value > limit:
        raise ValueError(
            f"{name}={value} не помещается в поле сайдкара (допустимо 0..{limit})"
        )


# --------------------------------------------------------------------------- #
# Чтение
# --------------------------------------------------------------------------- #


def iter_oix(source: Path | str | BinaryIO) -> Iterator[OixEntry]:
    """Потоково читает сайдкар, не поднимая его целиком в память.

    Принимает путь либо уже открытый бинарный поток — чтобы можно было читать
    сайдкар прямо из архива, не материализуя его на диске.
    """
    if isinstance(source, (str, Path)):
        with open(source, "rb") as fh:
            yield from _iter_stream(fh, name=str(source))
    else:
        yield from _iter_stream(source, name=getattr(source, "name", "<stream>"))


def _iter_stream(fh: BinaryIO, *, name: str) -> Iterator[OixEntry]:
    # Читаем крупными блоками, кратными размеру записи: побайтовые read()
    # на 266 644-байтовом файле стоят дороже самого разбора.
    block = ENTRY_SIZE * 4096
    tail = b""
    while True:
        chunk = fh.read(block)
        if not chunk:
            break
        if tail:
            chunk = tail + chunk
        usable = len(chunk) - len(chunk) % ENTRY_SIZE
        tail = chunk[usable:]
        for values in ENTRY_STRUCT.iter_unpack(chunk[:usable]):
            yield OixEntry(*values)
    if tail:
        raise ValueError(
            f"{name}: длина не кратна {ENTRY_SIZE} Б — остаток {len(tail)} Б. "
            "Файл обрезан или это не сайдкар .oix"
        )


def read_oix(source: Path | str | BinaryIO) -> list[OixEntry]:
    """Читает сайдкар целиком. 9 523 записи на шард — это 266 644 Б, безопасно."""
    return list(iter_oix(source))


# --------------------------------------------------------------------------- #
# Запись
# --------------------------------------------------------------------------- #


def write_oix(path: Path | str, entries: Iterable[OixEntry]) -> int:
    """Пишет сайдкар. Возвращает число записанных записей.

    Нужна для собственного среза: `slice.jsonl` получает `slice.oix` в том же
    формате, что и оригинальные шарды, и проходит те же проверки.
    """
    count = 0
    with open(path, "wb") as fh:
        for entry in entries:
            fh.write(entry.pack())
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Произвольный доступ
# --------------------------------------------------------------------------- #


class OixIndex:
    """Отображение ``doc_id -> OixEntry`` для произвольного доступа к шарду.

    Именно хеш-таблица, а не отсортированный массив: ``doc_id`` в сайдкаре идёт
    в порядке записи и не монотонен (замер на шарде 0000 — 458 инверсий).
    """

    __slots__ = ("_by_doc_id", "_entries")

    def __init__(self, entries: Iterable[OixEntry]):
        self._entries: list[OixEntry] = list(entries)
        self._by_doc_id: dict[int, OixEntry] = {}
        for entry in self._entries:
            if entry.doc_id in self._by_doc_id:
                raise ValueError(f"doc_id {entry.doc_id} встречается в сайдкаре дважды")
            self._by_doc_id[entry.doc_id] = entry

    @classmethod
    def load(cls, source: Path | str | BinaryIO) -> OixIndex:
        return cls(iter_oix(source))

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[OixEntry]:
        return iter(self._entries)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._by_doc_id

    def __getitem__(self, doc_id: int) -> OixEntry:
        return self._by_doc_id[doc_id]

    def get(self, doc_id: int) -> OixEntry | None:
        return self._by_doc_id.get(doc_id)

    @property
    def entries(self) -> list[OixEntry]:
        """Записи в порядке следования строк в шарде."""
        return self._entries

    def doc_ids(self) -> list[int]:
        return [e.doc_id for e in self._entries]

    def current_entries(self) -> list[OixEntry]:
        """Записи с флагом is_current.

        Позволяет отобрать действующие редакции по 7.47 МБ сайдкаров, не читая
        16.9 ГБ JSON: сумма бита 2 по всем 28 шардам равна 207 634 — ровно числу
        актов в банке.
        """
        return [e for e in self._entries if e.is_current]

    def total_bytes(self) -> int:
        """Суммарная длина строк — должна совпадать с размером шарда."""
        return sum(e.length for e in self._entries)


def read_line(fh: BinaryIO, entry: OixEntry, *, keep_newline: bool = False) -> bytes:
    """Читает строку шарда по записи сайдкара.

    По умолчанию перевод строки отбрасывается: ``entry.length`` его включает,
    а вызывающему коду он почти никогда не нужен.
    """
    fh.seek(entry.offset)
    raw = fh.read(entry.length)
    if len(raw) != entry.length:
        raise ValueError(
            f"doc_id {entry.doc_id}: прочитано {len(raw)} Б вместо {entry.length} Б — "
            "сайдкар не соответствует шарду"
        )
    if keep_newline:
        return raw
    if not raw.endswith(b"\n"):
        raise ValueError(
            f"doc_id {entry.doc_id}: строка не заканчивается на \\n по офсету "
            f"{entry.offset}+{entry.length}"
        )
    return raw[:-1]


# --------------------------------------------------------------------------- #
# Проверка инвариантов
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OixViolation:
    """Нарушенный инвариант сайдкара."""

    kind: str
    index: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] запись #{self.index}: {self.detail}"


def verify_oix(
    entries: list[OixEntry], *, shard_size: int | None = None
) -> list[OixViolation]:
    """Проверяет инварианты, на которые опирается весь остальной код.

    Проверяется:

    * непрерывность — ``offset[i] + length[i] == offset[i+1]``;
    * первая запись начинается с нуля;
    * ``length >= 1`` (строка не может быть короче собственного ``\\n``);
    * если известен размер шарда — последняя запись доходит ровно до его конца;
    * ``doc_id`` уникальны.

    Возвращает список нарушений (пустой, если всё сходится), а не бросает
    исключение: вызывающему обычно нужен полный отчёт, а не первая ошибка.
    """
    problems: list[OixViolation] = []
    if not entries:
        return problems

    if entries[0].offset != 0:
        problems.append(
            OixViolation("start", 0, f"первая запись начинается с {entries[0].offset}, ожидался 0")
        )

    seen: dict[int, int] = {}
    for i, entry in enumerate(entries):
        if entry.length < 1:
            problems.append(
                OixViolation("length", i, f"length={entry.length}, минимум 1 (сам \\n)")
            )
        previous = seen.get(entry.doc_id)
        if previous is not None:
            problems.append(
                OixViolation("doc_id", i, f"doc_id {entry.doc_id} уже был в записи #{previous}")
            )
        else:
            seen[entry.doc_id] = i

        if i + 1 < len(entries):
            expected = entry.end
            actual = entries[i + 1].offset
            if expected != actual:
                problems.append(
                    OixViolation(
                        "contiguity",
                        i,
                        f"offset+length={expected}, а следующая запись начинается с {actual} "
                        f"(разрыв {actual - expected:+d} Б)",
                    )
                )

    if shard_size is not None:
        end = entries[-1].end
        if end != shard_size:
            problems.append(
                OixViolation(
                    "eof",
                    len(entries) - 1,
                    f"последняя запись доходит до {end}, размер шарда {shard_size} "
                    f"(расхождение {shard_size - end:+d} Б)",
                )
            )

    return problems
