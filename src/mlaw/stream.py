"""Потоковое чтение корпуса MLAW — из архива `.tar.zst` или из распакованной папки.

Весь пайплайн читает корпус только отсюда, и только потоком. Полный проход по
всем 28 шардам с разбором JSON занимает ~40 с, поэтому распаковывать 16.9 ГБ на
диск ради инвентаризации незачем: распаковка идёт на лету со скоростью ~3.4 ГБ/с.

Два источника с одинаковым интерфейсом:

* :class:`ArchiveCorpus` — читает прямо из ``MLAW_dataset.tar.zst``;
* :class:`DirectoryCorpus` — читает уже распакованную папку ``MLAW/``.

Первый нужен, чтобы не держать на диске 16.9 ГБ. Второй — чтобы работать с
собственным срезом и иметь произвольный доступ по офсетам (в zstd-потоке
произвольного доступа нет: архив — один непрерывный кадр, и добраться до шарда
N можно только пройдя через предыдущие).
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

__all__ = [
    "Corpus",
    "ArchiveCorpus",
    "DirectoryCorpus",
    "Record",
    "open_corpus",
    "shard_name",
    "oix_name",
    "SHARD_COUNT",
]

SHARD_COUNT = 28
_MEMBER_PREFIX = "MLAW/"


def shard_name(index: int) -> str:
    """``0`` -> ``mlaw_0000.jsonl``"""
    return f"mlaw_{index:04d}.jsonl"


def oix_name(index: int) -> str:
    """``0`` -> ``mlaw_0000.oix``"""
    return f"mlaw_{index:04d}.oix"


@dataclass(frozen=True, slots=True)
class Record:
    """Одна строка шарда вместе с её адресом.

    ``line`` хранится сырыми байтами и разбирается по требованию: на полном
    проходе разбор JSON — основная статья расходов, и на многих задачах
    (например, отбор по ``doc_id``) он не нужен вовсе.
    """

    shard: int
    offset: int
    length: int
    line: bytes

    def json(self) -> dict:
        return json.loads(self.line)


class Corpus:
    """Общий интерфейс источника корпуса."""

    def iter_lines(self, shards: Sequence[int] | None = None) -> Iterator[Record]:
        raise NotImplementedError

    def iter_records(self, shards: Sequence[int] | None = None) -> Iterator[dict]:
        """Разобранные записи. Удобно, но всегда платит за ``json.loads``."""
        for record in self.iter_lines(shards):
            yield record.json()

    def read_sidecar(self, index: int) -> bytes:
        raise NotImplementedError

    def read_meta(self, name: str) -> bytes:
        """``mlaw_manifest.json`` или ``mlaw_stats.json`` сырыми байтами."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Архив
# --------------------------------------------------------------------------- #


class ArchiveCorpus(Corpus):
    """Читает корпус прямо из ``.tar.zst``, ничего не распаковывая на диск.

    Каждый вызов ``iter_*`` — отдельный последовательный проход по архиву.
    Просить шарды вразнобой можно, но выгоднее собирать их в один вызов:
    сквозной проход стоит одинаково независимо от того, сколько членов из него
    реально нужно.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"архив не найден: {self.path}")

    # -- распаковка -------------------------------------------------------- #

    @contextmanager
    def _decompressed(self) -> Iterator[BinaryIO]:
        """Отдаёт поток распакованного tar.

        Предпочитает библиотеку ``zstandard``; если её нет — вызывает бинарь
        ``zstd``. Оба пути дают непрерывный поток, поэтому tar открывается в
        режиме ``r|`` (последовательный, без перемоток).
        """
        try:
            import zstandard  # noqa: PLC0415 — опциональная зависимость
        except ImportError:
            zstandard = None

        if zstandard is not None:
            with open(self.path, "rb") as raw:
                reader = zstandard.ZstdDecompressor().stream_reader(raw)
                try:
                    yield reader
                finally:
                    reader.close()
            return

        proc = subprocess.Popen(
            ["zstd", "-dc", "--no-progress", str(self.path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdout is not None
        try:
            yield proc.stdout
        finally:
            # Потребитель почти всегда бросает чтение раньше конца архива
            # (например, прочитав только шард 0000) — гасим процесс, чтобы не
            # оставлять его дописывать 16.9 ГБ в закрытую трубу.
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
            proc.wait()

    @contextmanager
    def _tar(self) -> Iterator[tarfile.TarFile]:
        with self._decompressed() as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                yield tar

    # -- чтение ------------------------------------------------------------ #

    def iter_lines(self, shards: Sequence[int] | None = None) -> Iterator[Record]:
        wanted = _wanted_shards(shards)
        remaining = set(wanted)
        with self._tar() as tar:
            for member in tar:
                if not member.isfile():
                    continue
                index = _shard_index(member.name)
                if index is None or index not in remaining:
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                offset = 0
                for line in fh:
                    length = len(line)
                    yield Record(
                        shard=index,
                        offset=offset,
                        length=length,
                        line=line[:-1] if line.endswith(b"\n") else line,
                    )
                    offset += length
                remaining.discard(index)
                if not remaining:
                    # Все нужные шарды прочитаны — дальше тянуть архив незачем.
                    break

    def read_sidecar(self, index: int) -> bytes:
        return self._read_member(_MEMBER_PREFIX + oix_name(index))

    def read_meta(self, name: str) -> bytes:
        return self._read_member(_MEMBER_PREFIX + name)

    def _read_member(self, name: str) -> bytes:
        with self._tar() as tar:
            for member in tar:
                if member.name != name:
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    break
                return fh.read()
        raise KeyError(f"в архиве нет члена {name}")

    def extract(self, names: Sequence[str], dest_dir: Path | str) -> list[Path]:
        """Распаковывает перечисленные члены за один проход.

        Нужно там, где без произвольного доступа не обойтись: собственный срез
        читается по офсетам, а в zstd-потоке перемоток нет.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        remaining = {n if n.startswith(_MEMBER_PREFIX) else _MEMBER_PREFIX + n for n in names}
        written: list[Path] = []
        with self._tar() as tar:
            for member in tar:
                if member.name not in remaining:
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                out = dest / Path(member.name).name
                with open(out, "wb") as sink:
                    while chunk := fh.read(1 << 22):
                        sink.write(chunk)
                written.append(out)
                remaining.discard(member.name)
                if not remaining:
                    break
        if remaining:
            raise KeyError(f"в архиве нет членов: {sorted(remaining)}")
        return written


# --------------------------------------------------------------------------- #
# Распакованная папка
# --------------------------------------------------------------------------- #


class DirectoryCorpus(Corpus):
    """Читает распакованную папку ``MLAW/`` — с произвольным доступом по офсетам."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"не папка: {self.root}")

    def shard_path(self, index: int) -> Path:
        return self.root / shard_name(index)

    def sidecar_path(self, index: int) -> Path:
        return self.root / oix_name(index)

    def iter_lines(self, shards: Sequence[int] | None = None) -> Iterator[Record]:
        for index in _wanted_shards(shards):
            path = self.shard_path(index)
            if not path.exists():
                continue
            with open(path, "rb") as fh:
                offset = 0
                for line in fh:
                    length = len(line)
                    yield Record(
                        shard=index,
                        offset=offset,
                        length=length,
                        line=line[:-1] if line.endswith(b"\n") else line,
                    )
                    offset += length

    def read_sidecar(self, index: int) -> bytes:
        return self.sidecar_path(index).read_bytes()

    def read_meta(self, name: str) -> bytes:
        return (self.root / name).read_bytes()

    def available_shards(self) -> list[int]:
        return sorted(i for i in range(SHARD_COUNT) if self.shard_path(i).exists())


# --------------------------------------------------------------------------- #
# Помощники
# --------------------------------------------------------------------------- #


def open_corpus(path: Path | str) -> Corpus:
    """Открывает корпус, сам разбираясь, архив это или папка."""
    path = Path(path)
    if path.is_dir():
        return DirectoryCorpus(path)
    if path.suffix == ".zst" or path.name.endswith(".tar.zst"):
        return ArchiveCorpus(path)
    raise ValueError(f"не понимаю, что это за источник корпуса: {path}")


def _wanted_shards(shards: Sequence[int] | None) -> list[int]:
    if shards is None:
        return list(range(SHARD_COUNT))
    out = sorted(set(shards))
    for index in out:
        if not 0 <= index < SHARD_COUNT:
            raise ValueError(f"шард {index} вне диапазона 0..{SHARD_COUNT - 1}")
    return out


def _shard_index(member_name: str) -> int | None:
    """``MLAW/mlaw_0013.jsonl`` -> ``13``; всё остальное -> ``None``."""
    name = Path(member_name).name
    if not (name.startswith("mlaw_") and name.endswith(".jsonl")):
        return None
    try:
        return int(name[5:-6])
    except ValueError:
        return None
