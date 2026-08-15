"""Лексическая нога: BM25 с русской морфологией.

Зачем она вообще нужна рядом с плотным вектором. Замер шага 0.5 показал, что
`voyage-context-4` стягивает чанки одного документа друг к другу (косинус 0.92
между разными чанками против 0.74 между одним и тем же чанком в контексте
и вне его). Такая геометрия хорошо находит документ и хуже — место внутри него.
Лексика ведёт себя ровно наоборот: она не знает про документ ничего, но точное
совпадение редкого слова или номера акта ловит намертво.

Ни одна из двух выбранных плотных моделей разрежённых векторов не отдаёт,
поэтому BM25 строится отдельно — и это к лучшему: одна и та же лексическая
нога для обеих плотных моделей делает их сравнение честным.

Две вещи, специфичные для правового корпуса:

* **Номера актов не переживают обычную токенизацию.** `311-РП`, `171`,
  `1.2.3` — именно то, чем пользуются юристы для точного поиска. Они
  извлекаются отдельным правилом и добавляются к токенам как есть.
* **Русский требует стемминга.** «постановлениями» и «постановление» обязаны
  совпасть, иначе лексика проигрывает там, где должна выигрывать.

    python -m mlaw.lexical --build
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["tokenize", "LexicalIndex", "STOPWORDS"]

# Слова, которые есть почти в каждом акте: без отсева они дают вес там,
# где различающей силы нет.
STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за
    бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну
    вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там
    потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была
    сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому
    этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
    куда зачем всех никогда можно при наконец два об другой хоть после над больше
    тот через эти нас про всего них какая много разве три эту моя впрочем хорошо
    свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно
    всю между также г гг настоящего настоящему настоящим
    """.split()
)

# Токены-слова: кириллица и латиница, от двух букв.
_WORD = re.compile(r"[а-яёa-z]{2,}", re.IGNORECASE)
# Реквизиты и номера: 311-РП, 1.2.3, N 171, 44-ФЗ. Именно ими ищут точечно.
_CODE = re.compile(r"\d+(?:[-./]\w+)+|\d{2,}")


def _stemmer():
    import Stemmer

    return Stemmer.Stemmer("russian")


def tokenize(text: str, stemmer=None) -> list[str]:
    """Слова со стеммингом плюс номера как есть.

    Номер не стеммится и не режется: `311-РП` ценен ровно в том виде,
    в каком он напечатан в акте.
    """
    lowered = text.lower().replace("ё", "е")
    stemmer = stemmer or _stemmer()

    words = [w for w in _WORD.findall(lowered) if w not in STOPWORDS]
    stemmed = stemmer.stemWords(words)
    codes = _CODE.findall(lowered)
    return stemmed + codes


@dataclass(slots=True)
class LexicalHit:
    chunk_id: str
    score: float
    index: int


class LexicalIndex:
    """BM25 поверх чанков среза.

    Хранит соответствие «позиция в индексе -> chunk_id», чтобы выдачу можно
    было слить с плотной ногой и отфильтровать по тем же условиям.
    """

    def __init__(self, retriever, chunk_ids: list[str], payloads: list[dict]):
        self.retriever = retriever
        self.chunk_ids = chunk_ids
        self.payloads = payloads
        self._stemmer = _stemmer()

    # -- построение -------------------------------------------------------- #

    @classmethod
    def build(cls, chunks_path: Path, *, with_breadcrumb: bool = True) -> LexicalIndex:
        import bm25s

        stemmer = _stemmer()
        corpus: list[list[str]] = []
        chunk_ids: list[str] = []
        payloads: list[dict] = []

        started = time.time()
        with open(chunks_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                text = row["text"]
                if with_breadcrumb and row.get("breadcrumb"):
                    # Крошка попадает в лексический индекс: в ней лежат номер
                    # и название акта, по которым и ищут точечно.
                    text = f"{row['breadcrumb']}\n{text}"
                corpus.append(tokenize(text, stemmer))
                chunk_ids.append(row["chunk_id"])
                payloads.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "act_id": row["act_id"],
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                        "is_current": bool(row.get("is_current")),
                        "effective_date_begin": row.get("effective_date_begin"),
                        "effective_date_end": row.get("effective_date_end"),
                        "effective_date_end_sentinel": row.get("effective_date_end_sentinel"),
                        "title": row.get("title"),
                    }
                )

        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=False)
        print(f"  BM25: {len(corpus)} чанков за {time.time() - started:.1f} с")
        return cls(retriever, chunk_ids, payloads)

    # -- поиск ------------------------------------------------------------- #

    def search(self, query: str, k: int = 50) -> list[LexicalHit]:
        tokens = tokenize(query, self._stemmer)
        if not tokens:
            return []
        indices, scores = self.retriever.retrieve(
            [tokens], k=min(k, len(self.chunk_ids)), show_progress=False
        )
        return [
            LexicalHit(chunk_id=self.chunk_ids[int(i)], score=float(s), index=int(i))
            for i, s in zip(indices[0], scores[0], strict=True)
        ]

    # -- сохранение -------------------------------------------------------- #

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory / "bm25"))
        with open(directory / "meta.pkl", "wb") as fh:
            pickle.dump({"chunk_ids": self.chunk_ids, "payloads": self.payloads}, fh)

    @classmethod
    def load(cls, directory: Path) -> LexicalIndex:
        import bm25s

        retriever = bm25s.BM25.load(str(directory / "bm25"))
        with open(directory / "meta.pkl", "rb") as fh:
            meta = pickle.load(fh)
        return cls(retriever, meta["chunk_ids"], meta["payloads"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Лексический индекс BM25")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/bm25"))
    parser.add_argument("--no-breadcrumb", action="store_true")
    parser.add_argument("--query", default=None, help="проверочный запрос после сборки")
    args = parser.parse_args()

    print(f"Строю BM25 по {args.chunks}")
    index = LexicalIndex.build(args.chunks, with_breadcrumb=not args.no_breadcrumb)
    index.save(args.out)
    print(f"Сохранено в {args.out}")

    if args.query:
        print(f"\nЗапрос: {args.query!r}")
        for hit in index.search(args.query, k=5):
            payload = index.payloads[hit.index]
            print(f"  {hit.score:7.3f}  act {payload['act_id']:>7}  {payload['chunk_id']}")
            print(f"           {(payload['title'] or '')[:90]}")


if __name__ == "__main__":
    main()
