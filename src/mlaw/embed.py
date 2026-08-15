"""Эмбеддеры: локальный Qwen3 через Ollama и контекстуализированный Voyage.

Две модели асимметричны, и это не деталь реализации, а свойство, которое надо
удержать в интерфейсе:

* **Qwen3-Embedding** — поштучный эмбеддер. Запросу нужен префикс инструкции,
  документу — нет. Ollama сам его не добавит.
* **voyage-context-4** — контекстуализированный: вектор чанка считается с учётом
  остальных чанков того же документа, вход — ``List[List[str]]``.

Поэтому базовый интерфейс принимает документ как список чанков, а не плоский
список строк: поштучная модель просто игнорирует группировку, контекстная —
использует.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "EmbedResult",
    "Embedder",
    "OllamaEmbedder",
    "OllamaOverflow",
    "VoyageEmbedder",
    "DEFAULT_QUERY_TASK",
]

# Инструкция для запросов. Документы префикса не получают — асимметрия
# заложена в саму модель, и её нарушение тихо роняет качество.
DEFAULT_QUERY_TASK = (
    "Given a legal question, retrieve the passage of a Moscow city legal act "
    "that answers it"
)


@dataclass(slots=True)
class EmbedResult:
    """Векторы и учёт израсходованного."""

    vectors: list[list[float]]
    prompt_tokens: int = 0
    seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


class Embedder:
    """Общий интерфейс."""

    name: str
    dim: int

    def embed_documents(self, documents: Sequence[Sequence[str]]) -> EmbedResult:
        """``documents[i][j]`` — j-й чанк i-го документа."""
        raise NotImplementedError

    def embed_queries(self, queries: Sequence[str]) -> EmbedResult:
        raise NotImplementedError


class OllamaOverflow(RuntimeError):
    """Вход не поместился в ``num_ctx``.

    Существует отдельным типом, потому что альтернатива — молчаливое обрезание:
    при ``truncate=true`` Ollama обрежет вход по границе контекста и вернёт
    вектор половины чанка, ничем не отличимый от правильного.
    """


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class OllamaEmbedder(Embedder):
    """Qwen3-Embedding через локальный Ollama.

    Три предосторожности, каждая из которых закрывает измеренную ловушку:

    1. ``num_ctx`` задаётся **явно**. Runner поднимается со своим значением
       (замерено: 2048 при заявленном максимуме модели 40 960), и чанк на
       2 000 знаков русского текста — это ~694 токена, то есть запас невелик.
    2. ``truncate=False`` — переполнение обязано быть явной ошибкой.
       Проверено: Ollama отвечает HTTP 400 «the input length exceeds the
       context length». При ``truncate=True`` тот же вход проходит молча
       с ``prompt_eval_count`` ровно в ``num_ctx - 1``.
    3. Префикс инструкции добавляется **только запросам**.

    Батчинг намеренно не используется: замер на M4 показал, что список из 4
    входов обрабатывается медленнее (5.6 с на чанк), чем те же 4 по одному
    (4.5 с на чанк).
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:8b",
        *,
        host: str | None = None,
        num_ctx: int = 8192,
        dim: int | None = 1024,
        query_task: str = DEFAULT_QUERY_TASK,
        timeout: float = 1800.0,
    ):
        self.model = model
        self.name = f"ollama/{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.num_ctx = num_ctx
        # None — отдавать нативные 4096; иначе MRL-усечение до dim.
        self.dim = dim or 4096
        self.query_task = query_task
        self.timeout = timeout

    # -- низкий уровень ---------------------------------------------------- #

    def _call(self, text: str) -> tuple[list[float], int]:
        payload = {
            "model": self.model,
            "input": text,
            "truncate": False,
            "options": {"num_ctx": self.num_ctx},
        }
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 400 and "context length" in detail:
                raise OllamaOverflow(
                    f"вход не поместился в num_ctx={self.num_ctx}: {detail}"
                ) from exc
            raise RuntimeError(f"Ollama вернул HTTP {exc.code}: {detail}") from exc

        vector = body["embeddings"][0]
        return vector, body.get("prompt_eval_count") or 0

    def _embed_many(self, texts: Sequence[str]) -> EmbedResult:
        vectors: list[list[float]] = []
        tokens = 0
        started = time.time()
        for text in texts:
            vector, used = self._call(text)
            tokens += used
            if self.dim < len(vector):
                # MRL: усечение с последующей перенормировкой — без неё
                # косинус перестаёт быть косинусом.
                vector = _normalize(vector[: self.dim])
            vectors.append(vector)
        return EmbedResult(
            vectors=vectors,
            prompt_tokens=tokens,
            seconds=time.time() - started,
            meta={"model": self.model, "num_ctx": self.num_ctx, "dim": self.dim},
        )

    # -- интерфейс --------------------------------------------------------- #

    def embed_documents(self, documents: Sequence[Sequence[str]]) -> EmbedResult:
        """Документы идут без префикса — так требует модель."""
        flat = [chunk for document in documents for chunk in document]
        return self._embed_many(flat)

    def embed_queries(self, queries: Sequence[str]) -> EmbedResult:
        prefixed = [f"Instruct: {self.query_task}\nQuery: {q}" for q in queries]
        return self._embed_many(prefixed)


# --------------------------------------------------------------------------- #
# Voyage — контекстуализированные эмбеддинги
# --------------------------------------------------------------------------- #

# Лимиты запроса, подтверждённые документацией и smoke-тестом.
VOYAGE_MAX_TOKENS_PER_REQUEST = 120_000
VOYAGE_MAX_INPUTS = 1_000
VOYAGE_MAX_CHUNKS = 16_000

# Оценка длины в токенах по знакам. Замер на русском правовом тексте дал
# 3.87 знака на токен; берём 3.0 с запасом, чтобы оценка не занижала.
CHARS_PER_TOKEN_CONSERVATIVE = 3.0


class VoyageEmbedder(Embedder):
    """`voyage-context-4` — вектор чанка считается с учётом соседей по документу.

    Отсюда два следствия, ради которых класс сложнее обёртки над HTTP:

    1. **Документ — единица запроса.** Чанки нельзя перемешивать между
       документами: контекст возьмётся не тот.
    2. **Лимит 120K токенов на запрос ломается о длинный хвост.** Документ на
       15.7 млн знаков — это ~5 млн токенов. Такой документ режется на окна,
       и чанки из разных окон контекстуализированы **не одинаково**. Число
       документов, потребовавших окон, учитывается и идёт в отчёт: это
       ограничение метода, а не деталь реализации.
    """

    ENDPOINT = "https://api.voyageai.com/v1/contextualizedembeddings"

    def __init__(
        self,
        model: str = "voyage-context-4",
        *,
        api_key: str | None = None,
        dim: int = 1024,
        timeout: float = 300.0,
        max_retries: int = 5,
    ):
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("нет VOYAGE_API_KEY — ни в аргументе, ни в окружении")
        self._key = key
        self.model = model
        self.name = f"voyage/{model}"
        self.dim = dim
        self.timeout = timeout
        self.max_retries = max_retries
        self.windowed_documents = 0  # сколько документов не влезли в один запрос

    # -- низкий уровень ---------------------------------------------------- #

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                retryable = exc.code in (429, 500, 502, 503, 504)
                if not retryable or attempt == self.max_retries - 1:
                    raise RuntimeError(f"Voyage вернул HTTP {exc.code}: {detail}") from exc
                if exc.code == 429:
                    # Без привязанной карты лимит — 3 запроса и 10K токенов
                    # в минуту. Откат в секунду тут сам себе злейший враг:
                    # ретраи выедают ту же квоту, из-за которой и начались.
                    time.sleep(max(delay, 25.0))
                    delay = max(delay * 2, 50.0)
                else:
                    time.sleep(delay)
                    delay *= 2
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("недостижимо")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / CHARS_PER_TOKEN_CONSERVATIVE))

    def _window(self, chunks: Sequence[str]) -> list[list[str]]:
        """Режет документ на окна, укладывающиеся в лимит токенов запроса."""
        windows: list[list[str]] = []
        current: list[str] = []
        budget = 0
        for chunk in chunks:
            cost = self._estimate_tokens(chunk)
            if current and budget + cost > VOYAGE_MAX_TOKENS_PER_REQUEST:
                windows.append(current)
                current, budget = [], 0
            current.append(chunk)
            budget += cost
        if current:
            windows.append(current)
        return windows

    def _batch(self, documents: Sequence[Sequence[str]]) -> list[list[list[str]]]:
        """Собирает документы в запросы под все три лимита сразу."""
        batches: list[list[list[str]]] = []
        current: list[list[str]] = []
        tokens = chunks = 0
        for document in documents:
            cost = sum(self._estimate_tokens(c) for c in document)
            over = (
                current
                and (
                    tokens + cost > VOYAGE_MAX_TOKENS_PER_REQUEST
                    or len(current) + 1 > VOYAGE_MAX_INPUTS
                    or chunks + len(document) > VOYAGE_MAX_CHUNKS
                )
            )
            if over:
                batches.append(current)
                current, tokens, chunks = [], 0, 0
            current.append(list(document))
            tokens += cost
            chunks += len(document)
        if current:
            batches.append(current)
        return batches

    def _run(self, documents: Sequence[Sequence[str]], input_type: str) -> EmbedResult:
        # Документы, не влезающие в один запрос, распадаются на окна. Окно
        # становится самостоятельным «документом» с точки зрения API — и это
        # ровно то место, где контекстуализация теряет часть контекста.
        expanded: list[list[str]] = []
        for document in documents:
            windows = self._window(document)
            if len(windows) > 1:
                self.windowed_documents += 1
            expanded.extend(windows)

        vectors: list[list[float]] = []
        tokens = 0
        started = time.time()
        for batch in self._batch(expanded):
            payload = {
                "inputs": batch,
                "model": self.model,
                "input_type": input_type,
                "output_dimension": self.dim,
            }
            body = self._post(payload)
            tokens += (body.get("usage") or {}).get("total_tokens", 0)
            for document_out in sorted(body["data"], key=lambda d: d["index"]):
                for chunk_out in sorted(document_out["data"], key=lambda c: c["index"]):
                    vectors.append(chunk_out["embedding"])

        return EmbedResult(
            vectors=vectors,
            prompt_tokens=tokens,
            seconds=time.time() - started,
            meta={
                "model": self.model,
                "dim": self.dim,
                "windowed_documents": self.windowed_documents,
            },
        )

    # -- интерфейс --------------------------------------------------------- #

    def embed_documents(self, documents: Sequence[Sequence[str]]) -> EmbedResult:
        return self._run(documents, "document")

    def embed_queries(self, queries: Sequence[str]) -> EmbedResult:
        """Запрос — документ из одного чанка; префикс инструкции не нужен."""
        return self._run([[q] for q in queries], "query")
