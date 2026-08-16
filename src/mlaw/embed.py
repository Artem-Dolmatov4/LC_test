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

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

__all__ = [
    "EmbedResult",
    "Embedder",
    "OllamaEmbedder",
    "OllamaOverflow",
    "VoyageEmbedder",
    "DashScopeEmbedder",
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

# Лимиты. Их два разных, и документация их путает, а API — нет:
#   * 32 000 токенов — контекстное окно ОДНОГО документа. Документ длиннее
#     не контекстуализируется целиком, его приходится резать на окна.
#   * 120 000 токенов — потолок на весь запрос, сколько бы документов в нём
#     ни было.
# Проверено ответом API: «The example at index 0 in your batch has too many
# tokens and does not fit into the model's context window of 32000 tokens».
VOYAGE_MAX_TOKENS_PER_DOCUMENT = 32_000
VOYAGE_MAX_TOKENS_PER_REQUEST = 120_000
VOYAGE_MAX_INPUTS = 1_000
VOYAGE_MAX_CHUNKS = 16_000

# Оценка длины в токенах по знакам, нужна для нарезки на окна до обращения
# к API. Замер на настоящих чанках корпуса токенизатором Voyage: 2.85–2.93
# знака на токен (~600 токенов на чанк в 1 862 знака).
#
# Значение берётся НИЖЕ замеренного, а не выше: оценка должна ЗАВЫШАТЬ число
# токенов, иначе окно окажется больше контекста модели и запрос отвергнут.
# Первая версия стояла на 3.0 «с запасом» — запас был в обратную сторону,
# и окна не влезали.
#
# Токенизатор Qwen на том же тексте даёт 3.87 знака на токен — оценки моделей
# не взаимозаменяемы.
CHARS_PER_TOKEN_CONSERVATIVE = 2.2


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
        concurrency: int = 8,
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
        self.concurrency = max(1, concurrency)
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
        """Режет документ на окна по контекстному лимиту модели.

        Окно — 32 000 токенов, то есть примерно 53 чанка по 2 000 знаков.
        Документы длиннее контекстуализируются кусками, и чанки из разных
        окон видят разный контекст. Для среза это не редкость: у нас есть
        редакции по 12.5 млн знаков, то есть больше сотни окон на документ.
        """
        windows: list[list[str]] = []
        current: list[str] = []
        budget = 0
        for chunk in chunks:
            cost = self._estimate_tokens(chunk)
            if current and budget + cost > VOYAGE_MAX_TOKENS_PER_DOCUMENT:
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

    def _embed_examples(
        self, examples: list[list[str]], input_type: str
    ) -> tuple[list[list[list[float]]], int]:
        """Считает батч, сам дробя примеры, не влезшие в контекст модели.

        Оценка токенов по знакам работает для прозы, но ломается на таблицах
        и перечнях: там знаков на токен вдвое меньше, чем в тексте статьи,
        и единый коэффициент их не описывает. Вместо подбора коэффициента
        под худший случай — реакция на факт: API сказал «не влезло», значит
        делим пополам и повторяем.

        Цена дробления — более узкий контекст у затронутых чанков; счётчик
        `windowed_documents` это фиксирует, чтобы попало в отчёт.
        """
        if not examples:
            return [], 0
        payload = {
            "inputs": examples,
            "model": self.model,
            "input_type": input_type,
            "output_dimension": self.dim,
        }
        try:
            body = self._post(payload)
        except RuntimeError as exc:
            message = str(exc)

            # Батч тяжелее 120 000 токенов — делим сам батч, примеры целы.
            if "TOO_MANY_TOKENS_IN_BATCH" in message and len(examples) > 1:
                middle = len(examples) // 2
                left, left_tokens = self._embed_examples(examples[:middle], input_type)
                right, right_tokens = self._embed_examples(examples[middle:], input_type)
                return left + right, left_tokens + right_tokens

            # Отдельный пример не влез в контекстное окно — дробим примеры.
            if "context window" in message and any(len(e) > 1 for e in examples):
                halves: list[list[str]] = []
                for example in examples:
                    if len(example) > 1:
                        middle = len(example) // 2
                        halves.append(example[:middle])
                        halves.append(example[middle:])
                        self.windowed_documents += 1
                    else:
                        halves.append(example)
                return self._embed_examples(halves, input_type)

            raise

        tokens = (body.get("usage") or {}).get("total_tokens", 0)
        out = [
            [
                chunk["embedding"]
                for chunk in sorted(document["data"], key=lambda c: c["index"])
            ]
            for document in sorted(body["data"], key=lambda d: d["index"])
        ]
        return out, tokens

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

        # Батчи независимы: каждое окно — самостоятельный контекст, и порядок
        # ответов на векторы не влияет. Последовательная отправка упирается
        # не в модель, а в задержку сети: замер дал 3.8 чанка/с против 25
        # при параллельной.
        batches = self._batch(expanded)
        results: list[list[list[float]]] = [[] for _ in batches]
        tokens = 0
        started = time.time()

        if self.concurrency <= 1 or len(batches) == 1:
            for position, batch in enumerate(batches):
                groups, used = self._embed_examples(batch, input_type)
                tokens += used
                results[position] = [v for group in groups for v in group]
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._embed_examples, batch, input_type): position
                    for position, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    position = futures[future]
                    groups, used = future.result()
                    tokens += used
                    results[position] = [v for group in groups for v in group]

        vectors: list[list[float]] = [v for batch in results for v in batch]

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


# --------------------------------------------------------------------------- #
# DashScope (Alibaba) — Qwen3-семейство через API
# --------------------------------------------------------------------------- #


class DashScopeEmbedder(Embedder):
    """`text-embedding-v4` — поштучный эмбеддер семейства Qwen3.

    Нужен как практичная замена локальному `qwen3-embedding:8b`: та же семья
    моделей, но полная точность вместо Q4_K_M и скорость API вместо 0.252
    чанка в секунду. Ценой того, что нога перестаёт быть self-hosted —
    и это надо назвать в отчёте прямо, а не умолчать.

    Лимит батча — 10 текстов на запрос (ограничение API).
    """

    ENDPOINT = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings"
    )
    MAX_BATCH = 10

    def __init__(
        self,
        model: str = "text-embedding-v4",
        *,
        api_key: str | None = None,
        dim: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 5,
        endpoint: str | None = None,
        concurrency: int = 8,
    ):
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("нет DASHSCOPE_API_KEY — ни в аргументе, ни в окружении")
        self._key = key
        self.model = model
        self.name = f"dashscope/{model}"
        self.dim = dim
        self.timeout = timeout
        self.max_retries = max_retries
        self.endpoint = endpoint or self.ENDPOINT
        self.concurrency = max(1, concurrency)

    def _post(self, texts: Sequence[str]) -> dict:
        payload = {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dim,
            "encoding_format": "float",
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries - 1:
                    raise RuntimeError(f"DashScope вернул HTTP {exc.code}: {detail}") from exc
                time.sleep(delay)
                delay *= 2
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
                http.client.HTTPException,
            ):
                # Ответ на батч из 10 векторов по 1024 измерения — сотни
                # килобайт JSON, и он регулярно приходит оборванным.
                # Обрыв обязан быть ретраем, а не потерей части чанков.
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("недостижимо")

    def _run(self, texts: Sequence[str]) -> EmbedResult:
        """Батчи идут параллельно.

        Модель поштучная, порядок чанков внутри батча роли не играет, а API
        держит несколько запросов сразу — поэтому последовательный цикл здесь
        просто теряет время: замер дал 1.16 чанка/с против 15.3 часа на срез.
        Результаты собираются по индексу батча, так что порядок векторов
        совпадает с порядком входа независимо от порядка ответов.
        """
        batches = [
            texts[start : start + self.MAX_BATCH]
            for start in range(0, len(texts), self.MAX_BATCH)
        ]
        results: list[list[list[float]]] = [[] for _ in batches]
        tokens = 0
        started = time.time()

        if self.concurrency <= 1 or len(batches) == 1:
            for index, batch in enumerate(batches):
                body = self._post(batch)
                tokens += (body.get("usage") or {}).get("total_tokens", 0)
                results[index] = [
                    item["embedding"] for item in sorted(body["data"], key=lambda d: d["index"])
                ]
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._post, batch): index
                    for index, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    body = future.result()
                    tokens += (body.get("usage") or {}).get("total_tokens", 0)
                    results[index] = [
                        item["embedding"]
                        for item in sorted(body["data"], key=lambda d: d["index"])
                    ]

        vectors = [vector for batch in results for vector in batch]
        return EmbedResult(
            vectors=vectors,
            prompt_tokens=tokens,
            seconds=time.time() - started,
            meta={"model": self.model, "dim": self.dim, "concurrency": self.concurrency},
        )

    def embed_documents(self, documents: Sequence[Sequence[str]]) -> EmbedResult:
        return self._run([chunk for document in documents for chunk in document])

    def embed_queries(self, queries: Sequence[str]) -> EmbedResult:
        return self._run(list(queries))
