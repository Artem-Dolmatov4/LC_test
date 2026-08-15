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
