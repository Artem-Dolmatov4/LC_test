"""Тесты эмбеддера.

Сетевой слой подменяется: проверяется не Ollama, а наш контракт с ней —
асимметрия «запрос против документа», MRL-усечение и то, что переполнение
контекста превращается в исключение, а не в тихо укороченный вектор.
"""

from __future__ import annotations

import json
import math
import urllib.error

import pytest

from mlaw.embed import DEFAULT_QUERY_TASK, OllamaEmbedder, OllamaOverflow


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def capture(monkeypatch):
    """Перехватывает вызовы к Ollama и возвращает список отправленных тел."""
    sent: list[dict] = []

    def fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data))
        # Вектор из 4096 компонент, ненормированный — чтобы проверить,
        # что перенормировка после усечения действительно происходит.
        vector = [float(i % 7 + 1) for i in range(4096)]
        return FakeResponse({"embeddings": [vector], "prompt_eval_count": 42})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return sent


def test_documents_go_without_instruction_prefix(capture):
    embedder = OllamaEmbedder(dim=None)
    embedder.embed_documents([["первый чанк", "второй чанк"]])

    assert [body["input"] for body in capture] == ["первый чанк", "второй чанк"]
    assert all("Instruct:" not in body["input"] for body in capture)


def test_queries_get_the_instruction_prefix(capture):
    embedder = OllamaEmbedder(dim=None)
    embedder.embed_queries(["срок действия разрешения"])

    assert capture[0]["input"] == (
        f"Instruct: {DEFAULT_QUERY_TASK}\nQuery: срок действия разрешения"
    )


def test_document_grouping_is_flattened_for_a_pointwise_model(capture):
    """Qwen — поштучная модель: группировка по документам ей безразлична."""
    embedder = OllamaEmbedder(dim=None)
    result = embedder.embed_documents([["a", "b"], ["c"]])

    assert len(capture) == 3
    assert len(result.vectors) == 3


def test_truncate_is_always_disabled(capture):
    """truncate=true — это молчаливое обрезание, его быть не должно никогда."""
    embedder = OllamaEmbedder(dim=None)
    embedder.embed_documents([["чанк"]])
    assert capture[0]["truncate"] is False


def test_num_ctx_is_always_explicit(capture):
    """Полагаться на значение runner'а нельзя: замерено 2048 при максимуме 40 960."""
    embedder = OllamaEmbedder(dim=None, num_ctx=8192)
    embedder.embed_documents([["чанк"]])
    assert capture[0]["options"]["num_ctx"] == 8192


def test_mrl_truncation_produces_unit_vectors(capture):
    embedder = OllamaEmbedder(dim=1024)
    result = embedder.embed_documents([["чанк"]])

    assert result.dim == 1024
    norm = math.sqrt(sum(x * x for x in result.vectors[0]))
    assert norm == pytest.approx(1.0)


def test_native_dimension_is_returned_untruncated(capture):
    result = OllamaEmbedder(dim=None).embed_documents([["чанк"]])
    assert result.dim == 4096


def test_token_accounting(capture):
    result = OllamaEmbedder(dim=None).embed_documents([["a", "b", "c"]])
    assert result.prompt_tokens == 3 * 42


def test_context_overflow_raises_typed_error(monkeypatch):
    """HTTP 400 про длину контекста обязан стать OllamaOverflow, а не RuntimeError."""

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            url="http://localhost:11434/api/embed",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=_BytesFP(b'{"error":"the input length exceeds the context length"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(OllamaOverflow, match="num_ctx"):
        OllamaEmbedder(num_ctx=512).embed_documents([["очень длинный чанк"]])


def test_other_http_errors_are_not_swallowed(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            url="http://localhost:11434/api/embed",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=_BytesFP(b'{"error":"model runner crashed"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as excinfo:
        OllamaEmbedder().embed_documents([["чанк"]])
    assert not isinstance(excinfo.value, OllamaOverflow)


class _BytesFP:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass
