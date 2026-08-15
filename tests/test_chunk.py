"""Тесты нарезки.

Два блока. Первый — восстановление границ из дерева `contents`: там живут
обе ловушки формата (`char_end` как конец поддерева и узлы нулевого размаха).
Второй — тождество `text[char_start:char_end] == chunk.text`, без которого
проверка цитат в шаге 6 сравнивала бы разные строки.
"""

from __future__ import annotations

import pytest

from mlaw.chunk import Chunk, ChunkConfig, chunk_document, derive_segments

CFG = ChunkConfig(target_chars=200, max_chars=300, min_chars=20, overlap_chars=50)


def node(level: int, label: str, start: int, end: int, parent: int | None = None) -> dict:
    return {
        "level": level,
        "label": label,
        "char_start": start,
        "char_end": end,
        "parent": parent,
    }


def document(text: str, contents: list[dict] | None = None, **kwargs) -> dict:
    out = {
        "doc_id": 1,
        "title": "Постановление № 1",
        "text": text,
        "edition": {"act_id": 10, "n_editions": 1, "index": 0, "is_current": True},
    }
    if contents is not None:
        out["contents"] = contents
    out.update(kwargs)
    return out


# --------------------------------------------------------------------------- #
# Границы сегментов
# --------------------------------------------------------------------------- #


def test_segments_come_from_char_start_not_char_end():
    """Ловушка: `char_end` — конец поддерева.

    У «Статьи 5» char_end простирается до конца её пунктов. Если строить
    сегменты по char_end, статья перекроет собственные пункты и породит
    вложенные интервалы вместо соседних.
    """
    text = "х" * 300
    contents = [
        node(1, "Статья 5", 0, 300),
        node(2, "Пункт 1", 100, 200, parent=0),
        node(2, "Пункт 2", 200, 300, parent=0),
    ]
    segments = derive_segments(text, contents)

    assert [(s.start, s.end) for s in segments] == [(0, 100), (100, 200), (200, 300)]
    # Сегменты соседние и непрерывные, без перекрытий.
    for previous, following in zip(segments, segments[1:], strict=False):
        assert previous.end == following.start


def test_zero_span_nodes_are_kept_and_addressed():
    """Узлы нулевого размаха (0.92 % узлов) не выбрасываются.

    Два узла на одной позиции — самый глубокий даёт метку, но сегмент
    остаётся один: текста между ними нет.
    """
    text = "х" * 200
    contents = [
        node(1, "Раздел I", 0, 200),
        node(2, "Глава 1", 100, 100, parent=0),
        node(3, "Статья 7", 100, 200, parent=1),
    ]
    segments = derive_segments(text, contents)

    assert [(s.start, s.end) for s in segments] == [(0, 100), (100, 200)]
    assert segments[1].breadcrumb == ("Раздел I", "Глава 1", "Статья 7")


def test_breadcrumb_follows_parent_chain():
    text = "х" * 100
    contents = [
        node(1, "Приложение 3", 0, 100),
        node(2, "Раздел II", 50, 100, parent=0),
    ]
    segments = derive_segments(text, contents)
    assert segments[-1].breadcrumb == ("Приложение 3", "Раздел II")


def test_multiroot_document_is_normal():
    """Многокорневых документов 100 % — parent is None не означает один корень."""
    text = "х" * 200
    contents = [node(1, "Часть I", 0, 100), node(1, "Часть II", 100, 200)]
    segments = derive_segments(text, contents)
    assert [s.breadcrumb for s in segments] == [("Часть I",), ("Часть II",)]


def test_text_before_the_first_node_becomes_its_own_segment():
    """Преамбула и реквизиты акта до первого узла оглавления не теряются."""
    text = "х" * 100
    segments = derive_segments(text, [node(1, "Статья 1", 40, 100)])
    assert (segments[0].start, segments[0].end) == (0, 40)
    assert segments[0].breadcrumb == ()


def test_document_without_contents_is_one_segment():
    segments = derive_segments("х" * 500, None)
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (0, 500)


# --------------------------------------------------------------------------- #
# Тождество офсетов и текста
# --------------------------------------------------------------------------- #


def _check_offsets(record: dict, chunks: list[Chunk]) -> None:
    text = record["text"]
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_offsets_resolve_back_to_the_document_text():
    """Ради шага 6: цитата обязана резолвиться в исходный текст побайтово."""
    text = "\n".join(f"Абзац номер {i}. " + "с" * 120 for i in range(20))
    record = document(text)
    chunks = chunk_document(record, CFG)

    assert chunks
    _check_offsets(record, chunks)


def test_offsets_resolve_with_structure_too():
    text = "\n".join("п" * 90 for _ in range(30))
    contents = [
        node(1, "Раздел I", 0, len(text) // 2),
        node(1, "Раздел II", len(text) // 2, len(text)),
    ]
    record = document(text, contents)
    chunks = chunk_document(record, CFG)

    assert chunks
    _check_offsets(record, chunks)


def test_chunks_never_exceed_max_chars():
    text = "\n".join("я" * 250 for _ in range(20))
    for chunk in chunk_document(document(text), CFG):
        assert len(chunk.text) <= CFG.max_chars


def test_chunks_do_not_overlap_outside_the_emergency_path():
    text = "\n".join(f"Абзац {i}. " + "т" * 100 for i in range(30))
    chunks = chunk_document(document(text), CFG)
    normal = [c for c in chunks if c.boundary_kind != "mid_paragraph"]
    for previous, following in zip(normal, normal[1:], strict=False):
        assert following.char_start >= previous.char_end


# --------------------------------------------------------------------------- #
# Жёсткость структурных границ
# --------------------------------------------------------------------------- #


def test_chunk_never_crosses_a_segment_boundary_between_different_parents():
    """Два соседних приложения не склеиваются в один чанк."""
    half = "а" * 60
    text = half + "б" * 60
    contents = [node(1, "Приложение 1", 0, 60), node(1, "Приложение 2", 60, 120)]
    chunks = chunk_document(document(text, contents), CFG)

    assert len(chunks) == 2
    assert chunks[0].text == half
    assert "Приложение 1" in chunks[0].breadcrumb
    assert "Приложение 2" in chunks[1].breadcrumb


def test_siblings_under_one_parent_are_merged():
    """Пункты одной статьи сливаются — иначе чанки выродились бы в огрызки."""
    text = "".join("п" * 30 for _ in range(4))
    contents = [
        node(1, "Статья 5", 0, 120),
        node(2, "Пункт 1", 0, 30, parent=0),
        node(2, "Пункт 2", 30, 60, parent=0),
        node(2, "Пункт 3", 60, 90, parent=0),
        node(2, "Пункт 4", 90, 120, parent=0),
    ]
    chunks = chunk_document(document(text, contents), CFG)

    assert len(chunks) == 1
    assert chunks[0].segments_merged == 4


# --------------------------------------------------------------------------- #
# Аварийный путь
# --------------------------------------------------------------------------- #


def test_oversized_paragraph_is_split_on_sentence_boundaries():
    """Абзац крупнее окна режется по предложениям, а не посреди слова."""
    paragraph = " ".join(f"Предложение номер {i} с добавочным текстом." for i in range(30))
    chunks = chunk_document(document(paragraph), CFG)

    assert len(chunks) > 1
    assert all(len(c.text) <= CFG.max_chars for c in chunks)
    # Разрезы попали внутрь абзаца — и помечены именно так.
    assert {c.boundary_kind for c in chunks} == {"mid_paragraph"}
    for chunk in chunks[:-1]:
        assert chunk.text.rstrip().endswith(".")


def test_paragraph_without_sentence_boundaries_falls_back_to_window():
    """Если и предложение не влезает — окно с перекрытием, но не молча."""
    paragraph = "щ" * 1000
    record = document(paragraph)
    chunks = chunk_document(record, CFG)

    assert all(len(c.text) <= CFG.max_chars for c in chunks)
    assert all(c.boundary_kind == "mid_paragraph" for c in chunks)
    # Перекрытие есть — соседние куски пересекаются.
    assert chunks[1].char_start < chunks[0].char_end
    _check_offsets(record, chunks)


def test_normal_paragraphs_never_use_the_emergency_path():
    text = "\n".join("к" * 80 for _ in range(20))
    chunks = chunk_document(document(text), CFG)
    assert all(c.boundary_kind != "mid_paragraph" for c in chunks)


# --------------------------------------------------------------------------- #
# Полезная нагрузка
# --------------------------------------------------------------------------- #


def test_payload_carries_the_fields_needed_for_temporal_filtering():
    record = document(
        "к" * 100,
        status="Действует",
        effective_date_begin="2010-01-01",
        effective_date_end=None,
        effective_date_end_sentinel="indefinite",
        doc_type="Постановление",
    )
    payload = chunk_document(record, CFG)[0].payload(record)

    assert payload["chunk_id"] == f"1:{payload['char_start']}-{payload['char_end']}"
    assert payload["effective_date_end_sentinel"] == "indefinite"
    assert payload["is_current"] is True
    assert payload["act_id"] == 10
    assert payload["title"] == "Постановление № 1"


def test_breadcrumb_starts_with_the_act_title():
    text = "к" * 100
    contents = [node(1, "Статья 1", 0, 100)]
    chunk = chunk_document(document(text, contents), CFG)[0]
    assert chunk.breadcrumb == "Постановление № 1 › Статья 1"


def test_empty_text_produces_no_chunks():
    assert chunk_document(document(""), CFG) == []


def test_config_rejects_impossible_settings():
    with pytest.raises(ValueError, match="max_chars"):
        ChunkConfig(target_chars=2000, max_chars=1000)
    with pytest.raises(ValueError, match="перекрытие"):
        ChunkConfig(target_chars=200, max_chars=300, overlap_chars=200)
