"""Тесты конвейера: порядок стадий и то, что каждая делает заявленное."""

from __future__ import annotations

from mlaw.evaluate import Hit
from mlaw.search import SearchPipeline


def hits(spec: list[tuple[str, int, int]]) -> list[Hit]:
    return [Hit(chunk_id=c, doc_id=d, act_id=a, score=1.0) for c, d, a in spec]


def test_cap_per_act_limits_but_keeps_order():
    given = hits([("c1", 1, 10), ("c2", 1, 10), ("c3", 1, 10), ("c4", 2, 20)])
    capped = SearchPipeline.cap_per_act(given, 2)
    assert [h.chunk_id for h in capped] == ["c1", "c2", "c4"]


def test_collapse_keeps_first_per_act():
    """После реранка первый фрагмент акта — лучший, его и оставляем."""
    given = hits([("best", 1, 10), ("worse", 1, 10), ("other", 2, 20)])
    assert [h.chunk_id for h in SearchPipeline.collapse_by_act(given)] == ["best", "other"]


def test_collapse_after_rerank_differs_from_collapse_before():
    """Порядок стадий не косметика: схлопывание до и после реранка даёт разное.

    До реранка от акта остался бы 'a_weak' (он выше по слиянию), после —
    'a_strong', выбранный кросс-энкодером.
    """
    fused = hits([("a_weak", 1, 10), ("a_strong", 1, 10), ("b", 2, 20)])
    before = SearchPipeline.collapse_by_act(fused)
    reranked = hits([("a_strong", 1, 10), ("b", 2, 20), ("a_weak", 1, 10)])
    after = SearchPipeline.collapse_by_act(reranked)
    assert [h.chunk_id for h in before] == ["a_weak", "b"]
    assert [h.chunk_id for h in after] == ["a_strong", "b"]


def test_fusion_rewards_agreement_between_legs():
    """Документ, найденный обеими ногами, обязан обойти найденный одной."""
    dense = hits([("both", 1, 10), ("only_dense", 2, 20)])
    lexical = hits([("only_lex", 3, 30), ("both", 1, 10)])
    fused = SearchPipeline.fuse([dense, lexical])
    assert fused[0].chunk_id == "both"


def test_fusion_is_rank_based_not_score_based():
    """Шкалы косинуса и BM25 несопоставимы, поэтому вес score игнорируется."""
    a = [Hit("x", 1, 10, score=0.001)]
    b = [Hit("y", 2, 20, score=9999.0)]
    fused = SearchPipeline.fuse([a, b])
    assert {h.score for h in fused} == {1.0 / 61}


def test_empty_pools_are_safe():
    assert SearchPipeline.fuse([[], []]) == []
    assert SearchPipeline.cap_per_act([], 3) == []
    assert SearchPipeline.collapse_by_act([]) == []
