"""Тесты метрик и контролей.

Проверяется не «сколько получилось», а что метрика считает то, что заявлено:
ранг берётся первого совпадения, уровни различаются, а контроль умеет
обвалить число.
"""

from __future__ import annotations

from mlaw.evaluate import Hit, QueryResult, metrics_for, shuffled_gold


def hit(chunk_id: str, doc_id: int, act_id: int) -> Hit:
    return Hit(chunk_id=chunk_id, doc_id=doc_id, act_id=act_id, score=1.0)


def result(hits: list[Hit], qid: str = "q1", qtype: str = "manual") -> QueryResult:
    return QueryResult(query_id=qid, type=qtype, hits=hits)


def test_perfect_hit_at_rank_one():
    res = [result([hit("c1", 10, 100)])]
    golds = {"q1": {"act_id": 100, "doc_id": 10, "chunk_ids": ["c1"]}}
    m = metrics_for(res, golds)
    for level in ("act", "doc", "chunk"):
        assert m[level]["recall@1"] == 1.0
        assert m[level]["mrr"] == 1.0


def test_rank_matters_for_mrr_but_not_for_recall_at_10():
    hits = [hit(f"c{i}", i, i) for i in range(1, 5)] + [hit("gold", 99, 999)]
    res = [result(hits)]
    golds = {"q1": {"act_id": 999, "doc_id": 99, "chunk_ids": ["gold"]}}
    m = metrics_for(res, golds)
    assert m["act"]["recall@1"] == 0.0
    assert m["act"]["recall@10"] == 1.0
    assert m["act"]["mrr"] == round(1 / 5, 4)


def test_levels_are_independent():
    """Правильный акт, но не та редакция — уровень акта слеп, уровень doc нет."""
    res = [result([hit("other", 11, 100)])]
    golds = {"q1": {"act_id": 100, "doc_id": 10, "chunk_ids": ["c1"]}}
    m = metrics_for(res, golds)
    assert m["act"]["recall@1"] == 1.0
    assert m["doc"]["recall@1"] == 0.0
    assert m["chunk"]["recall@1"] == 0.0


def test_queries_without_chunk_gold_are_excluded_from_chunk_level():
    """У запроса по реквизитам эталона на уровне чанка нет — он не должен
    ни портить, ни улучшать метрику чанка."""
    res = [result([hit("c1", 10, 100)], "q1"), result([hit("c2", 20, 200)], "q2")]
    golds = {
        "q1": {"act_id": 100, "doc_id": 10, "chunk_ids": ["c1"]},
        "q2": {"act_id": 200, "doc_id": 20, "chunk_ids": []},
    }
    m = metrics_for(res, golds)
    assert m["act"]["queries"] == 2
    assert m["chunk"]["queries"] == 1
    assert m["chunk"]["recall@1"] == 1.0


def test_empty_result_scores_zero():
    res = [result([])]
    golds = {"q1": {"act_id": 100, "doc_id": 10, "chunk_ids": ["c1"]}}
    m = metrics_for(res, golds)
    assert m["act"]["recall@10"] == 0.0
    assert m["act"]["mrr"] == 0.0


def test_ndcg_decreases_with_rank():
    top = metrics_for([result([hit("g", 1, 1)])], {"q1": {"act_id": 1}})
    deep = metrics_for(
        [result([hit("x", 2, 2), hit("y", 3, 3), hit("g", 1, 1)])], {"q1": {"act_id": 1}}
    )
    assert top["act"]["ndcg@10"] > deep["act"]["ndcg@10"] > 0


def test_shuffled_gold_actually_changes_the_gold():
    """Контроль обязан подменить эталон, а не вернуть тот же самый."""
    queries = [
        {"query_id": f"q{i}", "type": "manual",
         "gold": {"act_id": i, "doc_id": i * 10, "chunk_ids": [f"c{i}"]}}
        for i in range(1, 11)
    ]
    shuffled = shuffled_gold(queries, seed=1)
    same = sum(1 for q in queries if shuffled[q["query_id"]]["act_id"] == q["gold"]["act_id"])
    assert same == 0, "подменённый эталон нигде не должен совпасть с настоящим"


def test_shuffled_gold_collapses_a_perfect_run():
    queries = [
        {"query_id": f"q{i}", "type": "manual",
         "gold": {"act_id": i, "doc_id": i * 10, "chunk_ids": [f"c{i}"]}}
        for i in range(1, 11)
    ]
    golds = {q["query_id"]: q["gold"] for q in queries}
    results = [
        result([hit(f"c{i}", i * 10, i)], f"q{i}") for i in range(1, 11)
    ]
    assert metrics_for(results, golds)["act"]["recall@1"] == 1.0
    assert metrics_for(results, shuffled_gold(queries, seed=1))["act"]["recall@10"] == 0.0
