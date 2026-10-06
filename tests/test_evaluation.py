from src.mini_rag.evaluation import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _hit(chunk_id):
    return {"chunk_id": chunk_id}


def test_hit_at_k_true_when_relevant_chunk_in_top_k():
    retrieved = [_hit("a"), _hit("b"), _hit("c")]
    assert hit_at_k(retrieved, {"c"}, k=3) == 1
    assert hit_at_k(retrieved, {"c"}, k=1) == 0


def test_hit_at_k_false_when_no_relevant_chunk_present():
    retrieved = [_hit("a"), _hit("b")]
    assert hit_at_k(retrieved, {"z"}, k=5) == 0


def test_precision_at_k_counts_correct_fraction():
    retrieved = [_hit("a"), _hit("b"), _hit("c"), _hit("d")]
    assert precision_at_k(retrieved, {"a", "c"}, k=4) == 0.5


def test_precision_at_k_empty_retrieved_is_zero():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_recall_at_k_uses_total_relevant_as_denominator():
    retrieved = [_hit("a"), _hit("x"), _hit("y")]
    # 3 relevant chunks exist in total, only 1 was retrieved
    assert recall_at_k(retrieved, {"a", "b", "c"}, k=3) == 1 / 3


def test_recall_at_k_no_relevant_chunks_is_zero():
    assert recall_at_k([_hit("a")], set(), k=5) == 0.0


def test_reciprocal_rank_uses_first_correct_position():
    retrieved = [_hit("x"), _hit("y"), _hit("a")]
    assert reciprocal_rank(retrieved, {"a"}) == 1 / 3


def test_reciprocal_rank_zero_when_never_found():
    retrieved = [_hit("x"), _hit("y")]
    assert reciprocal_rank(retrieved, {"a"}) == 0.0


def test_hit_at_1_zero_but_hit_at_5_one_moves_mrr_accordingly():
    # relevant chunk is at rank 4: Hit@1=0, Hit@5=1, MRR=1/4
    retrieved = [_hit("x"), _hit("y"), _hit("z"), _hit("a")]
    assert hit_at_k(retrieved, {"a"}, k=1) == 0
    assert hit_at_k(retrieved, {"a"}, k=5) == 1
    assert reciprocal_rank(retrieved, {"a"}) == 1 / 4