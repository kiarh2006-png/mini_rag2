import json
from pathlib import Path

from src.mini_rag.normalization import normalize_text

# Spaces, ZWNJ, newlines and tabs are dropped before comparing, because PDF
# extraction spaces Persian text inconsistently.
_DROPPED = str.maketrans("", "", " \u200c\n\t")


def squash(text):
    """Comparison form of a text: normalized, without spaces or ZWNJ."""
    return normalize_text(text).translate(_DROPPED)


def load_dataset(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def is_answerable(item):
    return item["expected_document"] is not None


def find_relevant_chunks(item, chunks):
    """All chunks of the expected document/page that contain the item's evidence."""
    if not is_answerable(item):
        return []
    evidence = squash(item["evidence"])
    return [
        c
        for c in chunks
        if c["metadata"]["document_id"] == item["expected_document"]
        and c["metadata"]["page"] == item["expected_page"]
        and evidence in squash(c["text"])
    ]


def find_chunks_with_absent_terms(item, chunks):
    """Chunk ids that contain any of the item's absent_terms (should be none)."""
    terms = [squash(t) for t in item.get("absent_terms", [])]
    return [
        c["chunk_id"]
        for c in chunks
        if any(t in squash(c["text"]) for t in terms)
    ]


def verify_dataset(dataset, chunks):
    """Return (item_id, message) for every item the index cannot support."""
    problems = []
    for item in dataset:
        if is_answerable(item):
            if not find_relevant_chunks(item, chunks):
                problems.append(
                    (item["id"], "evidence not found in the expected document/page")
                )
        else:
            found_in = find_chunks_with_absent_terms(item, chunks)
            if found_in:
                problems.append(
                    (item["id"], f"absent term found in {len(found_in)} chunks, e.g. {found_in[0]}")
                )
    return problems

def is_relevant_hit(hit, relevant_chunk_ids):
    return hit["chunk_id"] in relevant_chunk_ids


def hit_at_k(retrieved, relevant_chunk_ids, k):
    return int(any(is_relevant_hit(h, relevant_chunk_ids) for h in retrieved[:k]))


def precision_at_k(retrieved, relevant_chunk_ids, k):
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    correct = sum(is_relevant_hit(h, relevant_chunk_ids) for h in top_k)
    return correct / len(top_k)


def recall_at_k(retrieved, relevant_chunk_ids, k):
    if not relevant_chunk_ids:
        return 0.0
    correct = sum(is_relevant_hit(h, relevant_chunk_ids) for h in retrieved[:k])
    return correct / len(relevant_chunk_ids)


def reciprocal_rank(retrieved, relevant_chunk_ids):
    for rank, hit in enumerate(retrieved, start=1):
        if is_relevant_hit(hit, relevant_chunk_ids):
            return 1 / rank
    return 0.0

def relevant_ids(item, chunks):
    return {c["chunk_id"] for c in find_relevant_chunks(item, chunks)}


def evaluate_retrieval(dataset, chunks, embedder, vector_store, top_k=10, k_values=(1, 3, 5)):
    """Run raw vector search (no min_score filter) for every answerable item.

    Returns (per_item_rows, averages_by_k).
    """
    rows = []
    for item in dataset:
        if not is_answerable(item):
            continue
        relevant = relevant_ids(item, chunks)
        query_vector = embedder.embed_query(normalize_text(item["question"]))
        retrieved = vector_store.search(query_vector, top_k=top_k)

        row = {"id": item["id"], "type": item["type"], "relevant_count": len(relevant)}
        for k in k_values:
            row[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        row["mrr"] = reciprocal_rank(retrieved, relevant)
        row["retrieved"] = retrieved
        rows.append(row)

    averages = {}
    for k in k_values:
        averages[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / len(rows)
        averages[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in rows) / len(rows)
        averages[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / len(rows)
    averages["mrr"] = sum(r["mrr"] for r in rows) / len(rows)

    return rows, averages


def evaluate_rejection(dataset, retriever):
    """For unanswerable items, check whether the filtered Retriever correctly
    returns nothing. Returns (per_item_rows, correct_rejection_rate)."""
    rows = []
    for item in dataset:
        if is_answerable(item):
            continue
        hits = retriever.retrieve(item["question"], top_k=5)
        rows.append({"id": item["id"], "correctly_rejected": len(hits) == 0, "hits": hits})

    rate = sum(r["correctly_rejected"] for r in rows) / len(rows) if rows else 0.0
    return rows, rate
def check_faithfulness(answer, chunks):
    """Heuristic groundedness check: what fraction of the answer's sentences
    have notable word-overlap with the retrieved context."""
    context = squash(" ".join(c["text"] for c in chunks))
    sentences = [s.strip() for s in answer.replace("،", ".").split(".") if len(s.strip()) > 8]
    if not sentences:
        return {"grounded_ratio": 1.0, "sentence_count": 0}
    grounded = 0
    for s in sentences:
        words = [w for w in squash(s).split() if len(w) > 2]
        if not words:
            continue
        overlap = sum(1 for w in words if w in context)
        if overlap / len(words) >= 0.5:
            grounded += 1
    return {"grounded_ratio": grounded / len(sentences), "sentence_count": len(sentences)}