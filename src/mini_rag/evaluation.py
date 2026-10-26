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


def check_semantic_groundedness(answer, chunks, embedder, threshold=0.55):
    """Claim-level semantic groundedness: embed each answer sentence and the
    best-matching context chunk, and compare with cosine similarity.

    Unlike word-overlap, this recognizes paraphrases (e.g. "مجازات حبس است"
    vs "حبس در نظر گرفته شده است") as grounded, as long as the meaning
    matches a retrieved chunk closely enough.
    """
    from src.mini_rag.citation import split_sentences

    sentences = split_sentences(answer)
    if not sentences:
        return {"grounded_ratio": 1.0, "claims": []}
    if not chunks:
        return {"grounded_ratio": 0.0, "claims": [{"text": s, "score": 0.0, "grounded": False} for s in sentences]}

    chunk_vectors = embedder.embed_passages([c["text"] for c in chunks])
    sentence_vectors = embedder.embed_passages(sentences)

    claims = []
    for sentence, s_vec in zip(sentences, sentence_vectors):
        best_score = max(_cosine(s_vec, c_vec) for c_vec in chunk_vectors)
        claims.append({"text": sentence, "score": best_score, "grounded": best_score >= threshold})

    grounded_count = sum(c["grounded"] for c in claims)
    return {"grounded_ratio": grounded_count / len(claims), "claims": claims}


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def check_required_facts(answer, required_facts, embedder, threshold=0.7):
    """What fraction of required_facts are present in the answer.

    A fact counts as present if it's a lexical substring of the answer
    (after normalization), OR if its semantic similarity to some answer
    sentence passes `threshold`. Lexical check alone misses paraphrases;
    semantic-only misses short facts (e.g. a single word like "رگرسیون"),
    whose embedding similarity to a full sentence is diluted even when the
    word is literally present. Combining both covers both cases.
    """
    if not required_facts:
        return {"matched_ratio": 1.0, "facts": []}

    answer_squashed = squash(answer)
    answer_sentences = split_sentences_safe(answer)
    answer_vectors = embedder.embed_passages(answer_sentences)
    fact_vectors = embedder.embed_passages(required_facts)

    results = []
    for fact, f_vec in zip(required_facts, fact_vectors):
        lexical_match = squash(fact) in answer_squashed
        best_semantic = max(_cosine(f_vec, a_vec) for a_vec in answer_vectors)
        present = lexical_match or best_semantic >= threshold
        results.append(
            {"fact": fact, "score": best_semantic, "lexical_match": lexical_match, "present": present}
        )

    matched = sum(r["present"] for r in results)
    return {"matched_ratio": matched / len(required_facts), "facts": results}


def split_sentences_safe(text):
    from src.mini_rag.citation import split_sentences
    sentences = split_sentences(text)
    return sentences if sentences else [text]


def classify_answer(answer, required_facts, groundedness_ratio, embedder,
                     no_info_message, unsupported_threshold=0.5, fact_threshold=0.7):
    """CORRECT / PARTIALLY_CORRECT / INCORRECT / UNSUPPORTED / ABSTAINED.

    Groundedness and correctness are checked separately: an answer can be
    grounded but miss key facts (PARTIALLY_CORRECT), or contain the right
    facts while being poorly grounded overall (UNSUPPORTED).
    """
    if answer.strip() == no_info_message.strip():
        return "ABSTAINED", {"matched_ratio": None, "facts": []}

    if groundedness_ratio < unsupported_threshold:
        return "UNSUPPORTED", {"matched_ratio": None, "facts": []}

    fact_check = check_required_facts(answer, required_facts, embedder, fact_threshold)
    ratio = fact_check["matched_ratio"]
    if ratio >= 0.999:
        status = "CORRECT"
    elif ratio > 0:
        status = "PARTIALLY_CORRECT"
    else:
        status = "INCORRECT"
    return status, fact_check