from src.mini_rag.chunking import Chunk
from src.mini_rag.embedding import Embedder
from src.mini_rag.retrieval import Retriever
from src.mini_rag.vector_store import VectorStore


def _chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        filename="doc1.txt",
        title="سند تست",
        doc_type="law",
        page=1,
        chunk_index=0,
        text=text,
    )


def _seeded_retriever(tmp_path, min_score):
    embedder = Embedder()
    store = VectorStore(path=str(tmp_path / "chroma"))

    chunks = [
        _chunk("c1", "هر کس مال دیگری را بدزدد به حبس محکوم می‌شود"),
        _chunk("c2", "مجازات سرقت مسلحانه شدیدتر از سرقت عادی است"),
    ]
    vectors = embedder.embed_passages([c.text for c in chunks])
    store.add_chunks(chunks, vectors)

    return Retriever(embedder, store, min_score=min_score)


def test_retrieve_returns_relevant_chunks(tmp_path):
    retriever = _seeded_retriever(tmp_path, min_score=0.5)
    hits = retriever.retrieve("مجازات سرقت چیست؟", top_k=2)

    assert len(hits) > 0
    assert all(h["score"] >= 0.5 for h in hits)


def test_retrieve_respects_top_k(tmp_path):
    retriever = _seeded_retriever(tmp_path, min_score=0.0)
    hits = retriever.retrieve("مجازات سرقت چیست؟", top_k=1)

    assert len(hits) == 1


def test_retrieve_returns_empty_when_score_below_threshold(tmp_path):
    # min_score=1.01 is unreachable (scores are cosine similarity, max 1.0)
    retriever = _seeded_retriever(tmp_path, min_score=1.01)
    hits = retriever.retrieve("مجازات سرقت چیست؟", top_k=2)

    assert hits == []