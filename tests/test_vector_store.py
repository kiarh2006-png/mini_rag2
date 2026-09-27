from src.mini_rag.chunking import Chunk
from src.mini_rag.embedding import Embedder
from src.mini_rag.vector_store import VectorStore


def _chunk(chunk_id, text, page=1):
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        filename="doc1.txt",
        title="سند تست",
        doc_type="law",
        page=page,
        chunk_index=0,
        text=text,
    )


def test_embed_query_and_passages_have_same_dimension():
    embedder = Embedder()
    query_vec = embedder.embed_query("مجازات سرقت چیست؟")
    passage_vecs = embedder.embed_passages(["متن نمونه درباره‌ی سرقت"])

    assert len(query_vec) == len(passage_vecs[0])


def test_store_and_search_returns_most_similar_chunk(tmp_path):
    embedder = Embedder()
    store = VectorStore(path=str(tmp_path / "chroma"))

    chunks = [
        _chunk("c1", "هر کس مال دیگری را بدزدد به حبس محکوم می‌شود"),
        _chunk("c2", "شرایط عقد ازدواج و مهریه در قانون مدنی"),
    ]
    vectors = embedder.embed_passages([c.text for c in chunks])
    store.add_chunks(chunks, vectors)

    query_vec = embedder.embed_query("مجازات سرقت چیست؟")
    hits = store.search(query_vec, top_k=1)

    assert hits[0]["chunk_id"] == "c1"


def test_upsert_overwrites_existing_chunk(tmp_path):
    embedder = Embedder()
    store = VectorStore(path=str(tmp_path / "chroma"))

    chunk_v1 = _chunk("c1", "متن نسخه‌ی اول")
    store.add_chunks([chunk_v1], embedder.embed_passages([chunk_v1.text]))
    assert store.count() == 1

    chunk_v2 = _chunk("c1", "متن نسخه‌ی دوم")
    store.add_chunks([chunk_v2], embedder.embed_passages([chunk_v2.text]))

    assert store.count() == 1  # same chunk_id, not a new entry