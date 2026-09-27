from src.mini_rag.extraction import extract_text
from src.mini_rag.normalization import normalize_text
from src.mini_rag.metadata import build_metadata
from src.mini_rag.chunking import chunk_document
from src.mini_rag.extraction import PageText
from src.mini_rag.embedding import Embedder
from src.mini_rag.vector_store import VectorStore
from src.mini_rag.retrieval import Retriever
from src.mini_rag.generation import generate_answer, NO_INFO_MESSAGE


def _build_pipeline(tmp_path, file_text):
    file_path = tmp_path / "sample_law.txt"
    file_path.write_text(file_text, encoding="utf-8")

    catalog = {
        "sample_law.txt": {
            "title": "قانون آزمایشی",
            "doc_type": "law",
            "source": "test",
        }
    }

    pages = [
        PageText(p.page, normalize_text(p.text)) for p in extract_text(file_path)
    ]
    meta = build_metadata(file_path, catalog)
    chunks = chunk_document(pages, meta)

    embedder = Embedder()
    store = VectorStore(path=str(tmp_path / "chroma"))
    vectors = embedder.embed_passages([c.text for c in chunks])
    store.add_chunks(chunks, vectors)

    retriever = Retriever(embedder, store, min_score=0.5)
    return retriever


def test_full_pipeline_answers_relevant_question_with_citation(tmp_path):
    retriever = _build_pipeline(
        tmp_path,
        "ماده ۱: هر کس مال دیگری را بدزدد به یک تا پنج سال حبس محکوم می‌شود.",
    )

    hits = retriever.retrieve("مجازات سرقت چیست؟", top_k=3)
    answer, sources = generate_answer("مجازات سرقت چیست؟", hits)

    assert len(hits) > 0
    assert answer != NO_INFO_MESSAGE
    assert len(sources) > 0
    assert sources[0]["title"] == "قانون آزمایشی"


def test_full_pipeline_declines_unrelated_question(tmp_path):
    retriever = _build_pipeline(
        tmp_path,
        "ماده ۱: هر کس مال دیگری را بدزدد به یک تا پنج سال حبس محکوم می‌شود.",
    )

    hits = retriever.retrieve("دستور پخت لازانیا چیست؟", top_k=3)
    answer, sources = generate_answer("دستور پخت لازانیا چیست؟", hits)

    assert hits == []
    assert answer == NO_INFO_MESSAGE
    assert sources == []