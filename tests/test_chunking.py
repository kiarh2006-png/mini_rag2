from src.mini_rag.chunking import Chunk, chunk_document, split_into_chunks
from src.mini_rag.extraction import PageText
from src.mini_rag.metadata import DocumentMetadata


def _sample_metadata():
    return DocumentMetadata(
        document_id="doc1",
        filename="doc1.txt",
        file_type="txt",
        doc_type="law",
        title="سند نمونه",
        source="test",
    )


def test_split_respects_chunk_size():
    text = "\n".join(f"خط شماره {i} با چند کلمه‌ی دیگر برای طول بیشتر" for i in range(50))
    chunks = split_into_chunks(text, chunk_size=200, overlap=30)
    assert all(len(c) <= 200 for c in chunks)


def test_split_produces_overlap_between_consecutive_chunks():
    text = "\n".join(f"خط {i} با کمی متن اضافه برای رسیدن به طول کافی" for i in range(30))
    chunks = split_into_chunks(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # the end of one chunk should share some text with the start of the next
    assert chunks[0][-20:] in chunks[1] or chunks[1].startswith(chunks[0][-20:])


def test_single_short_text_produces_one_chunk():
    chunks = split_into_chunks("یک متن کوتاه", chunk_size=1000, overlap=150)
    assert chunks == ["یک متن کوتاه"]


def test_chunk_document_never_crosses_page_boundary():
    long_line = "کلمه " * 100
    pages = [
        PageText(page=1, text=long_line),
        PageText(page=2, text=long_line),
    ]
    chunks = chunk_document(pages, _sample_metadata(), chunk_size=100, overlap=20)

    page1_chunks = [c for c in chunks if c.page == 1]
    page2_chunks = [c for c in chunks if c.page == 2]
    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0
    assert all("کلمه" in c.text for c in page1_chunks + page2_chunks)


def test_chunk_ids_are_unique_and_sequential():
    pages = [PageText(page=1, text="متن " * 100)]
    chunks = chunk_document(pages, _sample_metadata(), chunk_size=50, overlap=10)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "doc1-0000"


def test_chunk_carries_document_metadata():
    pages = [PageText(page=1, text="یک متن ساده")]
    chunks = chunk_document(pages, _sample_metadata())

    assert chunks[0].document_id == "doc1"
    assert chunks[0].title == "سند نمونه"
    assert chunks[0].doc_type == "law"