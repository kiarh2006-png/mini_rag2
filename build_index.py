from pathlib import Path

from src.mini_rag.extraction import extract_text, PageText
from src.mini_rag.normalization import normalize_text
from src.mini_rag.metadata import load_catalog, build_metadata
from src.mini_rag.chunking import chunk_document
from src.mini_rag.embedding import Embedder
from src.mini_rag.vector_store import VectorStore

RAW_DIR = Path("data/raw")
CATALOG_PATH = "data/catalog.json"


def build_index():
    catalog = load_catalog(CATALOG_PATH)
    embedder = Embedder()
    store = VectorStore()

    paths = sorted(
        p for p in RAW_DIR.iterdir() if p.suffix.lower() in (".pdf", ".txt")
    )

    total_chunks = 0
    for path in paths:
        pages = [
            PageText(p.page, normalize_text(p.text)) for p in extract_text(path)
        ]
        meta = build_metadata(path, catalog)
        chunks = chunk_document(pages, meta)

        if chunks:
            vectors = embedder.embed_passages([c.text for c in chunks])
            store.add_chunks(chunks, vectors)

        total_chunks += len(chunks)
        print(f"{path.name}: {len(pages)} pages, {len(chunks)} chunks")

    print(f"\ntotal: {len(paths)} documents, {total_chunks} chunks")
    print(f"stored in vector store: {store.count()}")


if __name__ == "__main__":
    build_index()