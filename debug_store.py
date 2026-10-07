from src.mini_rag.chunking import Chunk
from src.mini_rag.embedding import Embedder
from src.mini_rag.vector_store import VectorStore

chunk = Chunk(
    chunk_id="debug-0000",
    document_id="debug",
    filename="debug.txt",
    title="تست",
    doc_type="test",
    page=1,
    chunk_index=0,
    text="این یک متن آزمایشی است",
)

embedder = Embedder()
vectors = embedder.embed_passages([chunk.text])
print("vector type:", type(vectors), type(vectors[0]) if vectors else None)
print("vector length:", len(vectors[0]) if vectors else 0)

store = VectorStore(path="data/processed/chroma_v2")
print("count before:", store.count())
store.add_chunks([chunk], vectors)
print("count after:", store.count())
