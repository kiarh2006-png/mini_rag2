import chromadb


class VectorStore:
    """Wraps a persistent Chroma collection for chunk storage and search."""

    def __init__(self, path="data/processed/chroma", collection_name="mini_rag"):
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks, embeddings):
        """Add or overwrite chunks. embeddings must be in the same order as chunks."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=list(embeddings),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "title": c.title,
                    "doc_type": c.doc_type,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def count(self):
        return self._collection.count()

    def search(self, query_embedding, top_k=5, where=None):
        """Return the top_k nearest chunks to query_embedding.

        `where` is an optional Chroma metadata filter, e.g. {"doc_type": "law"}.
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        hits = []
        for i in range(len(result["ids"][0])):
            hits.append(
                {
                    "chunk_id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    # Chroma returns cosine distance; convert to similarity
                    "score": 1 - result["distances"][0][i],
                }
            )
        return hits