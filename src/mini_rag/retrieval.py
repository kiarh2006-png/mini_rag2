from src.mini_rag.normalization import normalize_text


class Retriever:
    """Ties together query normalization, embedding, and vector search."""

    def __init__(self, embedder, vector_store, min_score=0.6):
        self.embedder = embedder
        self.vector_store = vector_store
        self.min_score = min_score

    def retrieve(self, query, top_k=5, doc_type=None):
        """Return the top_k relevant chunks for a query, or [] if none pass min_score."""
        normalized = normalize_text(query)
        query_vector = self.embedder.embed_query(normalized)

        where = {"doc_type": doc_type} if doc_type else None
        hits = self.vector_store.search(query_vector, top_k=top_k, where=where)

        return [h for h in hits if h["score"] >= self.min_score]