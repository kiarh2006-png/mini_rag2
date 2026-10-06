from src.mini_rag.embedding import Embedder

embedder = Embedder()
vec = embedder.embed_query("تست")
print("OK", len(vec))