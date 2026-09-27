from src.mini_rag.embedding import Embedder
from src.mini_rag.vector_store import VectorStore
from src.mini_rag.retrieval import Retriever

emb = Embedder()
store = VectorStore()
retriever = Retriever(emb, store)

queries = [
    "مجازات سرقت چیست؟",
    "قوانین طلاق در ایران چیست؟",
    "هوش مصنوعی چیست؟",
    "قوانین بازی فوتبال چیست؟",
    "قیمت بیت‌کوین امروز چند است؟",
]

for query in queries:
    hits = retriever.retrieve(query, top_k=1)
    if hits:
        print("OK", round(hits[0]["score"], 3), hits[0]["metadata"]["title"], "-", query)
    else:
        print("REJECTED -", query)