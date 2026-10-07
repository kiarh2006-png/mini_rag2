import json

from src.mini_rag.embedding import Embedder
from src.mini_rag.evaluation import evaluate_rejection, evaluate_retrieval, load_dataset
from src.mini_rag.retrieval import Retriever
from src.mini_rag.vector_store import VectorStore

dataset = load_dataset("data/eval/dataset.json")
embedder = Embedder()
store = VectorStore(path="data/processed/chroma_v2")
chunks = store.get_all()
retriever = Retriever(embedder, store)

rows, averages = evaluate_retrieval(dataset, chunks, embedder, store)
rejection_rows, rejection_rate = evaluate_rejection(dataset, retriever)

print("=== Retrieval metrics (chunk_size=500, raw ranking) ===")
for name, value in averages.items():
    print(f"{name}: {value:.3f}")

print(f"\n=== Rejection (unanswerable questions) ===")
print(f"correct_rejection_rate: {rejection_rate:.3f} ({sum(r['correctly_rejected'] for r in rejection_rows)}/{len(rejection_rows)})")

output = {
    "averages": averages,
    "rejection_rate": rejection_rate,
    "rows": [{k: v for k, v in r.items() if k != "retrieved"} for r in rows],
    "rejection_rows": [{k: v for k, v in r.items() if k != "hits"} for r in rejection_rows],
}
with open("data/eval/improved_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("\nsaved details to data/eval/improved_results.json")