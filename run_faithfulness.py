import json

from src.mini_rag.embedding import Embedder
from src.mini_rag.evaluation import check_faithfulness, load_dataset
from src.mini_rag.generation import generate_answer
from src.mini_rag.retrieval import Retriever
from src.mini_rag.vector_store import VectorStore

dataset = load_dataset("data/eval/dataset.json")
sample_ids = ["q01", "q06", "q08", "q17", "q20", "q23", "q27", "q32", "q43", "q44"]
dataset = {i["id"]: i for i in dataset}

embedder = Embedder()
store = VectorStore()
retriever = Retriever(embedder, store)

results = []
for qid in sample_ids:
    item = dataset[qid]
    hits = retriever.retrieve(item["question"], top_k=5)
    answer, sources = generate_answer(item["question"], hits)
    check = check_faithfulness(answer, hits)
    cites_expected_doc = any(s["title"] for s in sources)
    results.append({"id": qid, **check, "has_sources": len(sources) > 0})
    print(f"[{qid}] grounded_ratio={check['grounded_ratio']:.2f} sentences={check['sentence_count']} sources={len(sources)}")

avg = sum(r["grounded_ratio"] for r in results) / len(results)
print(f"\naverage grounded_ratio: {avg:.3f}")
json.dump(results, open("data/eval/faithfulness_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)