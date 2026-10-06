from src.mini_rag.embedding import Embedder
from src.mini_rag.evaluation import load_dataset, normalize_text
from src.mini_rag.vector_store import VectorStore

FAILURE_IDS = ["q06", "q08", "q14", "q17", "q18", "q20", "q26", "q27", "q28", "q32", "q42", "q43", "q44"]

dataset = {item["id"]: item for item in load_dataset("data/eval/dataset.json")}
embedder = Embedder()
store = VectorStore()

for qid in FAILURE_IDS:
    item = dataset[qid]
    query_vector = embedder.embed_query(normalize_text(item["question"]))
    hits = store.search(query_vector, top_k=5)

    print(f"[{qid}] {item['question']}")
    print(f"  expected: {item['expected_document']} / page {item['expected_page']}")
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        mark = "<-- expected doc" if m["document_id"] == item["expected_document"] else ""
        print(f"  {i}. {round(h['score'], 3)} {m['document_id']} p{m['page']} {mark}")
        print(f"     {h['text'][:100]}")
    print()