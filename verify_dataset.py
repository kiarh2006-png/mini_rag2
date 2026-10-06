from src.mini_rag.evaluation import load_dataset, verify_dataset
from src.mini_rag.vector_store import VectorStore

dataset = load_dataset("data/eval/dataset.json")
chunks = VectorStore().get_all()
problems = verify_dataset(dataset, chunks)

print(f"questions: {len(dataset)}, problems: {len(problems)}")
for item_id, message in problems:
    print(item_id, "-", message)