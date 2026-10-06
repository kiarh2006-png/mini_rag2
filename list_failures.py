import json

from src.mini_rag.evaluation import load_dataset

dataset = {item["id"]: item for item in load_dataset("data/eval/dataset.json")}
results = json.load(open("data/eval/baseline_results.json", encoding="utf-8"))

failures = [r for r in results["rows"] if r["hit@1"] == 0]
print(f"{len(failures)} questions with hit@1=0 (out of {len(results['rows'])})\n")

for row in failures:
    item = dataset[row["id"]]
    print(f"[{row['id']}] type={row['type']} hit@5={row['hit@5']} mrr={row['mrr']:.2f}")
    print(f"  question: {item['question']}")
    print(f"  expected: {item['expected_document']} / page {item['expected_page']}")
    print(f"  evidence: {item['evidence']}")
    print()