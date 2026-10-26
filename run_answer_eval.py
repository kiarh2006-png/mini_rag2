import json

from src.mini_rag.citation import analyze_citations
from src.mini_rag.embedding import Embedder
from src.mini_rag.evaluation import check_semantic_groundedness, classify_answer, load_dataset
from src.mini_rag.generation import NO_INFO_MESSAGE, generate_answer
from src.mini_rag.retrieval import Retriever
from src.mini_rag.vector_store import VectorStore

dataset = load_dataset("data/eval/dataset.json")
embedder = Embedder()
store = VectorStore()
retriever = Retriever(embedder, store)

answerable_results = []
abstention_results = []

for i, item in enumerate(dataset, start=1):
    print(f"[{i}/{len(dataset)}] {item['id']} {item['type']}: {item['question'][:40]}...")
    hits = retriever.retrieve(item["question"], top_k=5)
    answer, sources = generate_answer(item["question"], hits)

    if item["type"] == "unanswerable":
        correctly_abstained = (answer.strip() == NO_INFO_MESSAGE.strip()) and len(hits) == 0
        abstention_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "generated_answer": answer,
                "retrieved_count": len(hits),
                "correctly_abstained": correctly_abstained,
            }
        )
        continue

    citation_info = analyze_citations(answer, sources)
    groundedness = check_semantic_groundedness(answer, hits, embedder)
    status, fact_check = classify_answer(
        answer, item["required_facts"], groundedness["grounded_ratio"], embedder, NO_INFO_MESSAGE
    )

    answerable_results.append(
        {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "generated_answer": answer,
            "expected_document": item["expected_document"],
            "expected_page": item["expected_page"],
            "retrieved_count": len(hits),
            "status": status,
            "fact_match_ratio": fact_check["matched_ratio"],
            "grounded_ratio": groundedness["grounded_ratio"],
            "citation_coverage": citation_info["citation_coverage"],
            "valid_citation_rate": citation_info["valid_citation_rate"],
            "invalid_citation_rate": citation_info["invalid_citation_rate"],
            "missing_citation_rate": citation_info["missing_citation_rate"],
            "citations": citation_info["citations"],
            "sources": sources,
        }
    )

# --- Summary ---
n = len(answerable_results)
status_counts = {}
for r in answerable_results:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

avg = lambda key: sum(r[key] for r in answerable_results) / n if n else 0.0

print("\n=== Answer Evaluation Summary ===")
print(f"answerable questions: {n}")
for status in ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "UNSUPPORTED"]:
    print(f"  {status}: {status_counts.get(status, 0)}")
print(f"avg grounded_ratio: {avg('grounded_ratio'):.3f}")
print(f"avg citation_coverage: {avg('citation_coverage'):.3f}")
print(f"avg valid_citation_rate: {avg('valid_citation_rate'):.3f}")
print(f"avg invalid_citation_rate: {avg('invalid_citation_rate'):.3f}")
print(f"avg missing_citation_rate: {avg('missing_citation_rate'):.3f}")

correct_abstentions = sum(r["correctly_abstained"] for r in abstention_results)
print(f"\n=== Abstention Summary ===")
print(f"correct_abstention_rate: {correct_abstentions}/{len(abstention_results)}")
print(f"false_answer_rate: {len(abstention_results) - correct_abstentions}/{len(abstention_results)}")

output = {
    "answerable_results": answerable_results,
    "abstention_results": abstention_results,
    "summary": {
        "status_counts": status_counts,
        "avg_grounded_ratio": avg("grounded_ratio"),
        "avg_citation_coverage": avg("citation_coverage"),
        "avg_valid_citation_rate": avg("valid_citation_rate"),
        "avg_invalid_citation_rate": avg("invalid_citation_rate"),
        "avg_missing_citation_rate": avg("missing_citation_rate"),
        "correct_abstention_rate": correct_abstentions / len(abstention_results) if abstention_results else 0,
    },
}
with open("data/eval/answer_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("\nsaved to data/eval/answer_eval_results.json")