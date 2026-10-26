import json

from src.mini_rag.citation import analyze_citations
from src.mini_rag.embedding import Embedder
from src.mini_rag.evaluation import check_required_facts, classify_answer, load_dataset
from src.mini_rag.generation import NO_INFO_MESSAGE

results = json.load(open("data/eval/answer_eval_results.json", encoding="utf-8"))
dataset = {i["id"]: i for i in load_dataset("data/eval/dataset.json")}
embedder = Embedder()

for r in results["answerable_results"]:
    item = dataset[r["id"]]

    citation_info = analyze_citations(r["generated_answer"], r["sources"])
    r["citation_coverage"] = citation_info["citation_coverage"]
    r["valid_citation_rate"] = citation_info["valid_citation_rate"]
    r["invalid_citation_rate"] = citation_info["invalid_citation_rate"]
    r["missing_citation_rate"] = citation_info["missing_citation_rate"]
    r["citations"] = citation_info["citations"]

    status, fact_check = classify_answer(
        r["generated_answer"], item["required_facts"], r["grounded_ratio"], embedder, NO_INFO_MESSAGE
    )
    r["status"] = status
    r["fact_match_ratio"] = fact_check["matched_ratio"]

n = len(results["answerable_results"])
avg = lambda key: sum(r[key] for r in results["answerable_results"]) / n
status_counts = {}
for r in results["answerable_results"]:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

results["summary"]["status_counts"] = status_counts
for key in ["grounded_ratio", "citation_coverage", "valid_citation_rate", "invalid_citation_rate", "missing_citation_rate"]:
    results["summary"][f"avg_{key}"] = avg(key)

print("status_counts:", status_counts)
for key in ["grounded_ratio", "citation_coverage", "valid_citation_rate", "invalid_citation_rate", "missing_citation_rate"]:
    print(f"avg_{key}: {round(avg(key), 3)}")

json.dump(results, open("data/eval/answer_eval_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated data/eval/answer_eval_results.json")