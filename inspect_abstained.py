import json

results = json.load(open("data/eval/answer_eval_results.json", encoding="utf-8"))
rows = results["answerable_results"]

for status in ["INCORRECT", "PARTIALLY_CORRECT", "UNSUPPORTED"]:
    print(f"--- {status} ---")
    for r in [x for x in rows if x["status"] == status][:2]:
        print(f"[{r['id']}] {r['question']}")
        print(f"  expected: {r['expected_answer']}")
        print(f"  generated: {r['generated_answer']}")
        print(f"  grounded_ratio={r['grounded_ratio']:.2f} fact_match_ratio={r['fact_match_ratio']}")
        print()

invalid = [r for r in rows if r["invalid_citation_rate"] > 0]
print(f"--- invalid citation examples ({len(invalid)} total) ---")
for r in invalid[:2]:
    print(f"[{r['id']}] {r['question']}")
    print(f"  generated: {r['generated_answer']}")
    print(f"  citations: {r['citations']}")
    print()