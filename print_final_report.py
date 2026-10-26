import json

results = json.load(open("data/eval/answer_eval_results.json", encoding="utf-8"))
s = results["summary"]
sc = s["status_counts"]
n = len(results["answerable_results"])
m = len(results["abstention_results"])

lines = []
lines.append("```")
lines.append(f"Dataset: {n} answerable, {m} unanswerable")
lines.append(f"CORRECT: {sc.get('CORRECT',0)}  PARTIALLY_CORRECT: {sc.get('PARTIALLY_CORRECT',0)}  "
              f"INCORRECT: {sc.get('INCORRECT',0)}  UNSUPPORTED: {sc.get('UNSUPPORTED',0)}  ABSTAINED: {sc.get('ABSTAINED',0)}")
lines.append(f"avg_grounded_ratio: {s['avg_grounded_ratio']:.3f}")
lines.append(f"avg_citation_coverage: {s['avg_citation_coverage']:.3f}")
lines.append(f"avg_valid_citation_rate: {s['avg_valid_citation_rate']:.3f}")
lines.append(f"avg_invalid_citation_rate: {s['avg_invalid_citation_rate']:.3f}")
lines.append(f"avg_missing_citation_rate: {s['avg_missing_citation_rate']:.3f}")
lines.append(f"correct_abstention_rate: {s['correct_abstention_rate']:.3f} "
              f"({sum(r['correctly_abstained'] for r in results['abstention_results'])}/{m})")
lines.append("```")
lines.append("")
lines.append("| Metric | Result |")
lines.append("|---|---|")
lines.append(f"| Hit@1 (از ارزیابی قبلی) | 0.705 / 0.682 (Baseline/Improved) |")
lines.append(f"| Answer Correct | {sc.get('CORRECT',0)}/{n} |")
lines.append(f"| Answer Partially Correct | {sc.get('PARTIALLY_CORRECT',0)}/{n} |")
lines.append(f"| Answer Incorrect | {sc.get('INCORRECT',0)}/{n} |")
lines.append(f"| Answer Unsupported | {sc.get('UNSUPPORTED',0)}/{n} |")
lines.append(f"| Answer Wrongly Abstained | {sc.get('ABSTAINED',0)}/{n} |")
lines.append(f"| avg Groundedness | {s['avg_grounded_ratio']:.3f} |")
lines.append(f"| Citation Coverage | {s['avg_citation_coverage']:.3f} |")
lines.append(f"| Valid Citation Rate | {s['avg_valid_citation_rate']:.3f} |")
lines.append(f"| Correct Abstention (true unanswerable) | {s['correct_abstention_rate']:.3f} |")

text = "\n".join(lines)
print(text)
open("data/eval/report_section.md", "w", encoding="utf-8").write(text)
print("\nsaved to data/eval/report_section.md")