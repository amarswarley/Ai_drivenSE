"""
Problem complexity analysis for the 20-problem BioCoder subset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

PROBLEMS_FILE = Path("biocoder_dataset/python_problems.json")
if not PROBLEMS_FILE.exists():
    raise SystemExit(f"ERROR: {PROBLEMS_FILE} not found.")

with open(PROBLEMS_FILE, encoding="utf-8") as f:
    problems = json.load(f)


def count_lines(code: str) -> int:
    return len([line for line in code.splitlines() if line.strip()])


def count_branches(code: str) -> int:
    return sum(code.count(keyword) for keyword in ["if ", "elif ", "for ", "while ", "try:"])


def count_parameters(signature: str) -> int:
    match = re.match(r"def\s+\w+\s*\(([^)]*)\)", signature)
    if not match:
        return 0
    params = [p.strip() for p in match.group(1).split(",") if p.strip() and p.strip() != "self"]
    return len(params)


def has_recursion(code: str, signature: str) -> bool:
    fn_name = re.match(r"def\s+(\w+)", signature)
    if not fn_name:
        return False
    body = "\n".join(code.splitlines()[1:])
    return fn_name.group(1) + "(" in body


def uses_stdlib(code: str) -> bool:
    return "import" in code or any(
        marker in code
        for marker in [
            "math.",
            "collections.",
            "itertools.",
            "functools.",
            "sorted(",
            "enumerate(",
            "zip(",
            "map(",
            "filter(",
        ]
    )


def has_nested_loops(code: str) -> bool:
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"^\s{4,}for |^\s{4,}while ", line):
            for prev in lines[:i]:
                if re.search(r"for |while ", prev):
                    return True
    return False


def return_type_complexity(signature: str) -> int:
    sig = signature.lower()
    if "-> dict" in sig:
        return 3
    if "-> list" in sig or "-> tuple" in sig:
        return 2
    return 1


analysed = []
for problem in problems:
    code = problem.get("code", "")
    signature = problem.get("signature", "")
    docstring = problem.get("docstring", "")

    score = (
        min(count_lines(code) / 5, 3)
        + min(count_branches(code) * 0.5, 2)
        + (1 if uses_stdlib(code) else 0)
        + (1 if has_nested_loops(code) else 0)
        + (1 if has_recursion(code, signature) else 0)
        + min(return_type_complexity(signature) - 1, 1)
        + min(count_parameters(signature) / 3, 1)
    )

    difficulty = "Easy" if score < 2.5 else "Medium" if score < 4.5 else "Hard"

    analysed.append(
        {
            **problem,
            "lines": count_lines(code),
            "branches": count_branches(code),
            "params": count_parameters(signature),
            "recursive": has_recursion(code, signature),
            "uses_stdlib": uses_stdlib(code),
            "nested_loops": has_nested_loops(code),
            "doc_words": len(docstring.split()),
            "test_count": len(problem.get("tests", [])),
            "ret_complex": return_type_complexity(signature),
            "difficulty_score": round(score, 2),
            "difficulty": difficulty,
        }
    )

csv_path = OUTPUT_DIR / "problem_complexity.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(
        "ID,Category,Difficulty,Score,Lines,Branches,Params,"
        "Recursive,UsesStdlib,NestedLoops,ReturnComplexity,Tests,DocWords\n"
    )
    for row in analysed:
        f.write(
            f"{row['id']},{row['category']},{row['difficulty']},{row['difficulty_score']},"
            f"{row['lines']},{row['branches']},{row['params']},"
            f"{int(row['recursive'])},{int(row['uses_stdlib'])},{int(row['nested_loops'])},"
            f"{row['ret_complex']},{row['test_count']},{row['doc_words']}\n"
        )

txt_path = OUTPUT_DIR / "category_summary.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("PROBLEM COMPLEXITY ANALYSIS BY CATEGORY\n")
    f.write("=" * 70 + "\n\n")

    categories = {}
    for row in analysed:
        categories.setdefault(row["category"], []).append(row)

    for category, items in categories.items():
        scores = [item["difficulty_score"] for item in items]
        dist = {"Easy": 0, "Medium": 0, "Hard": 0}
        for item in items:
            dist[item["difficulty"]] += 1

        f.write(f"Category: {category}\n")
        f.write(f"  Problems:  {len(items)}\n")
        f.write(f"  Avg score: {sum(scores)/len(scores):.2f}/10\n")
        f.write(f"  Difficulty distribution: {dist}\n")
        f.write("  Problems:\n")
        for item in sorted(items, key=lambda x: x["difficulty_score"]):
            f.write(f"    [{item['difficulty']:<6} {item['difficulty_score']:>4}] {item['id']}: {item['signature'][:50]}\n")
        f.write("\n")

    all_scores = [row["difficulty_score"] for row in analysed]
    all_dist = {"Easy": 0, "Medium": 0, "Hard": 0}
    for row in analysed:
        all_dist[row["difficulty"]] += 1

    f.write("=" * 70 + "\n")
    f.write("OVERALL SUMMARY\n")
    f.write("=" * 70 + "\n")
    f.write(f"Total problems: {len(analysed)}\n")
    f.write(f"Mean difficulty score: {sum(all_scores)/len(all_scores):.2f}/10\n")
    f.write(f"Difficulty distribution: {all_dist}\n")
    f.write(f"  Easy:   {all_dist['Easy']} ({100*all_dist['Easy']/len(analysed):.0f}%)\n")
    f.write(f"  Medium: {all_dist['Medium']} ({100*all_dist['Medium']/len(analysed):.0f}%)\n")
    f.write(f"  Hard:   {all_dist['Hard']} ({100*all_dist['Hard']/len(analysed):.0f}%)\n")
    f.write("\nUSE IN PAPER:\n")
    f.write("  Section 3.2 (Dataset): use the difficulty distribution to justify that\n")
    f.write("  the benchmark spans simple sequence manipulation, moderate utility tasks,\n")
    f.write("  and a small set of harder algorithmic/statistical problems.\n")
    f.write("  Section 5 (Discussion): correlate per-problem difficulty scores with pass\n")
    f.write("  rates to test whether augmentation helps, hurts, or has no consistent effect\n")
    f.write("  across complexity bands.\n")

print(f"Saved: {csv_path}")
print(f"Saved: {txt_path}")
