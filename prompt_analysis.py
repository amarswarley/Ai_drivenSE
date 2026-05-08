"""
Patched prompt analysis for the ablation study.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from augmentation_pipeline_patched import AugmentationPipeline

OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

PROBLEMS_FILE = Path("biocoder_dataset/python_problems.json")
if PROBLEMS_FILE.exists():
    with open(PROBLEMS_FILE, encoding="utf-8") as f:
        PROBLEMS = json.load(f)
    SAMPLE = PROBLEMS[0]
else:
    PROBLEMS = []
    SAMPLE = {
        "id": "sample",
        "signature": "def gc_content(sequence: str) -> float:",
        "docstring": "Return the GC content of a DNA sequence as a fraction between 0.0 and 1.0.",
        "code": "def gc_content(sequence: str) -> float:\n    return (sequence.count('G') + sequence.count('C')) / len(sequence)",
        "tests": [],
    }

CONDITIONS = [
    "C1_zero_shot",
    "C2_few_shot",
    "C3_chain_of_thought",
    "C4_rag_context",
    "C5_iterative_repair",
]

CONDITION_LABELS = {
    "C1_zero_shot": "C1 Zero-Shot",
    "C2_few_shot": "C2 Few-Shot",
    "C3_chain_of_thought": "C3 Chain-of-Thought",
    "C4_rag_context": "C4 RAG Context",
    "C5_iterative_repair": "C5 Iterative Repair (initial prompt)",
}

AUGMENTATION_ADDED = {
    "C1_zero_shot": "None (baseline)",
    "C2_few_shot": "3 in-domain bioinformatics examples",
    "C3_chain_of_thought": "Step-by-step reasoning elicitation",
    "C4_rag_context": "API documentation + retrieved similar functions (patched RAG, no extra CoT wording)",
    "C5_iterative_repair": "CoT initial prompt; patched experiment also adds repair-round feedback tokens",
}


def build_prompt(condition: str) -> str:
    pipeline = AugmentationPipeline(condition)
    pool = PROBLEMS if PROBLEMS else [SAMPLE]
    if condition == "C4_rag_context":
        fallback_examples = [p for p in pool if p.get("id") != SAMPLE.get("id")][:3]
        if not fallback_examples:
            fallback_examples = [SAMPLE]
        pipeline._retrieve_similar = lambda problem, all_problems, top_k=3: fallback_examples[:top_k]
    return pipeline.generate_prompt(problem=SAMPLE, all_problems=pool)["prompt"]


results = []
for cond in CONDITIONS:
    prompt = build_prompt(cond)
    words = len(prompt.split())
    chars = len(prompt)
    lines = prompt.count("\n") + 1
    est_tokens = int(words / 0.75)
    results.append(
        {
            "condition": cond,
            "label": CONDITION_LABELS[cond],
            "augmentation": AUGMENTATION_ADDED[cond],
            "word_count": words,
            "char_count": chars,
            "line_count": lines,
            "est_tokens": est_tokens,
            "measures_initial_prompt_only": cond == "C5_iterative_repair",
            "uses_patched_pipeline": True,
        }
    )

c1_words = results[0]["word_count"]
c1_tokens = results[0]["est_tokens"]
for row in results:
    row["overhead_words"] = row["word_count"] - c1_words
    row["overhead_tokens"] = row["est_tokens"] - c1_tokens
    row["overhead_pct"] = round((row["word_count"] / c1_words - 1) * 100, 1)

csv_path = OUTPUT_DIR / "prompt_lengths.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(
        "Condition,Label,Words,Est_Tokens,Overhead_Words,Overhead_Tokens,"
        "Overhead_Pct,Measures_Initial_Prompt_Only,Augmentation_Added\n"
    )
    for row in results:
        f.write(
            f"{row['condition']},{row['label']},{row['word_count']},{row['est_tokens']},"
            f"{row['overhead_words']},{row['overhead_tokens']},{row['overhead_pct']}%,"
            f"{str(row['measures_initial_prompt_only'])},\"{row['augmentation']}\"\n"
        )

json_path = OUTPUT_DIR / "prompt_analysis.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

txt_path = OUTPUT_DIR / "prompt_comparison.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("PROMPT ANALYSIS - PATCHED AUGMENTATION CONDITIONS\n")
    f.write(f"Sample problem: {SAMPLE['signature']}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'Condition':<30} {'Words':>6} {'~Tokens':>8} {'vs C1':>8}\n")
    f.write("-" * 56 + "\n")
    for row in results:
        overhead = f"+{row['overhead_pct']}%" if row["overhead_pct"] > 0 else "baseline"
        f.write(f"  {row['label']:<28} {row['word_count']:>6} {row['est_tokens']:>8} {overhead:>8}\n")

    f.write("\n\nPATCHED PIPELINE NOTES:\n")
    f.write("-" * 56 + "\n")
    f.write("C4 RAG no longer appends explicit chain-of-thought wording.\n")
    f.write("C5 values here describe only the initial prompt. In the patched runner,\n")
    f.write("repair-round prompt and completion tokens are added to the final C5 cost.\n")

    f.write("\n\nFULL PROMPTS\n")
    f.write("=" * 70 + "\n\n")
    for row in results:
        prompt = build_prompt(row["condition"])
        f.write(f"\n{'-' * 60}\n{row['label']} ({row['word_count']} words, ~{row['est_tokens']} tokens)\n{'-' * 60}\n")
        f.write(prompt)
        f.write("\n")

print(f"Saved: {csv_path}")
print(f"Saved: {json_path}")
print(f"Saved: {txt_path}")
