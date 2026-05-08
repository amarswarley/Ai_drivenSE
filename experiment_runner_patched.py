"""
Patched experiment runner copy.

This wrapper keeps the original runner untouched while fixing:
1. Fragile code extraction for CoT/RAG/repair outputs.
2. C5 repair token accounting.
3. Reporting of initial vs final repair success.
4. Use of the patched augmentation pipeline copy.
"""

import argparse
import ast
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import experiment_runner_last as base_runner

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    model: str
    condition: str
    problem_id: str
    prompt_tokens: int
    completion_tokens: int
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    attempts: int = 1
    initial_passed: Optional[bool] = None
    repair_prompt_tokens: int = 0
    repair_completion_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionMetrics:
    condition: str
    model: str
    pass_rate: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    token_efficiency: float
    repair_convergence_rate: Optional[float] = None
    initial_pass_rate: Optional[float] = None
    num_problems: int = 0
    num_passed: int = 0
    error_rate: float = 0.0


def extract_code(text: str) -> str:
    """
    Extract the Python code portion from an LLM response.

    This handles fenced code blocks first, then falls back to finding the
    first Python-looking line and returning the largest parseable prefix.
    """

    def is_python_start(line: str) -> bool:
        return bool(re.match(r"^\s*(def|class)\s+\w+", line)) or bool(
            re.match(r"^\s*(@|from\s+\w+|import\s+\w+)", line)
        )

    def parseable_prefix(candidate_lines: List[str]) -> Optional[str]:
        for end in range(len(candidate_lines), 0, -1):
            candidate = "\n".join(candidate_lines[:end]).strip()
            if not candidate:
                continue
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                continue
        return None

    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)\n?```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    stripped = text.strip()
    if not stripped:
        return stripped

    try:
        ast.parse(stripped)
        return stripped
    except SyntaxError:
        pass

    lines = stripped.splitlines()
    for idx, line in enumerate(lines):
        if is_python_start(line):
            candidate = parseable_prefix(lines[idx:])
            if candidate:
                return candidate
            return "\n".join(lines[idx:]).strip()

    return stripped


base_runner.ExperimentResult = ExperimentResult
base_runner.ConditionMetrics = ConditionMetrics
base_runner.extract_code = extract_code


def resolve_conditions(selected_conditions: Optional[List[str]]) -> List[str]:
    """Validate and normalize an optional user-provided condition subset."""
    if not selected_conditions:
        return list(AblationExperimentRunner.BASE_CONDITIONS)

    invalid = [cond for cond in selected_conditions if cond not in AblationExperimentRunner.BASE_CONDITIONS]
    if invalid:
        valid = ", ".join(AblationExperimentRunner.BASE_CONDITIONS)
        invalid_str = ", ".join(invalid)
        raise ValueError(f"Unknown condition(s): {invalid_str}. Valid options are: {valid}")

    # Preserve order while removing duplicates.
    seen = set()
    resolved = []
    for cond in selected_conditions:
        if cond not in seen:
            resolved.append(cond)
            seen.add(cond)
    return resolved


class AblationExperimentRunner(base_runner.AblationExperimentRunner):
    """Patched runner that leaves the original file untouched."""

    BASE_CONDITIONS = list(base_runner.AblationExperimentRunner.CONDITIONS)

    def __init__(self, *args, conditions: Optional[List[str]] = None, **kwargs):
        self.CONDITIONS = resolve_conditions(conditions)
        super().__init__(*args, **kwargs)

    def _run_condition_for_model(
        self,
        model: str,
        condition: str,
        problems: List[Dict[str, Any]],
    ):
        from augmentation_pipeline_patched import AugmentationPipeline
        from code_executor import CodeExecutor
        from model_interface import ModelInterface

        model_interface = ModelInterface.get_interface(model)
        augmentation = AugmentationPipeline(condition)
        executor = CodeExecutor()

        passed = 0
        total_tokens = 0
        condition_results: List[ExperimentResult] = []

        for i, problem in enumerate(problems):
            try:
                prompt_data = augmentation.generate_prompt(
                    problem=problem,
                    model=model,
                    all_problems=problems,
                )

                start = time.time()
                response = model_interface.generate(prompt=prompt_data["prompt"])
                execution_time = time.time() - start

                generated_code = extract_code(response["text"])
                prompt_tokens = response["prompt_tokens"]
                completion_tokens = response["completion_tokens"]
                attempts = 1
                initial_passed = None
                repair_prompt_tokens = 0
                repair_completion_tokens = 0

                if condition == "C5_iterative_repair":
                    (
                        generated_code,
                        attempts,
                        initial_passed,
                        repair_prompt_tokens,
                        repair_completion_tokens,
                    ) = self._apply_iterative_repair(
                        model_interface,
                        problem,
                        generated_code,
                        max_attempts=self.max_repair_attempts,
                    )

                test_passed = executor.run_tests(
                    code=generated_code,
                    test_cases=problem.get("tests", []),
                    timeout=5,
                )

                if test_passed:
                    passed += 1

                total_prompt_tokens = prompt_tokens + repair_prompt_tokens
                total_completion_tokens = completion_tokens + repair_completion_tokens
                total_tokens += total_prompt_tokens + total_completion_tokens

                condition_results.append(
                    ExperimentResult(
                        model=model,
                        condition=condition,
                        problem_id=problem.get("id", f"problem_{i}"),
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        passed=test_passed,
                        execution_time=execution_time,
                        attempts=attempts,
                        initial_passed=initial_passed,
                        repair_prompt_tokens=repair_prompt_tokens,
                        repair_completion_tokens=repair_completion_tokens,
                    )
                )

            except Exception as e:
                logger.error(f"  Error - problem {problem.get('id', i)}: {e}")
                condition_results.append(
                    ExperimentResult(
                        model=model,
                        condition=condition,
                        problem_id=problem.get("id", f"problem_{i}"),
                        prompt_tokens=0,
                        completion_tokens=0,
                        passed=False,
                        execution_time=0.0,
                        error_message=str(e),
                        attempts=1,
                    )
                )

            if (i + 1) % 5 == 0:
                denom = i + 1
                logger.info(
                    f"  Progress: {denom}/{len(problems)} | "
                    f"Passed: {passed}/{denom} ({100 * passed / denom:.1f}%) | "
                    f"Avg tokens: {total_tokens // denom}"
                )

        final_label = "FinalPass" if condition == "C5_iterative_repair" else "Pass@1"
        pass_rate = 100.0 * passed / len(problems) if problems else 0.0
        logger.info(
            f"  DONE Finished - {final_label}={pass_rate:.1f}% ({passed}/{len(problems)}) | "
            f"Avg tokens/problem={total_tokens // max(len(problems), 1)}"
        )
        self.results.extend(condition_results)

    def _apply_iterative_repair(
        self,
        model_interface,
        problem: Dict[str, Any],
        initial_code: str,
        max_attempts: int = 2,
    ) -> Tuple[str, int, bool, int, int]:
        from code_executor import CodeExecutor

        executor = CodeExecutor()
        code = initial_code
        initial_passed = False
        repair_prompt_tokens = 0
        repair_completion_tokens = 0

        for attempt in range(1, max_attempts + 1):
            test_passed, error_output = executor.run_tests(
                code=code,
                test_cases=problem.get("tests", []),
                return_error=True,
                timeout=5,
            )
            if attempt == 1:
                initial_passed = test_passed
            if test_passed:
                return code, attempt, initial_passed, repair_prompt_tokens, repair_completion_tokens

            if attempt < max_attempts:
                repair_prompt = (
                    f"The following bioinformatics function has a bug.\n\n"
                    f"Error output:\n{error_output}\n\n"
                    f"Task description:\n{problem.get('docstring', '')}\n\n"
                    f"Buggy code:\n```python\n{code}\n```\n\n"
                    f"Diagnose the error, then provide ONLY the corrected function code."
                )
                response = model_interface.generate(prompt=repair_prompt)
                repair_prompt_tokens += response["prompt_tokens"]
                repair_completion_tokens += response["completion_tokens"]
                code = extract_code(response["text"])

        return code, max_attempts, initial_passed, repair_prompt_tokens, repair_completion_tokens

    def _compute_metrics(self):
        logger.info("Computing aggregate metrics ...")

        for model in self.models:
            self.condition_metrics[model] = {}

            for condition in self.CONDITIONS:
                filtered = [
                    r for r in self.results
                    if r.model == model and r.condition == condition
                ]
                if not filtered:
                    continue

                n = len(filtered)
                n_passed = sum(1 for r in filtered if r.passed)
                n_errors = sum(1 for r in filtered if r.error_message)

                pass_rate = 100.0 * n_passed / n
                error_rate = n_errors / n

                avg_prompt = float(np.mean([r.prompt_tokens for r in filtered]))
                avg_completion = float(np.mean([r.completion_tokens for r in filtered]))
                total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in filtered)
                token_efficiency = (n_passed * 1000.0) / total_tokens if total_tokens else 0.0

                repair_convergence = None
                initial_pass_rate = None
                if condition == "C5_iterative_repair":
                    avg_attempts = float(np.mean([r.attempts for r in filtered]))
                    initial_passes = sum(1 for r in filtered if r.initial_passed)
                    initial_pass_rate = 100.0 * initial_passes / n
                    repair_convergence = (
                        (1.0 - (avg_attempts - 1.0) / max(self.max_repair_attempts - 1, 1)) * 100.0
                    )

                self.condition_metrics[model][condition] = ConditionMetrics(
                    condition=condition,
                    model=model,
                    pass_rate=pass_rate,
                    avg_prompt_tokens=avg_prompt,
                    avg_completion_tokens=avg_completion,
                    token_efficiency=token_efficiency,
                    repair_convergence_rate=repair_convergence,
                    initial_pass_rate=initial_pass_rate,
                    num_problems=n,
                    num_passed=n_passed,
                    error_rate=error_rate,
                )

    def _generate_summary_report(self):
        report_path = self.output_dir / "RESULTS_SUMMARY.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION STUDY - RESULTS SUMMARY (PATCHED)\n")
            f.write(f"Generated: {base_runner.datetime.now().isoformat()}\n")
            f.write(f"Models: {', '.join(self.models)}\n")
            f.write(f"Problems per cell: {self.num_problems}\n")
            f.write("=" * 80 + "\n")

            for model in self.models:
                f.write(f"\n{'-' * 72}\n  MODEL: {model}\n{'-' * 72}\n")
                f.write(f"{'Condition':<24}{'Pass':>9}{'InitP@1':>11}{'TokEff':>10}{'RepConv':>12}\n")
                f.write("-" * 72 + "\n")
                for cond in self.CONDITIONS:
                    metric = self.condition_metrics.get(model, {}).get(cond)
                    if not metric:
                        continue
                    init_pass = (
                        f"{metric.initial_pass_rate:.1f}%"
                        if metric.initial_pass_rate is not None else
                        "N/A"
                    )
                    rep_conv = (
                        f"{metric.repair_convergence_rate:.1f}%"
                        if metric.repair_convergence_rate is not None else
                        "N/A"
                    )
                    f.write(
                        f"{cond:<24}{metric.pass_rate:>8.1f}%{init_pass:>11}"
                        f"{metric.token_efficiency:>10.3f}{rep_conv:>12}\n"
                    )

        logger.info(f"Summary report -> {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Patched ablation study runner for bioinformatics code generation"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=AblationExperimentRunner.DEFAULT_MODELS,
        help="Space-separated list of model identifiers",
    )
    parser.add_argument(
        "--biocoder-dir",
        default="./biocoder_dataset",
        help="Path to BioCoder dataset folder (not needed with --mock)",
    )
    parser.add_argument("--output-dir", default="./results_patched")
    parser.add_argument("--num-problems", type=int, default=80)
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help=(
            "Optional subset of conditions to run. "
            "Choices: C1_zero_shot C2_few_shot C3_chain_of_thought "
            "C4_rag_context C5_iterative_repair"
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic bioinformatics problems instead of the BioCoder dataset.",
    )
    args = parser.parse_args()
    selected_conditions = resolve_conditions(args.conditions)

    runner = AblationExperimentRunner(
        models=args.models,
        biocoder_dir=Path(args.biocoder_dir),
        output_dir=Path(args.output_dir),
        num_problems=args.num_problems,
        max_repair_attempts=args.max_repair_attempts,
        seed=args.seed,
        mock=args.mock,
        conditions=selected_conditions,
    )
    runner.run_experiment()


if __name__ == "__main__":
    main()
