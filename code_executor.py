"""
Code Executor: Safely runs generated Python code against test cases
Uses subprocess isolation with timeout enforcement
"""

import subprocess
import sys
import tempfile
import textwrap
import os
import logging
from typing import List, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Executes and tests generated Python functions in an isolated subprocess."""

    def __init__(self, timeout: int = 5):
        self.default_timeout = timeout

    def run_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        timeout: int = None,
        return_error: bool = False,
    ) -> Union[bool, Tuple[bool, str]]:
        """
        Run test cases against generated code.

        Args:
            code: Generated Python function code.
            test_cases: List of {"input": ..., "expected": ...} dicts,
                        or strings that are valid Python assert statements.
            timeout: Per-test timeout in seconds.
            return_error: If True, return (passed, error_output) tuple.

        Returns:
            bool (or (bool, str) if return_error=True)
        """
        timeout = timeout or self.default_timeout

        if not test_cases:
            result = (True, "") if return_error else True
            return result

        # Build test script
        test_script = self._build_test_script(code, test_cases)

        passed, error_output = self._execute_script(test_script, timeout)

        if return_error:
            return passed, error_output
        return passed

    def _build_test_script(
        self, code: str, test_cases: List[Any]
    ) -> str:
        lines = ["import sys", ""]
        # Inject generated code
        lines.append(textwrap.dedent(code))
        lines.append("")

        for i, tc in enumerate(test_cases):
            if isinstance(tc, str):
                # Raw assert statement
                lines.append(f"try:")
                lines.append(f"    {tc}")
                lines.append(f"    print('PASS test_{i}')")
                lines.append(f"except Exception as e:")
                lines.append(f"    print(f'FAIL test_{i}: {{e}}', file=sys.stderr)")
                lines.append(f"    sys.exit(1)")
            elif isinstance(tc, dict):
                fn_call = tc.get("call", "")
                expected = tc.get("expected", None)
                if fn_call:
                    lines.append("try:")
                    lines.append(f"    _result = {fn_call}")
                    if expected is not None:
                        # Use double-quoted outer string so repr() single quotes
                        # don't cause a SyntaxError inside the generated script
                        exp_r = repr(expected)
                        lines.append(
                            '    assert _result == ' + exp_r +
                            ', "Got " + repr(_result) + ", expected ' + exp_r + '"'
                        )
                    lines.append(f"    print('PASS test_{i}')")
                    lines.append("except Exception as e:")
                    lines.append(f"    print('FAIL test_{i}: ' + str(e), file=sys.stderr)")
                    lines.append("    sys.exit(1)")

        return "\n".join(lines)

    def _execute_script(self, script: str, timeout: int) -> Tuple[bool, str]:
        # Use the OS-appropriate temp directory (fixes Windows where /tmp doesn't exist)
        tmp_dir = tempfile.gettempdir()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tmp_dir,
            encoding="utf-8"
        ) as f:
            f.write(script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                return True, ""
            else:
                error_out = result.stderr.strip() or result.stdout.strip()
                return False, error_out
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: execution exceeded {timeout}s"
        except Exception as e:
            return False, f"ExecutorError: {str(e)}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _demo():
    executor = CodeExecutor()
    code = "def gc_content(seq):\n    return (seq.count('G') + seq.count('C')) / len(seq)"
    tests = [
        {"call": "gc_content('ACGT')", "expected": 0.5},
        {"call": "gc_content('GGCC')", "expected": 1.0},
    ]
    passed = executor.run_tests(code, tests)
    print(f"Test passed: {passed}")


if __name__ == "__main__":
    _demo()
