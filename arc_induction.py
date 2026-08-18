"""
MATHX ARC-AGI INDUCTIVE REASONING ENGINE
Implements Program Induction (Program Synthesis + Sandbox Verification)
from 'Combining Induction and Transduction for Abstract Reasoning' (Cornell).
"""

from __future__ import annotations
import json
import time
import traceback
import numpy as np
from typing import Callable, Optional, Any
from dataclasses import dataclass

Grid = np.ndarray

def grid_to_str(g: Grid) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in g)

def parse_grid_str(s: str) -> Grid:
    lines = [line.strip().split() for line in s.strip().split("\n") if line.strip()]
    return np.array([[int(c) for c in row] for row in lines], dtype=np.int32)

@dataclass
class InductiveCandidate:
    code: str
    fn: Callable[[Grid], Grid]
    train_score: float  # 1.0 = exact match on all training pairs
    train_exact: bool
    execution_time: float


class InductiveSandbox:
    """Safe execution sandbox for synthesized Python transformation programs."""
    
    @staticmethod
    def execute_code(code_str: str, function_name: str = "transform") -> Optional[Callable[[Grid], Grid]]:
        local_scope: dict[str, Any] = {"np": np, "numpy": np}
        try:
            # Clean markdown codeblocks if present
            cleaned = code_str
            if "```python" in cleaned:
                cleaned = cleaned.split("```python")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]

            exec(cleaned, local_scope, local_scope)
            fn = local_scope.get(function_name)
            if callable(fn):
                return fn
        except Exception:
            pass
        return None

    @classmethod
    def verify_program(cls, fn: Callable[[Grid], Grid], train_pairs: list[tuple[Grid, Grid]]) -> tuple[bool, float]:
        """Verify if synthesized function exactly solves all training examples."""
        if not train_pairs:
            return False, 0.0

        correct = 0
        for inp, expected in train_pairs:
            try:
                pred = fn(inp.copy())
                if pred is not None and pred.shape == expected.shape and np.array_equal(pred, expected):
                    correct += 1
            except Exception:
                return False, 0.0

        score = correct / len(train_pairs)
        return (score == 1.0), score


class InductivePromptGenerator:
    """Formats ARC tasks into few-shot program induction prompts."""

    SYSTEM_PROMPT = """You are an expert AI programmer solving the Abstraction and Reasoning Corpus (ARC-AGI).
Your task is to write a clean, exact Python function `transform(input_grid: np.ndarray) -> np.ndarray` that reproduces the transformation demonstrated in the training input-output pairs.

Rules:
1. The function must accept a 2D numpy array `input_grid` of integers (0-9) representing color values.
2. Return a 2D numpy array representing the output grid.
3. The logic must be completely general so that it generalizes to unseen test inputs.
4. Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=teal, 9=maroon.
5. Use standard numpy operations. Keep the code self-contained inside `def transform(input_grid: np.ndarray) -> np.ndarray:`.
"""

    @classmethod
    def format_task_prompt(cls, task: dict) -> str:
        prompt_lines = ["Given the following training examples:"]
        for idx, ex in enumerate(task["train"]):
            inp = np.array(ex["input"], dtype=np.int32)
            out = np.array(ex["output"], dtype=np.int32)
            prompt_lines.append(f"\n--- Example {idx + 1} ---")
            prompt_lines.append(f"Input ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_str(inp)}")
            prompt_lines.append(f"Output ({out.shape[0]}x{out.shape[1]}):\n{grid_to_str(out)}")

        if "test" in task and task["test"]:
            test_inp = np.array(task["test"][0]["input"], dtype=np.int32)
            prompt_lines.append(f"\n--- Test Input ({test_inp.shape[0]}x{test_inp.shape[1]}) ---")
            prompt_lines.append(f"{grid_to_str(test_inp)}")

        prompt_lines.append("\nWrite the complete Python function `def transform(input_grid: np.ndarray) -> np.ndarray:` to solve this task.")
        return "\n".join(prompt_lines)
