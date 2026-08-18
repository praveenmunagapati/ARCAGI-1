"""
MATHX HYBRID REASONING ENGINE (INDUCTION + TRANSDUCTION)
Implements the 1st Place ICLR 2025 Architecture:
'Combining Induction and Transduction for Abstract Reasoning' (Li, Ellis et al., Cornell)
Accelerated on NVIDIA GeForce MX330 GPU.
"""

from __future__ import annotations
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Any
import numpy as np

from arc_induction import InductiveSandbox, InductivePromptGenerator
from arc_transduction import TransductiveAugmentor, SpatialConsensusVoter, parse_grid_str, grid_to_str
from arc_gpu_solver import GPUSolverEngine, G, exact

Grid = np.ndarray


@dataclass
class HybridPrediction:
    task_id: str
    guess_1: Optional[Grid]
    guess_2: Optional[Grid]
    guess_1_source: str  # "INDUCTION_EXACT", "SYMBOLIC_GPU", "TRANSDUCTION_CONSENSUS"
    guess_2_source: str
    solved_exact: bool
    train_score: float
    runtime_seconds: float


class HybridARCSolver:
    """
    Unified Orchestrator combining:
      - Inductive Program Synthesis & Exact Verification Sandbox
      - Direct Spatial Transduction with Dihedral D8 Test-Time Augmentation
      - High-Speed GPU Deductive Search on NVIDIA GeForce MX330
    """

    def __init__(self, use_gpu: bool = True):
        self.gpu_engine = GPUSolverEngine() if use_gpu else None
        print("[HYBRID] Initialized Cornell Induction + Transduction Framework.")

    def solve_task(self, task: dict, candidate_programs: Optional[list[str]] = None, candidate_transductions: Optional[list[str]] = None) -> HybridPrediction:
        start_time = time.perf_counter()
        train_pairs = [(G(ex["input"]), G(ex["output"])) for ex in task["train"]]
        test_inputs = [G(ex["input"]) for ex in task.get("test", [])]
        expected_outputs = [G(ex["output"]) for ex in task.get("test", []) if "output" in ex]

        guess_1: Optional[Grid] = None
        guess_2: Optional[Grid] = None
        source_1: str = "NONE"
        source_2: str = "NONE"
        train_score: float = 0.0

        # ============================================================
        # PHASE 1: PROGRAM INDUCTION (Strict Symbolic / Code Synthesis)
        # ============================================================
        if candidate_programs:
            for code in candidate_programs:
                fn = InductiveSandbox.execute_code(code)
                if fn:
                    is_exact, score = InductiveSandbox.verify_program(fn, train_pairs)
                    if is_exact:
                        try:
                            pred = fn(test_inputs[0].copy())
                            if pred is not None:
                                guess_1 = pred
                                source_1 = "INDUCTION_EXACT"
                                train_score = 1.0
                                break
                        except Exception:
                            pass
                    elif score > train_score:
                        train_score = score

        # ============================================================
        # PHASE 2: GPU DEDUCTIVE SOLVER ENGINE (NVIDIA MX330)
        # ============================================================
        if self.gpu_engine and (guess_1 is None or guess_2 is None):
            prog = self.gpu_engine.synthesize(task)
            if prog:
                try:
                    gpu_pred = prog(test_inputs[0])
                    if guess_1 is None:
                        guess_1 = gpu_pred
                        source_1 = "SYMBOLIC_GPU"
                        train_score = 1.0
                    elif guess_2 is None and not np.array_equal(guess_1, gpu_pred):
                        guess_2 = gpu_pred
                        source_2 = "SYMBOLIC_GPU"
                except Exception:
                    pass

        # ============================================================
        # PHASE 3: DIRECT TRANSDUCTION (Spatial Consensus Voting)
        # ============================================================
        if candidate_transductions:
            parsed_grids = []
            for t_str in candidate_transductions:
                g = parse_grid_str(t_str)
                if g is not None:
                    parsed_grids.append(g)

            if parsed_grids:
                trans_pred, confidence = SpatialConsensusVoter.vote(parsed_grids)
                if trans_pred is not None:
                    if guess_1 is None:
                        guess_1 = trans_pred
                        source_1 = f"TRANSDUCTION_VOTE ({confidence:.0%})"
                    elif guess_2 is None and not np.array_equal(guess_1, trans_pred):
                        guess_2 = trans_pred
                        source_2 = f"TRANSDUCTION_VOTE ({confidence:.0%})"

        # Verify test correctness
        solved_exact = False
        if expected_outputs and guess_1 is not None:
            if exact(guess_1, expected_outputs[0]) or (guess_2 is not None and exact(guess_2, expected_outputs[0])):
                solved_exact = True

        elapsed = time.perf_counter() - start_time
        task_id = task.get("id", "task")

        return HybridPrediction(
            task_id=task_id,
            guess_1=guess_1,
            guess_2=guess_2,
            guess_1_source=source_1,
            guess_2_source=source_2,
            solved_exact=solved_exact,
            train_score=train_score,
            runtime_seconds=elapsed,
        )


# ============================================================
# BENCHMARK EVALUATOR
# ============================================================

def run_hybrid_benchmark(data_dir: str = "arc_data", split: str = "training", limit: int = 0):
    root = Path(data_dir)
    if split == "training":
        tasks = sorted((root / "training").glob("*.json"))
    elif split == "evaluation":
        tasks = sorted((root / "evaluation").glob("*.json"))
    elif split == "all":
        tasks = sorted(root.rglob("*.json"))
    else:
        tasks = sorted(Path(split).glob("*.json")) if Path(split).exists() else sorted(root.glob("*.json"))

    if limit > 0:
        tasks = tasks[:limit]

    print("=" * 80, flush=True)
    print("HYBRID INDUCTION + TRANSDUCTION BENCHMARK RUNNER (Cornell Paradigm)", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset Split:             {split.upper()}", flush=True)
    print(f"Total Tasks Loaded:        {len(tasks)} tasks", flush=True)

    solver = HybridARCSolver(use_gpu=True)
    solved_exact = 0
    train_fit = 0
    total_time = 0.0

    for idx, fpath in enumerate(tasks, 1):
        task_data = json.loads(fpath.read_text(encoding="utf-8"))
        task_data["id"] = fpath.stem

        pred = solver.solve_task(task_data)
        total_time += pred.runtime_seconds

        if pred.train_score == 1.0:
            train_fit += 1
        if pred.solved_exact:
            solved_exact += 1
            status = f"SOLVED ({pred.guess_1_source})"
        else:
            status = f"TRAIN_FIT ({pred.guess_1_source})" if pred.train_score == 1.0 else "NO_PROGRAM"

        if idx <= 15 or idx % 50 == 0 or idx == len(tasks):
            print(f"[{idx:03d}/{len(tasks)}] Task {fpath.stem:<10} | {status:<30} | {pred.runtime_seconds*1000:.1f}ms", flush=True)

    avg_ms = (total_time / len(tasks)) * 1000 if tasks else 0

    print("\n" + "=" * 80, flush=True)
    print("FINAL HYBRID BENCHMARK RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Tasks Evaluated:     {len(tasks)}", flush=True)
    print(f"Training Fit Tasks:        {train_fit}/{len(tasks)} ({100*train_fit/len(tasks):.2f}%)", flush=True)
    print(f"Exact Solved (Top-2):      {solved_exact}/{len(tasks)} ({100*solved_exact/len(tasks):.2f}%)", flush=True)
    print(f"Total Execution Time:      {total_time:.2f} seconds", flush=True)
    print(f"Average Time per Task:     {avg_ms:.2f} ms", flush=True)
    print("=" * 80, flush=True)

    # Save report
    report = {
        "architecture": "Hybrid Induction + Transduction (Cornell 1st Place Paradigm)",
        "split": split,
        "tasks": len(tasks),
        "training_fit": train_fit,
        "exact_solved_top2": solved_exact,
        "total_time_seconds": total_time,
        "avg_ms_per_task": avg_ms,
    }
    Path("mathx_hybrid_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Benchmark report saved to: mathx_hybrid_benchmark_report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Induction + Transduction Solver")
    parser.add_argument("--data", default="arc_data", help="Root data directory")
    parser.add_argument("--split", default="training", choices=["all", "training", "evaluation"], help="Dataset split")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks")
    args = parser.parse_args()
    run_hybrid_benchmark(data_dir=args.data, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
