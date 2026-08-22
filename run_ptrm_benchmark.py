r"""
PTRM Benchmark Runner & Ablation Suite
======================================
Reproduces experiments, scaling sweeps, and ablations from:
"Probabilistic Tiny Recursive Model" (arXiv:2605.19943v1)

Supported Experiments:
1. Width Scaling ($K \in [1, 5, 10, 25, 100]$): pass@K vs best-Q@K vs mode@K
2. Depth Scaling ($D \in [16, 32, 48, 64]$): depth vs width scaling trade-offs
3. Noise Scale Ablation ($\sigma \in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]$)
4. ARC-AGI Test-Time Augmentation + PTRM Stochastic Rollouts
5. Fast Synthetic Sudoku / Grid Constraint Reasoning Benchmarking
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ptrm import (
    PTRMConfig,
    PTRMRolloutEngine,
    PTRMRolloutResult,
    ARCPTRMEvaluator,
    PTRMPuzzleEvaluator,
)
from trm_arc import (
    TRM,
    TRMConfig,
    TRMCarry,
    load_arc_tasks,
    augment_arc_puzzle,
    inverse_augment_grid,
    grid_to_seq,
    seq_to_grid,
    SEQ_LEN,
    VOCAB_SIZE,
)


# ============================================================
# SYNTHETIC PUZZLE BENCHMARK GENERATOR (SUDOKU / GRID REASONING)
# ============================================================
def generate_synthetic_sudoku_batch(
    num_samples: int = 50,
    grid_size: int = 9,
    clue_ratio: float = 0.35,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic Sudoku-like constraint puzzles for fast benchmarking.
    Outputs:
        inputs: [num_samples, SEQ_LEN] token sequences (clues with 0 for blanks)
        targets: [num_samples, SEQ_LEN] completed solution sequences
    """
    rng = np.random.RandomState(seed)
    inputs_list = []
    targets_list = []

    # Valid base 9x9 latin square
    base_pattern = np.array([
        [(i * 3 + i // 3 + j) % 9 + 1 for j in range(9)]
        for i in range(9)
    ])

    for s in range(num_samples):
        # Permute numbers 1-9
        perm = rng.permutation(np.arange(1, 10))
        sol_grid = perm[base_pattern - 1]

        # Apply random row/col band swaps
        for b in range(3):
            r_perm = rng.permutation(3) + b * 3
            sol_grid[b * 3:(b + 1) * 3, :] = sol_grid[r_perm, :]
            c_perm = rng.permutation(3) + b * 3
            sol_grid[:, b * 3:(b + 1) * 3] = sol_grid[:, c_perm]

        # Mask clues (set non-clues to 0)
        mask = rng.rand(grid_size, grid_size) < clue_ratio
        inp_grid = np.where(mask, sol_grid, 0)

        # Pad to 30x30 standard ARC / model sequence format
        inp_full = np.zeros((30, 30), dtype=np.int64)
        tgt_full = np.zeros((30, 30), dtype=np.int64)

        # Use color offset = 2
        inp_full[:grid_size, :grid_size] = np.where(inp_grid > 0, inp_grid + 2, 0)
        tgt_full[:grid_size, :grid_size] = tgt_full[:grid_size, :grid_size] + (sol_grid + 2)

        inputs_list.append(torch.tensor(inp_full.flatten(), dtype=torch.long))
        targets_list.append(torch.tensor(tgt_full.flatten(), dtype=torch.long))

    return torch.stack(inputs_list, dim=0), torch.stack(targets_list, dim=0)


# ============================================================
# BENCHMARK SUITES
# ============================================================
def run_width_scaling_benchmark(
    model: TRM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    k_list: List[int] = [1, 5, 10, 25, 50, 100],
    sigma: float = 0.2,
    depth: int = 16,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """
    Evaluate width scaling: pass@K, best-Q@K, and mode@K as K varies (Paper Figure 6).
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 1: Width Scaling (K sweep: {k_list}, sigma={sigma}, D={depth})")
    print("=" * 80)

    results = {
        "K": k_list,
        "pass@K": [],
        "best-Q@K": [],
        "mode@K": [],
        "time_per_sample_ms": [],
    }

    B = inputs.shape[0]

    for K in k_list:
        cfg = PTRMConfig(
            num_rollouts=K,
            supervision_steps=depth,
            noise_scale=0.0 if K == 1 else sigma,
            selector="all",
        )
        engine = PTRMRolloutEngine(cfg)

        t0 = time.perf_counter()
        rollout_res = engine.run_rollouts(
            model=model,
            inputs=inputs.to(device),
            config=cfg,
        )
        dt = time.perf_counter() - t0

        pass_k, _ = rollout_res.compute_pass_at_k(targets.to(device), ignore_index=0)
        best_q, _ = rollout_res.compute_best_q_accuracy(targets.to(device), ignore_index=0)
        mode_k, _ = rollout_res.compute_mode_accuracy(targets.to(device), ignore_index=0)
        ms_per_sample = (dt / B) * 1000

        results["pass@K"].append(pass_k * 100)
        results["best-Q@K"].append(best_q * 100)
        results["mode@K"].append(mode_k * 100)
        results["time_per_sample_ms"].append(ms_per_sample)

        print(
            f"  K = {K:3d} | "
            f"pass@K: {pass_k*100:6.2f}% | "
            f"best-Q@K: {best_q*100:6.2f}% | "
            f"mode@K: {mode_k*100:6.2f}% | "
            f"Latency: {ms_per_sample:6.1f} ms/sample"
        )

    return results


def run_noise_ablation_benchmark(
    model: TRM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    sigma_list: List[float] = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    K: int = 25,
    depth: int = 16,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """
    Evaluate noise scale ablation: pass@K, best-Q@K, mode@K as sigma varies (Paper Figure 7).
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2: Noise Scale Ablation (sigma sweep: {sigma_list}, K={K}, D={depth})")
    print("=" * 80)

    results = {
        "sigma": sigma_list,
        "pass@K": [],
        "best-Q@K": [],
        "mode@K": [],
    }

    for sigma in sigma_list:
        cfg = PTRMConfig(
            num_rollouts=K,
            supervision_steps=depth,
            noise_scale=sigma,
            selector="all",
        )
        engine = PTRMRolloutEngine(cfg)

        rollout_res = engine.run_rollouts(
            model=model,
            inputs=inputs.to(device),
            config=cfg,
        )

        pass_k, _ = rollout_res.compute_pass_at_k(targets.to(device), ignore_index=0)
        best_q, _ = rollout_res.compute_best_q_accuracy(targets.to(device), ignore_index=0)
        mode_k, _ = rollout_res.compute_mode_accuracy(targets.to(device), ignore_index=0)

        results["pass@K"].append(pass_k * 100)
        results["best-Q@K"].append(best_q * 100)
        results["mode@K"].append(mode_k * 100)

        print(
            f"  sigma = {sigma:4.2f} | "
            f"pass@K: {pass_k*100:6.2f}% | "
            f"best-Q@K: {best_q*100:6.2f}% | "
            f"mode@K: {mode_k*100:6.2f}%"
        )

    return results


def run_depth_scaling_benchmark(
    model: TRM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    depth_list: List[int] = [16, 32, 48, 64],
    K: int = 25,
    sigma: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """
    Evaluate depth scaling: pass@K and best-Q@K as recursion depth D varies (Paper Table 1).
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 3: Depth Scaling (D sweep: {depth_list}, K={K}, sigma={sigma})")
    print("=" * 80)

    results = {
        "depth": depth_list,
        "pass@K": [],
        "best-Q@K": [],
        "mode@K": [],
    }

    for D in depth_list:
        cfg = PTRMConfig(
            num_rollouts=K,
            supervision_steps=D,
            noise_scale=sigma,
            selector="all",
        )
        engine = PTRMRolloutEngine(cfg)

        rollout_res = engine.run_rollouts(
            model=model,
            inputs=inputs.to(device),
            config=cfg,
        )

        pass_k, _ = rollout_res.compute_pass_at_k(targets.to(device), ignore_index=0)
        best_q, _ = rollout_res.compute_best_q_accuracy(targets.to(device), ignore_index=0)
        mode_k, _ = rollout_res.compute_mode_accuracy(targets.to(device), ignore_index=0)

        results["pass@K"].append(pass_k * 100)
        results["best-Q@K"].append(best_q * 100)
        results["mode@K"].append(mode_k * 100)

        print(
            f"  D = {D:2d} | "
            f"pass@K: {pass_k*100:6.2f}% | "
            f"best-Q@K: {best_q*100:6.2f}% | "
            f"mode@K: {mode_k*100:6.2f}%"
        )

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Probabilistic TRM Benchmark Runner (arXiv:2605.19943v1)")

    parser.add_argument("--suite", choices=["synthetic", "arc", "all"], default="synthetic", help="Benchmark suite")
    parser.add_argument("--data", default="arc_data", help="Path to ARC data directory")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--arch", choices=["trm-att", "trm-mlp"], default="trm-att", help="Architecture variant")
    parser.add_argument("--k-sweep", default="1,5,10,25", help="Comma-separated list of rollouts K")
    parser.add_argument("--sigma-sweep", default="0.0,0.2,0.6", help="Comma-separated list of noise scales sigma")
    parser.add_argument("--depth-sweep", default="4,8,16", help="Comma-separated list of depths D")
    parser.add_argument("--eval-depth", type=int, default=4, help="Supervision depth D for width and noise sweeps")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of benchmark samples")
    parser.add_argument("--num-aug", type=int, default=10, help="Number of ARC augmentations")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden dimension (default: 64 for fast CPU run)")
    parser.add_argument("--n-latent", type=int, default=2, help="Latent reasoning cycles L_cycles")
    parser.add_argument("--n-deep", type=int, default=2, help="Deep recursion cycles H_cycles")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-json", default="ptrm_benchmark_report.json", help="Path to save output report")

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 80, flush=True)
    print("Probabilistic Tiny Recursive Model (PTRM) - Benchmark Suite", flush=True)
    print("Paper: arXiv:2605.19943v1", flush=True)
    print(f"Device: {device} | Architecture: {args.arch} | D={args.eval_depth}", flush=True)
    print("=" * 80, flush=True)

    # Initialize model
    config = TRMConfig(
        hidden_size=args.hidden_size if args.checkpoint is None else 512,
        num_heads=4 if args.checkpoint is None else 8,
        num_layers=1 if args.checkpoint is None else 2,
        n_latent_cycles=args.n_latent,
        n_deep_cycles=args.n_deep,
        halt_max_steps=16,
        forward_dtype="float32",
        mlp_t=(args.arch == "trm-mlp"),
    )

    model = TRM(config, num_puzzle_ids=1)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

    model = model.to(device)
    model.eval()

    k_list = [int(x.strip()) for x in args.k_sweep.split(",")]
    sigma_list = [float(x.strip()) for x in args.sigma_sweep.split(",")]
    depth_list = [int(x.strip()) for x in args.depth_sweep.split(",")]

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": config.__dict__}

    if args.suite in ("synthetic", "all"):
        print("\nGenerating synthetic reasoning puzzle dataset...", flush=True)
        inputs, targets = generate_synthetic_sudoku_batch(
            num_samples=args.num_samples,
            seed=args.seed,
        )

        w_results = run_width_scaling_benchmark(
            model=model,
            inputs=inputs,
            targets=targets,
            k_list=k_list,
            sigma=0.2,
            depth=args.eval_depth,
            device=device,
        )
        report["width_scaling"] = w_results

        n_results = run_noise_ablation_benchmark(
            model=model,
            inputs=inputs,
            targets=targets,
            sigma_list=sigma_list,
            K=max(k_list),
            depth=args.eval_depth,
            device=device,
        )
        report["noise_ablation"] = n_results

        d_results = run_depth_scaling_benchmark(
            model=model,
            inputs=inputs,
            targets=targets,
            depth_list=depth_list,
            K=max(k_list),
            sigma=0.2,
            device=device,
        )
        report["depth_scaling"] = d_results

    if args.suite in ("arc", "all") and os.path.exists(args.data):
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: ARC-AGI Benchmark with Test-Time Augmentation + PTRM")
        print("=" * 80)
        eval_tasks = load_arc_tasks(args.data, "evaluation")
        evaluator = ARCPTRMEvaluator(
            ptrm_config=PTRMConfig(num_rollouts=min(k_list[-1], 25), noise_scale=0.2, supervision_steps=16),
            num_aug=args.num_aug,
        )

        top1_correct = 0
        top2_correct = 0
        total = min(len(eval_tasks), args.num_samples)

        for i in range(total):
            task = eval_tasks[i]
            if len(task.test_examples) == 0:
                continue
            inp, out = task.test_examples[0]
            if out.shape == (1, 1) and out[0, 0] == 0:
                continue

            res = evaluator.evaluate_task_example(
                model=model,
                train_examples=task.train_examples,
                test_inp=inp,
                test_out=out,
                augment_fn=augment_arc_puzzle,
                inverse_augment_fn=inverse_augment_grid,
                grid_to_seq_fn=grid_to_seq,
                seq_to_grid_fn=seq_to_grid,
                device=device,
            )
            if res["is_correct_top1"]:
                top1_correct += 1
            if res["is_correct_top2"]:
                top2_correct += 1

            if (i + 1) % max(1, total // 5) == 0:
                print(f"  [{i+1}/{total}] Top-1: {top1_correct}/{i+1} ({top1_correct/(i+1)*100:.1f}%), Top-2: {top2_correct}/{i+1} ({top2_correct/(i+1)*100:.1f}%)")

        report["arc_evaluation"] = {
            "evaluated_tasks": total,
            "top1_accuracy": top1_correct / max(1, total),
            "top2_accuracy": top2_correct / max(1, total),
        }

    # Save report
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark finished successfully! Saved report to: {args.output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
