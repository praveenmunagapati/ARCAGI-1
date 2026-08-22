"""
Probabilistic Tiny Recursive Model (PTRM)
=========================================
Implementation of "Probabilistic Tiny Recursive Model"
(arXiv:2605.19943v1, Amin Sghaier, Ali Parviz, Alexia Jolicoeur-Martineau)

A test-time compute scaling framework for Tiny Recursive Models (TRM).
PTRM injects Gaussian noise into the reasoning latent state at each deep
recursion step, running K parallel rollouts to explore diverse latent basins,
and uses the model's trained Q-head to select the optimal solution.
"""

from __future__ import annotations
import copy
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PTRM CONFIGURATION
# ============================================================
@dataclass
class PTRMConfig:
    """
    Configuration parameters for Probabilistic TRM inference.

    Parameters:
        num_rollouts (K): Number of parallel stochastic rollouts per puzzle (default: 25).
        supervision_steps (D): Number of deep recursion / supervision steps (depth scaling, default: 16).
        noise_scale (sigma): Standard deviation of Gaussian noise injected at each step (default: 0.2).
        selector: Selection strategy ('best_q', 'mode', 'both', 'pass_at_k', 'all').
        noise_type: Noise injection strategy ('gaussian' or 'langevin').
        langevin_lr (eta): Step size for Langevin dynamics guidance (Appendix C).
        langevin_steps (N): Number of Langevin steps per recursion step (default: 1).
        use_amp: Whether to use automatic mixed precision during rollout.
        forward_dtype: Computation dtype string ('float32', 'bfloat16', 'float16').
        seed: Optional random seed for reproducible rollouts.
    """
    num_rollouts: int = 25              # K in paper (1, 5, 10, 25, 100)
    supervision_steps: int = 16         # D in paper (16, 48, 64)
    noise_scale: float = 0.2           # sigma in paper (0.0 to 1.0)
    selector: str = "best_q"           # 'best_q', 'mode', 'both', 'pass_at_k', 'all'
    noise_type: str = "gaussian"       # 'gaussian' or 'langevin'
    langevin_lr: float = 0.01          # eta for Langevin guidance
    langevin_steps: int = 1            # N steps per recursion
    use_amp: bool = True
    forward_dtype: str = "float32"
    seed: Optional[int] = None

    # Benchmark presets from paper
    @classmethod
    def for_sudoku_extreme(cls) -> "PTRMConfig":
        """Configuration for Sudoku-Extreme from paper: K=100, D=64, sigma=0.3."""
        return cls(num_rollouts=100, supervision_steps=64, noise_scale=0.3, selector="best_q")

    @classmethod
    def for_maze_hard(cls) -> "PTRMConfig":
        """Configuration for Maze-Hard from paper: K=100, D=16, sigma=1.0."""
        return cls(num_rollouts=100, supervision_steps=16, noise_scale=1.0, selector="mode")

    @classmethod
    def for_arc_agi(cls) -> "PTRMConfig":
        """Configuration for ARC-AGI from paper: K=25, D=16, sigma=0.2."""
        return cls(num_rollouts=25, supervision_steps=16, noise_scale=0.2, selector="best_q")

    @classmethod
    def for_ppbench(cls) -> "PTRMConfig":
        """Configuration for PPBench from paper: K=100, D=48, sigma=0.2."""
        return cls(num_rollouts=100, supervision_steps=48, noise_scale=0.2, selector="best_q")


# ============================================================
# PTRM ROLLOUT RESULT
# ============================================================
@dataclass
class PTRMRolloutResult:
    """
    Results from executing K parallel stochastic rollouts.
    """
    best_q_pred: torch.Tensor
    best_q_score: torch.Tensor
    best_q_indices: torch.Tensor
    mode_pred: Optional[torch.Tensor] = None
    all_preds: Optional[torch.Tensor] = None
    all_q_scores: Optional[torch.Tensor] = None
    all_logits: Optional[torch.Tensor] = None
    elapsed_time: float = 0.0

    def compute_pass_at_k(self, targets: torch.Tensor, ignore_index: int = 0) -> Tuple[float, torch.Tensor]:
        """
        Compute Pass@K: True if ANY rollout exactly matches the target.

        Args:
            targets: Ground truth tensor [B, SEQ_LEN]
            ignore_index: Token ID to ignore when checking exact match

        Returns:
            pass_rate: Scalar pass@K fraction across batch
            is_pass: Boolean tensor [B]
        """
        assert self.all_preds is not None, "all_preds required for pass@k computation"
        B, K, L = self.all_preds.shape
        targets_exp = targets.unsqueeze(1).expand(B, K, L)

        if ignore_index is not None:
            valid_mask = (targets_exp != ignore_index)
            matches = (self.all_preds == targets_exp) | ~valid_mask
            all_valid_match = matches.all(dim=-1)  # [B, K]
        else:
            all_valid_match = (self.all_preds == targets_exp).all(dim=-1)  # [B, K]

        is_pass = all_valid_match.any(dim=-1)  # [B]
        return is_pass.float().mean().item(), is_pass

    def compute_best_q_accuracy(self, targets: torch.Tensor, ignore_index: int = 0) -> Tuple[float, torch.Tensor]:
        """
        Compute best-Q@K exact accuracy: True if the best-Q rollout exactly matches target.
        """
        if ignore_index is not None:
            valid_mask = (targets != ignore_index)
            matches = (self.best_q_pred == targets) | ~valid_mask
            is_correct = matches.all(dim=-1)
        else:
            is_correct = (self.best_q_pred == targets).all(dim=-1)
        return is_correct.float().mean().item(), is_correct

    def compute_mode_accuracy(self, targets: torch.Tensor, ignore_index: int = 0) -> Tuple[float, torch.Tensor]:
        """
        Compute mode@K exact accuracy: True if the majority vote rollout matches target.
        """
        assert self.mode_pred is not None, "mode_pred required for mode accuracy computation"
        if ignore_index is not None:
            valid_mask = (targets != ignore_index)
            matches = (self.mode_pred == targets) | ~valid_mask
            is_correct = matches.all(dim=-1)
        else:
            is_correct = (self.mode_pred == targets).all(dim=-1)
        return is_correct.float().mean().item(), is_correct


# ============================================================
# PTRM ROLLOUT ENGINE
# ============================================================
class PTRMRolloutEngine:
    """
    High-performance batched rollout engine for Probabilistic TRM.

    Executes K parallel stochastic rollouts per input by vectorizing
    across the batch dimension: effective_batch = B * K.
    """

    def __init__(self, config: Optional[PTRMConfig] = None):
        self.config = config or PTRMConfig()

    @torch.no_grad()
    def run_rollouts(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        config: Optional[PTRMConfig] = None,
    ) -> PTRMRolloutResult:
        """
        Execute K parallel stochastic rollouts on a TRM model.

        Algorithm (Paper Section 4 & Figure 4):
        1. Expand inputs from [B, L] to [B * K, L]
        2. Initialize latents z_0, y_0
        3. For t = 1, ..., D:
             z_(t-1) += eps, where eps ~ N(0, sigma^2 I)
             z_t, y_t <- rec(x, z_(t-1), y_(t-1))
        4. y_hat^(k) = argmax f_O(y_D^(k))
        5. q_hat^(k) = f_Q(y_D^(k))
        6. Select best-Q, mode, and pass candidates

        Args:
            model: TRM model (must have .inner or compatible recursive step interface)
            inputs: Token sequence inputs [B, SEQ_LEN]
            puzzle_ids: Optional puzzle identifiers [B]
            config: Optional override of PTRMConfig

        Returns:
            PTRMRolloutResult containing best-Q, mode, all preds, and Q scores.
        """
        cfg = config or self.config
        t_start = time.perf_counter()

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        model.eval()
        device = inputs.device
        B, seq_len = inputs.shape[:2]
        K = cfg.num_rollouts
        D = cfg.supervision_steps
        sigma = cfg.noise_scale

        if puzzle_ids is None:
            puzzle_ids = torch.zeros(B, dtype=torch.long, device=device)

        # Vectorize: expand inputs and puzzle_ids to [B * K, ...]
        inputs_expanded = inputs.repeat_interleave(K, dim=0)          # [B * K, SEQ_LEN]
        puzzle_ids_expanded = puzzle_ids.repeat_interleave(K, dim=0)  # [B * K]
        total_batch = B * K

        # Get access to inner recursive module
        inner_model = getattr(model, "inner", model)

        # Initialize carry states
        carry = inner_model.empty_carry(total_batch)
        halted = torch.ones(total_batch, dtype=torch.bool, device=device)
        carry = inner_model.reset_carry(carry, halted)

        # Precision context
        fwd_dtype = getattr(torch, cfg.forward_dtype, torch.float32)
        use_amp = cfg.use_amp and device.type == "cuda" and fwd_dtype != torch.float32

        # Supervision recursion loop with Gaussian noise injection
        for step in range(D):
            # Recurrent stochastic perturbation: z_(t-1) += eps, eps ~ N(0, sigma^2 I)
            if sigma > 0.0:
                noise = torch.randn_like(carry.z) * sigma
                carry.z.add_(noise)

            # Deep recursion step forward
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=fwd_dtype if use_amp else torch.float32):
                carry, logits, q_halt = inner_model(carry, inputs_expanded, puzzle_ids_expanded)

        # Extract predictions and Q-scores
        preds_flat = torch.argmax(logits, dim=-1)  # [B * K, SEQ_LEN]

        # Reshape to [B, K, ...]
        all_preds = preds_flat.view(B, K, seq_len)  # [B, K, SEQ_LEN]
        all_q_scores = q_halt.view(B, K).float()    # [B, K]

        # --- 1. Selection Strategy: best-Q@K ---
        # Select the rollout with the highest Q logit: k* = argmax_k q_hat^(k)
        best_q_indices = torch.argmax(all_q_scores, dim=-1)  # [B]
        batch_indices = torch.arange(B, device=device)

        best_q_pred = all_preds[batch_indices, best_q_indices]        # [B, SEQ_LEN]
        best_q_score = all_q_scores[batch_indices, best_q_indices]    # [B]

        # --- 2. Selection Strategy: mode@K (Majority Voting) ---
        mode_pred = None
        if cfg.selector in ("mode", "both", "all"):
            mode_preds_list = []
            all_preds_cpu = all_preds.cpu().numpy()
            for b in range(B):
                rollout_hashes = [all_preds_cpu[b, k].tobytes() for k in range(K)]
                counter = Counter(rollout_hashes)
                most_common_bytes = counter.most_common(1)[0][0]
                first_k = rollout_hashes.index(most_common_bytes)
                mode_preds_list.append(all_preds[b, first_k])
            mode_pred = torch.stack(mode_preds_list, dim=0)  # [B, SEQ_LEN]

        all_logits_reshaped = None
        if cfg.selector == "all":
            all_logits_reshaped = logits.view(B, K, seq_len, -1)

        dt = time.perf_counter() - t_start

        return PTRMRolloutResult(
            best_q_pred=best_q_pred,
            best_q_score=best_q_score,
            best_q_indices=best_q_indices,
            mode_pred=mode_pred,
            all_preds=all_preds,
            all_q_scores=all_q_scores,
            all_logits=all_logits_reshaped,
            elapsed_time=dt,
        )


# ============================================================
# LANGEVIN DYNAMICS SAMPLING (APPENDIX C ABLATION)
# ============================================================
class LangevinGuidanceSampler:
    """
    Q-guided Langevin Dynamics exploration module (Paper Appendix C).

    Samples from p(z) ~ exp(-E(z)) where E(z) = -log sigmoid(f_Q(z))
    using the update rule:
        z <- z - eta * grad_z E(z) + sqrt(2 * eta) * xi,  xi ~ N(0, I)
    """

    def __init__(self, eta: float = 0.01, num_steps: int = 1):
        self.eta = eta
        self.num_steps = num_steps

    def step(self, model_inner: nn.Module, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply Langevin dynamics step to latent state z.
        """
        cur_z = z.clone()
        for _ in range(self.num_steps):
            cur_z.requires_grad_(True)
            if hasattr(model_inner, "q_head"):
                q_val = model_inner.q_head(cur_z[:, 0])[..., 0]
                energy = -F.logsigmoid(q_val).sum()
                grad_z = torch.autograd.grad(energy, cur_z, retain_graph=False)[0]
            else:
                grad_z = torch.zeros_like(cur_z)

            with torch.no_grad():
                noise = torch.randn_like(cur_z) * math.sqrt(2.0 * self.eta)
                cur_z = cur_z - self.eta * grad_z + noise
                cur_z = cur_z.detach()

        return cur_z


# ============================================================
# ARC-AGI PTRM EVALUATION PIPELINE
# ============================================================
class ARCPTRMEvaluator:
    """
    Full ARC-AGI test-time augmentation + PTRM stochastic rollout pipeline.

    For each test puzzle example:
    1. Generates `num_aug` data augmentations (dihedral + color permutations).
    2. Runs `K` parallel PTRM stochastic rollouts per augmentation.
    3. Selects the best rollout per augmentation using Q-head scoring (`best-Q@K`).
    4. Inversely transforms predictions back to canonical space.
    5. Aggregates across augmentations using frequency / Q-weighted voting.
    6. Computes Pass@1, Pass@2, and Pass@K metrics.
    """

    def __init__(
        self,
        ptrm_config: Optional[PTRMConfig] = None,
        num_aug: int = 25,
        submission_k: int = 2,
    ):
        self.config = ptrm_config or PTRMConfig.for_arc_agi()
        self.num_aug = num_aug
        self.submission_k = submission_k
        self.engine = PTRMRolloutEngine(self.config)

    @torch.no_grad()
    def evaluate_task_example(
        self,
        model: nn.Module,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_inp: np.ndarray,
        test_out: np.ndarray,
        augment_fn: Callable,
        inverse_augment_fn: Callable,
        grid_to_seq_fn: Callable,
        seq_to_grid_fn: Callable,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Evaluate a single ARC test example with PTRM.
        """
        model.eval()
        K = self.config.num_rollouts
        candidates_map: Dict[bytes, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "sum_q": 0.0, "max_q": -float("inf"), "grid": None})

        for aug_idx in range(self.num_aug):
            # 1. Generate augmentation
            all_pairs = train_examples + [(test_inp, test_out)]
            aug_pairs, trans_id, color_perm = augment_fn(
                all_pairs,
                do_color_perm=(aug_idx > 0),
                do_dihedral=(aug_idx > 0),
                do_translation=False,
            )

            aug_test_inp, aug_test_out = aug_pairs[-1]
            inp_seq, lbl_seq = grid_to_seq_fn(aug_test_inp, aug_test_out, do_translation=False)

            inp_tensor = torch.tensor(inp_seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
            pid_tensor = torch.zeros(1, dtype=torch.long, device=device)

            # 2. Run K stochastic PTRM rollouts on this augmentation
            rollout_res = self.engine.run_rollouts(
                model=model,
                inputs=inp_tensor,
                puzzle_ids=pid_tensor,
                config=self.config,
            )

            # 3. Select best rollout for this augmentation using Q-head
            best_k = rollout_res.best_q_indices[0].item()
            best_seq = rollout_res.all_preds[0, best_k].cpu().numpy()
            best_q = rollout_res.all_q_scores[0, best_k].item()

            # Decode to grid
            pred_grid = seq_to_grid_fn(best_seq)

            # 4. Inverse transform back to canonical space
            if aug_idx > 0:
                try:
                    pred_grid = inverse_augment_fn(pred_grid.astype(np.uint8), trans_id, color_perm)
                except Exception:
                    continue

            pred_grid = np.array(pred_grid, dtype=np.int64)
            grid_key = pred_grid.tobytes() + b"|" + f"{pred_grid.shape}".encode()

            candidates_map[grid_key]["count"] += 1
            candidates_map[grid_key]["sum_q"] += best_q
            candidates_map[grid_key]["max_q"] = max(candidates_map[grid_key]["max_q"], best_q)
            candidates_map[grid_key]["grid"] = pred_grid

        # 5. Rank aggregated candidates across augmentations
        sorted_candidates = sorted(
            candidates_map.values(),
            key=lambda c: (c["count"], c["sum_q"] / max(1, c["count"])),
            reverse=True,
        )

        top_grids = [c["grid"] for c in sorted_candidates if c["grid"] is not None]

        # 6. Check Top-1, Top-2, and Top-K correctness
        top1_correct = False
        top2_correct = False
        top_k_correct = False

        if len(top_grids) > 0:
            top1_correct = (top_grids[0].shape == test_out.shape and np.array_equal(top_grids[0], test_out))

        if len(top_grids) > 1:
            for g in top_grids[:2]:
                if g.shape == test_out.shape and np.array_equal(g, test_out):
                    top2_correct = True
                    break
        else:
            top2_correct = top1_correct

        for g in top_grids[:self.submission_k]:
            if g.shape == test_out.shape and np.array_equal(g, test_out):
                top_k_correct = True
                break

        return {
            "is_correct_top1": top1_correct,
            "is_correct_top2": top2_correct,
            "is_correct_top_k": top_k_correct,
            "num_unique_predictions": len(sorted_candidates),
            "top_candidates": top_grids[:self.submission_k],
        }


# ============================================================
# GENERAL REASONING EVALUATOR (SUDOKU, PPBENCH, MAZE)
# ============================================================
class PTRMPuzzleEvaluator:
    """
    Evaluator for fixed-grid and sequence reasoning benchmarks (Sudoku-Extreme, PPBench, Maze-Hard).
    Computes pass@K, best-Q@K, and mode@K metrics across test batches.
    """

    def __init__(self, config: Optional[PTRMConfig] = None):
        self.config = config or PTRMConfig()
        self.engine = PTRMRolloutEngine(self.config)

    @torch.no_grad()
    def evaluate_dataset(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        ignore_index: int = 0,
    ) -> Dict[str, float]:
        """
        Evaluate full dataset with PTRM.
        """
        model.eval()
        total_samples = 0
        total_pass_k = 0
        total_best_q = 0
        total_mode = 0

        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["labels"].to(device)
            puzzle_ids = batch.get("puzzle_identifiers", None)
            if puzzle_ids is not None:
                puzzle_ids = puzzle_ids.to(device)

            B = inputs.shape[0]
            rollout_res = self.engine.run_rollouts(
                model=model,
                inputs=inputs,
                puzzle_ids=puzzle_ids,
                config=self.config,
            )

            # Metrics
            _, is_pass = rollout_res.compute_pass_at_k(targets, ignore_index=ignore_index)
            _, is_best_q = rollout_res.compute_best_q_accuracy(targets, ignore_index=ignore_index)
            _, is_mode = rollout_res.compute_mode_accuracy(targets, ignore_index=ignore_index)

            total_samples += B
            total_pass_k += is_pass.sum().item()
            total_best_q += is_best_q.sum().item()
            total_mode += is_mode.sum().item()

        return {
            "samples": total_samples,
            "pass@K": total_pass_k / max(1, total_samples),
            "best_q@K": total_best_q / max(1, total_samples),
            "mode@K": total_mode / max(1, total_samples),
        }
