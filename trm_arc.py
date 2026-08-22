"""
Tiny Recursive Model (TRM) for ARC-AGI
========================================
Implementation of "Less is More: Recursive Reasoning with Tiny Networks"
(arXiv:2510.04871v1, Alexia Jolicoeur-Martineau, Samsung SAIL Montréal)

A 7M-parameter recursive reasoning model that achieves 45% on ARC-AGI-1
using a single 2-layer Transformer network with deep supervision + ACT + EMA.

Usage:
    # Train on ARC-AGI-1
    python trm_arc.py --mode train --data arc_data --epochs 100000

    # Quick smoke test (few epochs, small batch)
    python trm_arc.py --mode train --data arc_data --epochs 5 --batch-size 4 --eval-interval 2 --smoke-test

    # Evaluate a saved checkpoint
    python trm_arc.py --mode eval --data arc_data --checkpoint trm_checkpoints/best.pt

Architecture from paper (Algorithm 3):
    def latent_recursion(x, y, z, n=6):
        for i in range(n):       # latent reasoning
            z = net(x, y, z)
        y = net(y, z)            # refine output answer
        return y, z

    def deep_recursion(x, y, z, n=6, T=3):
        with torch.no_grad():
            for j in range(T-1):
                y, z = latent_recursion(x, y, z, n)
        y, z = latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), output_head(y), Q_head(y)
"""

from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ptrm import PTRMConfig, PTRMRolloutEngine, ARCPTRMEvaluator


# ============================================================
# CONSTANTS
# ============================================================
ARC_MAX_GRID = 30
SEQ_LEN = ARC_MAX_GRID * ARC_MAX_GRID  # 900
VOCAB_SIZE = 12       # PAD=0, EOS=1, colors 0-9 → tokens 2-11
PAD_ID = 0
EOS_ID = 1
IGNORE_LABEL_ID = 0   # Same as PAD: positions to ignore in loss
COLOR_OFFSET = 2      # color c → token c+2


# ============================================================
# DATA AUGMENTATION
# ============================================================
def dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply one of 8 dihedral-group transforms (D4 symmetry)."""
    g = grid.copy()
    if transform_id >= 4:
        g = np.fliplr(g)
    rot = transform_id % 4
    g = np.rot90(g, k=rot)
    return g


def inverse_dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """Inverse of dihedral_transform."""
    g = grid.copy()
    rot = transform_id % 4
    g = np.rot90(g, k=(4 - rot) % 4)
    if transform_id >= 4:
        g = np.fliplr(g)
    return g


def augment_arc_puzzle(
    examples: List[Tuple[np.ndarray, np.ndarray]],
    do_color_perm: bool = True,
    do_dihedral: bool = True,
    do_translation: bool = True,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, np.ndarray]:
    """
    Apply random augmentation to a list of ARC input/output grid pairs.
    Returns (augmented_examples, transform_id, color_mapping).
    """
    # Color permutation: keep 0 (black) fixed, shuffle 1-9
    if do_color_perm:
        perm = np.concatenate([
            np.array([0], dtype=np.uint8),
            np.random.permutation(np.arange(1, 10, dtype=np.uint8))
        ])
    else:
        perm = np.arange(10, dtype=np.uint8)

    # Dihedral transform
    trans_id = np.random.randint(0, 8) if do_dihedral else 0

    augmented = []
    for inp, out in examples:
        a_inp = dihedral_transform(perm[inp], trans_id)
        a_out = dihedral_transform(perm[out], trans_id)
        augmented.append((a_inp, a_out))

    return augmented, trans_id, perm


def inverse_augment_grid(grid: np.ndarray, trans_id: int, color_perm: np.ndarray) -> np.ndarray:
    """Inverse augmentation on a single grid."""
    inv_perm = np.argsort(color_perm).astype(np.uint8)
    return inv_perm[inverse_dihedral_transform(grid, trans_id)]


def grid_to_seq(
    inp: np.ndarray,
    out: np.ndarray,
    do_translation: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert input/output grid pair to flat sequences of length SEQ_LEN.
    Encoding: PAD=0, EOS=1, color c → c+2.
    Grid is placed at a random offset (translation augmentation).
    """
    max_rows = max(inp.shape[0], out.shape[0])
    max_cols = max(inp.shape[1], out.shape[1])

    if do_translation:
        pad_r = np.random.randint(0, ARC_MAX_GRID - max_rows + 1)
        pad_c = np.random.randint(0, ARC_MAX_GRID - max_cols + 1)
    else:
        pad_r = pad_c = 0

    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        # Place grid with color offset
        padded = np.zeros((ARC_MAX_GRID, ARC_MAX_GRID), dtype=np.int64)
        padded[pad_r:pad_r + nrow, pad_c:pad_c + ncol] = grid + COLOR_OFFSET

        # Add EOS markers at the boundaries
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARC_MAX_GRID:
            padded[eos_row, pad_c:eos_col] = EOS_ID
        if eos_col < ARC_MAX_GRID:
            padded[pad_r:eos_row, eos_col] = EOS_ID

        result.append(padded.flatten())

    return result[0], result[1]


def seq_to_grid(seq: np.ndarray) -> np.ndarray:
    """
    Convert a flat sequence back to a 2D grid.
    Finds the bounding box delimited by EOS tokens.
    """
    grid = seq.reshape(ARC_MAX_GRID, ARC_MAX_GRID)

    # Find rows/cols with actual content (>= COLOR_OFFSET)
    content_mask = grid >= COLOR_OFFSET
    if not content_mask.any():
        return np.zeros((1, 1), dtype=np.int64)

    rows = np.where(content_mask.any(axis=1))[0]
    cols = np.where(content_mask.any(axis=0))[0]

    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()

    cropped = grid[r_min:r_max + 1, c_min:c_max + 1].copy()
    # Convert back: token → color
    result = np.where(cropped >= COLOR_OFFSET, cropped - COLOR_OFFSET, 0)
    return result.astype(np.int64)


# ============================================================
# ARC DATASET
# ============================================================
@dataclass
class ARCTask:
    """A single ARC task with train and test examples."""
    task_id: str
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_examples: List[Tuple[np.ndarray, np.ndarray]]


def load_arc_tasks(data_dir: str, split: str = "training") -> List[ARCTask]:
    """Load ARC-AGI tasks from JSON files."""
    task_dir = Path(data_dir) / split
    tasks = []
    for fp in sorted(task_dir.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        train_examples = [
            (np.array(ex["input"], dtype=np.uint8), np.array(ex["output"], dtype=np.uint8))
            for ex in data.get("train", [])
        ]
        test_examples = [
            (
                np.array(ex["input"], dtype=np.uint8),
                np.array(ex["output"], dtype=np.uint8) if "output" in ex else np.zeros((1, 1), dtype=np.uint8)
            )
            for ex in data.get("test", [])
        ]
        tasks.append(ARCTask(task_id=fp.stem, train_examples=train_examples, test_examples=test_examples))

    return tasks


class ARCPuzzleDataset(Dataset):
    """
    Dataset that serves individual (input, label) pairs for TRM training.

    Each ARC task is expanded:
    - Each task produces multiple (train_example_input, train_example_output) pairs
    - Each pair is augmented `num_aug` times (color perm, dihedral, translation)
    - Puzzle identifiers link all examples from the same task+augmentation
    """

    def __init__(
        self,
        tasks: List[ARCTask],
        num_aug: int = 1000,
        include_test: bool = False,
        do_translation: bool = True,
        puzzle_id_start: int = 1,
    ):
        self.tasks = tasks
        self.num_aug = num_aug
        self.include_test = include_test
        self.do_translation = do_translation

        # Build index: list of (task_idx, example_idx, aug_idx)
        # Puzzle identifier = unique per (task, augmentation)
        self.samples: List[Tuple[int, int, int]] = []
        self.puzzle_identifiers: Dict[Tuple[int, int], int] = {}
        pid = puzzle_id_start

        for t_idx, task in enumerate(tasks):
            examples = task.train_examples
            if include_test:
                examples = examples + task.test_examples
            for aug_idx in range(max(1, num_aug)):
                key = (t_idx, aug_idx)
                self.puzzle_identifiers[key] = pid
                pid += 1
                for ex_idx in range(len(examples)):
                    self.samples.append((t_idx, ex_idx, aug_idx))

        self.num_puzzle_identifiers = pid
        print(f"ARCPuzzleDataset: {len(self.samples)} samples, {pid} puzzle IDs, "
              f"{len(tasks)} tasks, {num_aug} augmentations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_idx, ex_idx, aug_idx = self.samples[idx]
        task = self.tasks[t_idx]

        examples = task.train_examples
        if self.include_test:
            examples = examples + task.test_examples

        inp_grid, out_grid = examples[ex_idx]

        # Apply augmentation (same seed for all examples in same puzzle)
        rng_state = np.random.get_state()
        np.random.seed(t_idx * 100000 + aug_idx)
        aug_examples, trans_id, color_perm = augment_arc_puzzle(
            [(inp_grid, out_grid)],
            do_color_perm=(aug_idx > 0),
            do_dihedral=(aug_idx > 0),
            do_translation=self.do_translation and (aug_idx > 0),
        )
        np.random.set_state(rng_state)

        a_inp, a_out = aug_examples[0]

        # Convert to sequences
        inp_seq, out_seq = grid_to_seq(
            a_inp, a_out,
            do_translation=self.do_translation and (aug_idx > 0)
        )

        puzzle_id = self.puzzle_identifiers[(t_idx, aug_idx)]

        return {
            "inputs": torch.tensor(inp_seq, dtype=torch.long),
            "labels": torch.tensor(out_seq, dtype=torch.long),
            "puzzle_identifiers": torch.tensor(puzzle_id, dtype=torch.long),
        }


class ARCGroupedBatchSampler:
    """
    Batch sampler that groups examples from the same puzzle together.
    Each batch contains examples from `batch_size / group_size` different puzzles,
    with `group_size` examples per puzzle.
    """

    def __init__(self, dataset: ARCPuzzleDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group samples by (task_idx, aug_idx)
        self.groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i, (t_idx, ex_idx, aug_idx) in enumerate(dataset.samples):
            self.groups[(t_idx, aug_idx)].append(i)

        self.group_keys = list(self.groups.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.group_keys)

        batch = []
        for key in self.group_keys:
            indices = self.groups[key]
            batch.extend(indices)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        if batch:
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        return (total + self.batch_size - 1) // self.batch_size


# ============================================================
# MODEL COMPONENTS
# ============================================================
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization (JAX-style)."""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            lower, upper = -2.0, 2.0
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Root Mean Square Layer Normalization."""
    dtype = x.dtype
    x = x.float()
    variance = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x.to(dtype)


class CastedLinear(nn.Module):
    """Linear layer that casts weights to input dtype for mixed precision."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_features, in_features), std=1.0 / (in_features ** 0.5))
        )
        self.bias_param = None
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        b = self.bias_param.to(x.dtype) if self.bias_param is not None else None
        return F.linear(x, w, b)


class CastedEmbedding(nn.Module):
    """Embedding that casts to specified dtype."""

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to query and key tensors."""
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # QKV projection
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        # Output projection
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cos_sin=None) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # q, k, v: [B, L, num_heads, head_dim]

        # Apply RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention: [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (non-causal)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # [B, H, L, D] -> [B, L, H, D] -> [B, L, hidden_size]
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.hidden_size)

        return self.o_proj(attn_output)


def _find_multiple(a, b):
    """Ceiling to nearest multiple of b."""
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ============================================================
# TRM BLOCKS
# ============================================================
class TRMBlock(nn.Module):
    """
    Single Transformer / MLP block for TRM.
    TRM-Att: Post-norm Self-Attention + SwiGLU FFN
    TRM-MLP: Post-norm Sequence SwiGLU + Hidden SwiGLU FFN
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        rms_eps: float = 1e-5,
        mlp_t: bool = False,
        seq_len: int = 916,
    ):
        super().__init__()
        self.mlp_t = mlp_t
        if mlp_t:
            self.seq_mlp = SwiGLU(seq_len, expansion)
            self.attn = None
        else:
            self.attn = Attention(hidden_size, num_heads)
            self.seq_mlp = None
        self.ffn = SwiGLU(hidden_size, expansion)
        self.rms_eps = rms_eps

    def forward(self, x: torch.Tensor, cos_sin=None) -> torch.Tensor:
        if self.mlp_t:
            # Sequence MLP (transpose [B, L, D] -> [B, D, L])
            x_t = x.transpose(1, 2)
            out = self.seq_mlp(x_t).transpose(1, 2)
            x = rms_norm(x + out, eps=self.rms_eps)
        else:
            # Self-attention with post-norm residual
            x = rms_norm(x + self.attn(x, cos_sin=cos_sin), eps=self.rms_eps)
        # Feed-forward with post-norm residual
        x = rms_norm(x + self.ffn(x), eps=self.rms_eps)
        return x


class TRMReasoningModule(nn.Module):
    """Stack of TRM blocks with input injection."""

    def __init__(self, blocks: List[TRMBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin=None) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for block in self.blocks:
            hidden_states = block(hidden_states, cos_sin=cos_sin)
        return hidden_states


# ============================================================
# LOSS FUNCTIONS
# ============================================================
def stablemax(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """StableMax function: s(x) = 1/(1-x) if x<0, else x+1."""
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-StableMax: log(s(x) / sum(s(x)))."""
    s_x = stablemax(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, valid_mask=None) -> torch.Tensor:
    """
    Cross-entropy loss using StableMax instead of Softmax.
    Returns per-element loss (same shape as labels).
    """
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = (labels != IGNORE_LABEL_ID)
    safe_labels = torch.where(valid_mask, labels, 0)
    pred_logprobs = torch.gather(logprobs, dim=-1, index=safe_labels.long().unsqueeze(-1)).squeeze(-1)
    return -torch.where(valid_mask, pred_logprobs, torch.zeros_like(pred_logprobs))


# ============================================================
# TRM INNER MODEL
# ============================================================
@dataclass
class TRMCarry:
    """Latent state carried across supervision steps."""
    y: torch.Tensor   # solution embedding [B, L+P, D]
    z: torch.Tensor   # latent reasoning    [B, L+P, D]


@dataclass
class TRMConfig:
    """TRM hyperparameters."""
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    num_layers: int = 2          # L_layers in paper (tiny = 2)
    n_latent_cycles: int = 6     # L_cycles: latent reasoning steps per recursion
    n_deep_cycles: int = 3       # H_cycles: number of full recursion cycles
    halt_max_steps: int = 16     # N_sup: max deep supervision steps
    halt_exploration_prob: float = 0.1
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    puzzle_emb_dim: int = 512    # puzzle embedding dimension
    puzzle_emb_len: int = 16     # number of puzzle embedding positions
    forward_dtype: str = "float32"  # Use float32 for CPU compatibility, bfloat16 for GPU
    mlp_t: bool = False          # Use MLP on sequence length (TRM-MLP) instead of Attention (TRM-Att)


class TRMInner(nn.Module):
    """
    Core TRM model.

    Architecture:
    - Input embedding: token_embed(x) + puzzle_embed + position_embed
    - Single 2-layer network (shared for latent and solution updates)
    - LM head: projects y back to vocab logits
    - Q head: predicts halting probability / correctness

    Forward pass (per supervision step):
        For T cycles (T-1 without grad, 1 with grad):
            For n steps:
                z = net(z, y + x)   # latent reasoning
            y = net(y, z)           # refine solution
    """

    def __init__(self, config: TRMConfig, num_puzzle_ids: int):
        super().__init__()
        self.config = config
        D = config.hidden_size
        self.fwd_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(D)
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding
        self.embed_tokens = CastedEmbedding(VOCAB_SIZE, D, init_std=embed_init_std, cast_to=self.fwd_dtype)

        # Puzzle embedding: each puzzle gets puzzle_emb_dim floats,
        # padded and reshaped to [puzzle_emb_len, hidden_size]
        self.puzzle_emb_len = config.puzzle_emb_len
        self.puzzle_emb_dim = config.puzzle_emb_dim
        if config.puzzle_emb_dim > 0 and num_puzzle_ids > 0:
            # Sparse-style: store only puzzle_emb_dim per puzzle (=512 by default)
            # At forward time, pad to puzzle_emb_len * D and reshape
            self.puzzle_emb_weight = nn.Parameter(
                torch.zeros(num_puzzle_ids, config.puzzle_emb_dim, dtype=self.fwd_dtype)
            )
        else:
            self.puzzle_emb_weight = None

        # Total sequence length including puzzle embedding prefix
        total_seq_len = SEQ_LEN + self.puzzle_emb_len

        # Rotary Position Embedding (for attention variant)
        self.rotary_emb = RotaryEmbedding(
            dim=D // config.num_heads,
            max_position_embeddings=total_seq_len,
            base=config.rope_theta,
        )

        # The single shared network (2-layer Transformer or MLP)
        self.net = TRMReasoningModule([
            TRMBlock(
                hidden_size=D,
                num_heads=config.num_heads,
                expansion=config.expansion,
                rms_eps=config.rms_eps,
                mlp_t=config.mlp_t,
                seq_len=total_seq_len,
            )
            for _ in range(config.num_layers)
        ])

        # Output heads
        self.lm_head = CastedLinear(D, VOCAB_SIZE, bias=False)
        self.q_head = CastedLinear(D, 2, bias=True)

        # Initial states for y and z
        self.register_buffer("y_init", trunc_normal_init_(torch.empty(D, dtype=self.fwd_dtype), std=1))
        self.register_buffer("z_init", trunc_normal_init_(torch.empty(D, dtype=self.fwd_dtype), std=1))

        # Initialize Q-head to near-zero for stable bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias_param.fill_(-5.0)

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_ids: torch.Tensor) -> torch.Tensor:
        """Compute input embeddings with puzzle prefix."""
        # Token embedding: [B, L] -> [B, L, D]
        emb = self.embed_tokens(inputs.int())  # [B, SEQ_LEN, D]

        # Puzzle embedding prefix
        if self.puzzle_emb_weight is not None:
            D = self.config.hidden_size
            # [B, puzzle_emb_dim]
            p_emb = F.embedding(puzzle_ids, self.puzzle_emb_weight.to(self.fwd_dtype))
            # Pad to puzzle_emb_len * D
            target_size = self.puzzle_emb_len * D
            pad_count = target_size - p_emb.shape[-1]
            if pad_count > 0:
                p_emb = F.pad(p_emb, (0, pad_count))
            # Reshape to [B, puzzle_emb_len, D]
            p_emb = p_emb.view(-1, self.puzzle_emb_len, D)
            # Prepend puzzle embedding
            emb = torch.cat([p_emb, emb], dim=1)  # [B, P+L, D]

        return self.embed_scale * emb

    def empty_carry(self, batch_size: int) -> TRMCarry:
        """Create empty carry tensors."""
        total_len = SEQ_LEN + self.puzzle_emb_len
        D = self.config.hidden_size
        return TRMCarry(
            y=torch.empty(batch_size, total_len, D, dtype=self.fwd_dtype, device=self.y_init.device),
            z=torch.empty(batch_size, total_len, D, dtype=self.fwd_dtype, device=self.z_init.device),
        )

    def reset_carry(self, carry: TRMCarry, reset_mask: torch.Tensor) -> TRMCarry:
        """Reset carry for halted sequences (re-initialize from learned init)."""
        mask = reset_mask.view(-1, 1, 1)
        return TRMCarry(
            y=torch.where(mask, self.y_init.expand_as(carry.y), carry.y),
            z=torch.where(mask, self.z_init.expand_as(carry.z), carry.z),
        )

    def forward(
        self,
        carry: TRMCarry,
        inputs: torch.Tensor,
        puzzle_ids: torch.Tensor,
    ) -> Tuple[TRMCarry, torch.Tensor, torch.Tensor]:
        """
        Single deep recursion step.

        Args:
            carry: Previous (y, z) latent state
            inputs: Input token IDs [B, SEQ_LEN]
            puzzle_ids: Puzzle identifier IDs [B]

        Returns:
            new_carry: Updated (y, z) with gradients detached
            logits: Output predictions [B, SEQ_LEN, VOCAB_SIZE]
            q_halt_logits: Halting logits [B]
        """
        cos_sin = self.rotary_emb()

        # Input embeddings (with puzzle prefix)
        x = self._input_embeddings(inputs, puzzle_ids)

        y, z = carry.y, carry.z

        # === Deep Recursion ===
        # T-1 cycles without gradient
        with torch.no_grad():
            for _ in range(self.config.n_deep_cycles - 1):
                # n latent reasoning steps
                for _ in range(self.config.n_latent_cycles):
                    z = self.net(z, y + x, cos_sin=cos_sin)
                # 1 solution refinement step
                y = self.net(y, z, cos_sin=cos_sin)

        # 1 cycle with gradient (backprop through this)
        for _ in range(self.config.n_latent_cycles):
            z = self.net(z, y + x, cos_sin=cos_sin)
        y = self.net(y, z, cos_sin=cos_sin)

        # === Output Heads ===
        # Detach carry for next supervision step
        new_carry = TRMCarry(y=y.detach(), z=z.detach())

        # LM output: skip puzzle embedding prefix
        logits = self.lm_head(y[:, self.puzzle_emb_len:])  # [B, SEQ_LEN, VOCAB_SIZE]

        # Q-head: use first position (puzzle embedding position)
        q_logits = self.q_head(y[:, 0]).float()  # [B, 2]
        q_halt = q_logits[..., 0]      # halt logit

        return new_carry, logits, q_halt


# ============================================================
# TRM WRAPPER WITH ACT
# ============================================================
class TRM(nn.Module):
    """
    Full TRM model with Adaptive Computational Time (ACT) wrapper.

    Training: Uses ACT to decide when to halt (skip remaining supervision steps).
    Evaluation: Always runs full halt_max_steps supervision steps.
    """

    def __init__(self, config: TRMConfig, num_puzzle_ids: int):
        super().__init__()
        self.config = config
        self.inner = TRMInner(config, num_puzzle_ids)

    def forward_step(
        self,
        carry: TRMCarry,
        inputs: torch.Tensor,
        puzzle_ids: torch.Tensor,
        labels: torch.Tensor,
        halted: torch.Tensor,
        steps: torch.Tensor,
        new_inputs: torch.Tensor,
        new_puzzle_ids: torch.Tensor,
        new_labels: torch.Tensor,
    ) -> Tuple[TRMCarry, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single supervision step with ACT.

        When a sequence is "halted", it gets replaced with the next batch item.
        """
        # Reset carry and swap data for halted sequences
        carry = self.inner.reset_carry(carry, halted)
        mask_expand = halted.unsqueeze(-1)
        inputs = torch.where(mask_expand, new_inputs, inputs)
        labels = torch.where(mask_expand, new_labels, labels)
        puzzle_ids = torch.where(halted, new_puzzle_ids, puzzle_ids)
        steps = torch.where(halted, torch.zeros_like(steps), steps)

        # Forward through inner model
        new_carry, logits, q_halt = self.inner(carry, inputs, puzzle_ids)

        steps = steps + 1

        return new_carry, logits, q_halt, inputs, labels, puzzle_ids, steps, halted

    def compute_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        q_halt: torch.Tensor,
        halted: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute LM loss + Q-halt loss."""
        # Masks
        valid_mask = (labels != IGNORE_LABEL_ID)
        loss_counts = valid_mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1).float()

        # StableMax cross-entropy loss
        lm_loss = (stablemax_cross_entropy(logits, labels, valid_mask=valid_mask) / loss_divisor).sum()

        # Predictions and correctness
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            is_correct = valid_mask & (preds == labels)
            seq_is_correct = (is_correct.sum(-1) == loss_counts)

        # Q-halt loss: binary CE for "has the model solved it?"
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt, seq_is_correct.float(), reduction="sum"
        )

        total_loss = lm_loss + 0.5 * q_halt_loss

        # Metrics
        valid_metrics = halted & (loss_counts > 0)
        metrics = {
            "lm_loss": lm_loss.item(),
            "q_halt_loss": q_halt_loss.item(),
            "accuracy": torch.where(
                valid_metrics,
                (is_correct.float() / loss_divisor).sum(-1),
                torch.zeros_like(q_halt)
            ).sum().item(),
            "exact_accuracy": (valid_metrics & seq_is_correct).sum().item(),
            "count": valid_metrics.sum().item(),
        }

        return total_loss, metrics


# ============================================================
# EMA HELPER
# ============================================================
class EMAHelper:
    """Exponential Moving Average of model weights."""

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow: Dict[str, torch.Tensor] = {}

    def register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

    def apply(self, model: nn.Module):
        """Copy EMA weights to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def copy_to_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of model with EMA weights."""
        model_copy = copy.deepcopy(model)
        self.apply(model_copy)
        return model_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================
def cosine_schedule_with_warmup(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 1.0) -> float:
    """Cosine decay learning rate with linear warmup."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine_decay


# ============================================================
# TRAINING LOOP
# ============================================================
def train(
    data_dir: str = "arc_data",
    epochs: int = 100000,
    batch_size: int = 768,
    lr: float = 1e-4,
    puzzle_emb_lr: float = 1e-2,
    weight_decay: float = 0.1,
    warmup_steps: int = 2000,
    eval_interval: int = 10000,
    num_aug: int = 1000,
    ema_rate: float = 0.999,
    use_ema: bool = True,
    checkpoint_dir: str = "trm_checkpoints",
    checkpoint_path: Optional[str] = None,
    smoke_test: bool = False,
    device_str: str = "auto",
    # TRM architecture hyperparams
    hidden_size: int = 512,
    num_heads: int = 8,
    num_layers: int = 2,
    n_latent_cycles: int = 6,
    n_deep_cycles: int = 3,
    halt_max_steps: int = 16,
):
    """Main training loop."""
    # Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    use_amp = device.type == "cuda"
    fwd_dtype = "bfloat16" if (use_amp and torch.cuda.is_bf16_supported()) else "float32"

    print("=" * 80)
    print("TRM: Tiny Recursive Model for ARC-AGI")
    print("Paper: arXiv:2510.04871v1")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Forward dtype: {fwd_dtype}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Architecture: {num_layers}-layer, D={hidden_size}, H={num_heads}")
    print(f"Recursion: n={n_latent_cycles}, T={n_deep_cycles}")
    print(f"Deep supervision: max {halt_max_steps} steps")
    print(f"Augmentations: {num_aug}")
    if smoke_test:
        print("*** SMOKE TEST MODE ***")
        num_aug = 2
        epochs = min(epochs, 5)
        halt_max_steps = 2
        eval_interval = min(eval_interval, 2)
    print("=" * 80)

    # Load data
    print("\nLoading ARC tasks...")
    train_tasks = load_arc_tasks(data_dir, "training")
    eval_tasks = load_arc_tasks(data_dir, "evaluation")
    print(f"Training tasks: {len(train_tasks)}, Evaluation tasks: {len(eval_tasks)}")

    # Create datasets
    train_dataset = ARCPuzzleDataset(train_tasks, num_aug=num_aug, do_translation=True)

    # Create model
    config = TRMConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        n_latent_cycles=n_latent_cycles,
        n_deep_cycles=n_deep_cycles,
        halt_max_steps=halt_max_steps,
        forward_dtype=fwd_dtype,
        puzzle_emb_dim=hidden_size,
    )
    model = TRM(config, num_puzzle_ids=train_dataset.num_puzzle_identifiers)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  ~ {total_params / 1e6:.1f}M parameters")

    # Optimizers
    # Separate puzzle embeddings from rest of the model
    puzzle_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "puzzle_emb" in name:
            puzzle_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        other_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    puzzle_optimizer = None
    if puzzle_params:
        puzzle_optimizer = torch.optim.AdamW(
            puzzle_params,
            lr=puzzle_emb_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

    # EMA
    ema = None
    if use_ema:
        ema = EMAHelper(mu=ema_rate)
        ema.register(model)

    # Load checkpoint
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if puzzle_optimizer and "puzzle_optimizer" in ckpt:
            puzzle_optimizer.load_state_dict(ckpt["puzzle_optimizer"])
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # DataLoader
    # For efficiency, we just randomly sample batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Training
    os.makedirs(checkpoint_dir, exist_ok=True)
    total_steps = epochs
    best_eval_acc = 0.0
    step = start_epoch

    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and fwd_dtype == "float16")

    model.train()
    epoch_iter = iter(train_loader)

    for step in range(start_epoch, total_steps):
        t0 = time.perf_counter()

        # Get batch (cycle through dataset)
        try:
            batch = next(epoch_iter)
        except StopIteration:
            epoch_iter = iter(train_loader)
            batch = next(epoch_iter)

        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        puzzle_ids = batch["puzzle_identifiers"].to(device)

        B = inputs.shape[0]

        # LR schedule
        lr_mult = cosine_schedule_with_warmup(step, total_steps, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_mult
        if puzzle_optimizer:
            for pg in puzzle_optimizer.param_groups:
                pg["lr"] = puzzle_emb_lr * lr_mult

        # === Deep Supervision Loop ===
        carry = model.inner.empty_carry(B)
        carry = TRMCarry(
            y=carry.y.to(device),
            z=carry.z.to(device),
        )
        halted = torch.ones(B, dtype=torch.bool, device=device)  # Start all halted → will be reset
        steps_counter = torch.zeros(B, dtype=torch.long, device=device)

        total_loss = torch.tensor(0.0, device=device)
        total_metrics: Dict[str, float] = defaultdict(float)
        num_sup_steps = 0

        for sup_step in range(halt_max_steps):
            # Reset halted sequences
            carry = model.inner.reset_carry(carry, halted)
            steps_counter = torch.where(halted, torch.zeros_like(steps_counter), steps_counter)

            # Forward pass
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=getattr(torch, fwd_dtype) if fwd_dtype != "float32" else torch.float32):
                new_carry, logits, q_halt = model.inner(carry, inputs, puzzle_ids)

            # Compute loss
            loss, metrics = model.compute_losses(logits, labels, q_halt, halted)

            # Backward
            if use_amp and fwd_dtype == "float16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss = total_loss + loss.detach()
            for k, v in metrics.items():
                total_metrics[k] += v
            num_sup_steps += 1

            # Update carry (detached)
            carry = new_carry
            steps_counter = steps_counter + 1

            # ACT: determine halting
            is_last_step = (sup_step >= halt_max_steps - 1)
            with torch.no_grad():
                halted = torch.ones(B, dtype=torch.bool, device=device) if is_last_step else (q_halt > 0)

                # Exploration: sometimes don't halt early
                if model.training and not is_last_step:
                    explore = (torch.rand(B, device=device) < config.halt_exploration_prob)
                    min_halt = torch.randint(2, halt_max_steps + 1, (B,), device=device)
                    halted = halted & (steps_counter >= min_halt) | (halted & ~explore)

            # If all halted, break early
            if halted.all() and not is_last_step:
                break

        # Optimizer step
        if use_amp and fwd_dtype == "float16":
            scaler.step(optimizer)
            scaler.update()
        else:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if puzzle_optimizer:
            puzzle_optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if puzzle_optimizer:
            puzzle_optimizer.zero_grad(set_to_none=True)

        # EMA update
        if ema:
            ema.update(model)

        dt = time.perf_counter() - t0

        # Logging
        if step % max(1, eval_interval // 10) == 0 or step < 10:
            count = max(1, total_metrics["count"])
            print(
                f"[Step {step:6d}/{total_steps}] "
                f"loss={total_loss.item() / max(1, num_sup_steps):.4f} "
                f"acc={total_metrics['accuracy'] / count:.4f} "
                f"exact={total_metrics['exact_accuracy'] / count:.4f} "
                f"sup_steps={num_sup_steps} "
                f"lr={lr * lr_mult:.2e} "
                f"dt={dt:.2f}s"
            )

        # Evaluation
        if (step > 0 and step % eval_interval == 0) or step == total_steps - 1:
            eval_model = model
            if ema:
                eval_model = ema.copy_to_model(model)

            eval_acc = evaluate(
                eval_model, eval_tasks, config,
                num_aug=min(num_aug, 100) if not smoke_test else 2,
                device=device,
                max_tasks=10 if smoke_test else 0,
            )

            model.train()

            # Save checkpoint
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": step,
                "eval_acc": eval_acc,
                "config": config.__dict__,
            }
            if puzzle_optimizer:
                save_dict["puzzle_optimizer"] = puzzle_optimizer.state_dict()
            if ema:
                save_dict["ema"] = ema.state_dict()

            torch.save(save_dict, os.path.join(checkpoint_dir, "latest.pt"))

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save(save_dict, os.path.join(checkpoint_dir, "best.pt"))
                print(f"  *** New best: {eval_acc:.4f} ***")

    print(f"\n{'='*80}")
    print(f"Training complete! Best eval accuracy: {best_eval_acc:.4f}")
    print(f"{'='*80}")


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(
    model: TRM,
    tasks: List[ARCTask],
    config: TRMConfig,
    num_aug: int = 100,
    device: torch.device = torch.device("cpu"),
    max_tasks: int = 0,
    use_ptrm: bool = False,
    ptrm_k: int = 25,
    ptrm_sigma: float = 0.2,
    ptrm_depth: int = 16,
    ptrm_selector: str = "best_q",
) -> float:
    """
    Evaluate TRM on ARC tasks with test-time compute scaling and majority voting.

    Standard TRM (use_ptrm=False):
        Runs 1 deterministic rollout per augmentation and votes across augmentations.

    Probabilistic TRM (use_ptrm=True, arXiv:2605.19943v1):
        Runs K parallel stochastic rollouts with Gaussian noise sigma per augmentation,
        picks the best rollout per augmentation via Q-head scoring, then votes across
        augmentations.
    """
    model.eval()
    fwd_dtype = getattr(torch, config.forward_dtype)

    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    total_correct_top1 = 0
    total_correct_top2 = 0
    total_tests = 0

    mode_str = f"PTRM (K={ptrm_k}, sigma={ptrm_sigma}, D={ptrm_depth})" if use_ptrm else "Deterministic TRM"
    print(f"\n  Evaluating on {len(tasks)} tasks using {mode_str}, {num_aug} augmentations each...")
    t0 = time.perf_counter()

    ptrm_engine = None
    if use_ptrm:
        ptrm_cfg = PTRMConfig(
            num_rollouts=ptrm_k,
            supervision_steps=ptrm_depth if ptrm_depth > 0 else config.halt_max_steps,
            noise_scale=ptrm_sigma,
            selector=ptrm_selector,
            forward_dtype=config.forward_dtype,
        )
        ptrm_engine = PTRMRolloutEngine(ptrm_cfg)

    for task_idx, task in enumerate(tasks):
        for test_idx, (test_inp, test_out) in enumerate(task.test_examples):
            # Skip dummy outputs
            if test_out.shape == (1, 1) and test_out[0, 0] == 0:
                continue

            candidates_map = defaultdict(lambda: {"count": 0, "sum_q": 0.0, "grid": None})

            for aug_idx in range(num_aug):
                # Create all examples for this task (train + test)
                all_examples = task.train_examples + [(test_inp, test_out)]

                # Apply augmentation
                rng_state = np.random.get_state()
                np.random.seed(task_idx * 100000 + aug_idx + 999999)
                aug_examples, trans_id, color_perm = augment_arc_puzzle(
                    all_examples,
                    do_color_perm=(aug_idx > 0),
                    do_dihedral=(aug_idx > 0),
                    do_translation=False,  # No translation at test time for cleaner de-augmentation
                )
                np.random.set_state(rng_state)

                # Process test example as forward pass
                aug_test_inp, aug_test_out = aug_examples[-1]

                # Convert to sequence
                inp_seq, lbl_seq = grid_to_seq(aug_test_inp, aug_test_out, do_translation=False)
                inp_tensor = torch.tensor(inp_seq, dtype=torch.long).unsqueeze(0).to(device)
                pid_tensor = torch.zeros(1, dtype=torch.long, device=device)

                if use_ptrm and ptrm_engine is not None:
                    # Run K stochastic rollouts
                    rollout_res = ptrm_engine.run_rollouts(
                        model=model,
                        inputs=inp_tensor,
                        puzzle_ids=pid_tensor,
                    )
                    best_k = rollout_res.best_q_indices[0].item()
                    pred_seq = rollout_res.all_preds[0, best_k].cpu().numpy()
                    pred_q = rollout_res.all_q_scores[0, best_k].item()
                else:
                    # Run standard deterministic supervision (no ACT)
                    carry = model.inner.empty_carry(1)
                    carry = TRMCarry(y=carry.y.to(device), z=carry.z.to(device))
                    halted = torch.ones(1, dtype=torch.bool, device=device)
                    carry = model.inner.reset_carry(carry, halted)

                    with torch.amp.autocast(device.type, enabled=(device.type == "cuda"), dtype=fwd_dtype if fwd_dtype != torch.float32 else torch.float32):
                        for _ in range(config.halt_max_steps):
                            carry, logits, q_halt = model.inner(carry, inp_tensor, pid_tensor)

                    pred_seq = torch.argmax(logits[0], dim=-1).cpu().numpy()
                    pred_q = q_halt[0].item()

                pred_grid = seq_to_grid(pred_seq)

                # De-augment
                if aug_idx > 0:
                    try:
                        pred_grid = inverse_augment_grid(pred_grid.astype(np.uint8), trans_id, color_perm)
                    except Exception:
                        continue

                pred_grid = np.array(pred_grid, dtype=np.int64)
                grid_key = pred_grid.tobytes() + b"|" + str(pred_grid.shape).encode()
                candidates_map[grid_key]["count"] += 1
                candidates_map[grid_key]["sum_q"] += pred_q
                candidates_map[grid_key]["grid"] = pred_grid

            # Majority / Q-weighted vote across augmentations
            if candidates_map:
                sorted_candidates = sorted(
                    candidates_map.values(),
                    key=lambda c: (c["count"], c["sum_q"] / max(1, c["count"])),
                    reverse=True,
                )
                top_grids = [c["grid"] for c in sorted_candidates if c["grid"] is not None]

                if len(top_grids) > 0 and top_grids[0].shape == test_out.shape and np.array_equal(top_grids[0], test_out):
                    total_correct_top1 += 1
                    total_correct_top2 += 1
                elif len(top_grids) > 1 and top_grids[1].shape == test_out.shape and np.array_equal(top_grids[1], test_out):
                    total_correct_top2 += 1

            total_tests += 1

        if (task_idx + 1) % max(1, len(tasks) // 10) == 0:
            print(f"    [{task_idx + 1}/{len(tasks)}] Top-1={total_correct_top1}/{total_tests}, Top-2={total_correct_top2}/{total_tests}")

    dt = time.perf_counter() - t0
    acc_top1 = total_correct_top1 / max(1, total_tests)
    acc_top2 = total_correct_top2 / max(1, total_tests)
    print(f"  Evaluation Complete: Top-1 = {total_correct_top1}/{total_tests} ({acc_top1*100:.2f}%), Top-2 = {total_correct_top2}/{total_tests} ({acc_top2*100:.2f}%) [{dt:.1f}s]")

    return acc_top1


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="TRM: Tiny Recursive Model for ARC-AGI (arXiv:2510.04871v1 & PTRM arXiv:2605.19943v1)")

    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Train or evaluate")
    parser.add_argument("--data", default="arc_data", help="Path to ARC data directory")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint for resume/eval")
    parser.add_argument("--checkpoint-dir", default="trm_checkpoints", help="Directory to save checkpoints")

    # Training hyperparameters (paper defaults)
    parser.add_argument("--epochs", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=768, help="Global batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--puzzle-emb-lr", type=float, default=1e-2, help="Puzzle embedding learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="LR warmup steps")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Steps between evaluations")
    parser.add_argument("--num-aug", type=int, default=1000, help="Number of augmentations per puzzle")
    parser.add_argument("--ema-rate", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")

    # Architecture hyperparameters (paper defaults)
    parser.add_argument("--arch", choices=["trm-att", "trm-mlp"], default="trm-att", help="Architecture variant: trm-att (Transformer) or trm-mlp (MLP)")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer/MLP layers")
    parser.add_argument("--n-latent", type=int, default=6, help="Latent reasoning cycles (L_cycles)")
    parser.add_argument("--n-deep", type=int, default=3, help="Deep recursion cycles (H_cycles)")
    parser.add_argument("--halt-max", type=int, default=16, help="Max supervision steps (N_sup)")

    # PTRM Inference hyperparameters (arXiv:2605.19943v1)
    parser.add_argument("--use-ptrm", action="store_true", help="Enable Probabilistic TRM (PTRM) test-time compute scaling")
    parser.add_argument("--ptrm-k", type=int, default=25, help="Number of parallel stochastic rollouts K (paper default: 25)")
    parser.add_argument("--ptrm-sigma", type=float, default=0.2, help="Gaussian noise scale sigma injected at each supervision step (default: 0.2)")
    parser.add_argument("--ptrm-depth", type=int, default=16, help="Supervision depth D for test-time scaling (default: 16)")
    parser.add_argument("--ptrm-selector", choices=["best_q", "mode", "both"], default="best_q", help="Rollout selection strategy")

    # Misc
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test with minimal settings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == "train":
        train(
            data_dir=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            puzzle_emb_lr=args.puzzle_emb_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            eval_interval=args.eval_interval,
            num_aug=args.num_aug,
            ema_rate=args.ema_rate,
            use_ema=not args.no_ema,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_path=args.checkpoint,
            smoke_test=args.smoke_test,
            device_str=args.device,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            n_latent_cycles=args.n_latent,
            n_deep_cycles=args.n_deep,
            halt_max_steps=args.halt_max,
        )

    elif args.mode == "eval":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for evaluation mode")
            sys.exit(1)

        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        fwd_dtype = "bfloat16" if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else "float32"

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config_dict = dict(ckpt.get("config", {}))
        config_dict["mlp_t"] = (args.arch == "trm-mlp") or config_dict.get("mlp_t", False)
        config = TRMConfig(**config_dict)
        config.forward_dtype = fwd_dtype

        eval_tasks = load_arc_tasks(args.data, "evaluation")

        num_puzzle_ids = 1  # Dummy for eval (puzzle embeddings not used with pid=0)
        model = TRM(config, num_puzzle_ids=num_puzzle_ids)
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(device)

        accuracy = evaluate(
            model, eval_tasks, config,
            num_aug=args.num_aug,
            device=device,
            use_ptrm=args.use_ptrm,
            ptrm_k=args.ptrm_k,
            ptrm_sigma=args.ptrm_sigma,
            ptrm_depth=args.ptrm_depth,
            ptrm_selector=args.ptrm_selector,
        )
        print(f"\nFinal evaluation accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")


if __name__ == "__main__":
    main()
