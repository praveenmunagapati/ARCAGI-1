"""
MATHX ARC-AGI TRANSDUCTIVE REASONING ENGINE
Implements Direct Transduction (Test-Time Augmentation + Spatial Consensus Voting)
from 'Combining Induction and Transduction for Abstract Reasoning' (Cornell).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

Grid = np.ndarray

def grid_to_str(g: Grid) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in g)

def parse_grid_str(s: str) -> Optional[Grid]:
    try:
        lines = [line.strip().split() for line in s.strip().split("\n") if line.strip()]
        if not lines:
            return None
        return np.array([[int(c) for c in row] for row in lines], dtype=np.int32)
    except Exception:
        return None


# ============================================================
# DIHEDRAL (D8) & COLOR PERMUTATION AUGMENTATIONS
# ============================================================

class DihedralTransform:
    """8 symmetries of the 2D plane: Rotations {0, 90, 180, 270} x Flips {Identity, Horizontal}."""

    @staticmethod
    def apply(g: Grid, op_idx: int) -> Grid:
        # op_idx from 0 to 7
        rot = op_idx % 4
        flip = op_idx >= 4
        out = g.copy()
        if rot > 0:
            out = np.rot90(out, -rot)
        if flip:
            out = np.fliplr(out)
        return out

    @staticmethod
    def invert(g: Grid, op_idx: int) -> Grid:
        rot = op_idx % 4
        flip = op_idx >= 4
        out = g.copy()
        if flip:
            out = np.fliplr(out)
        if rot > 0:
            out = np.rot90(out, rot)
        return out


class TransductiveAugmentor:
    """Generates augmented task variants and de-augments predicted solutions."""

    @classmethod
    def augment_task(cls, task: dict, op_idx: int, color_perm: Optional[dict[int, int]] = None) -> dict:
        aug_task = {"train": [], "test": []}
        
        for ex in task["train"]:
            inp = DihedralTransform.apply(np.array(ex["input"], dtype=np.int32), op_idx)
            out = DihedralTransform.apply(np.array(ex["output"], dtype=np.int32), op_idx)
            if color_perm:
                inp = cls._apply_color_perm(inp, color_perm)
                out = cls._apply_color_perm(out, color_perm)
            aug_task["train"].append({"input": inp.tolist(), "output": out.tolist()})

        for ex in task["test"]:
            inp = DihedralTransform.apply(np.array(ex["input"], dtype=np.int32), op_idx)
            if color_perm:
                inp = cls._apply_color_perm(inp, color_perm)
            item = {"input": inp.tolist()}
            if "output" in ex:
                out = DihedralTransform.apply(np.array(ex["output"], dtype=np.int32), op_idx)
                if color_perm:
                    out = cls._apply_color_perm(out, color_perm)
                item["output"] = out.tolist()
            aug_task["test"].append(item)

        return aug_task

    @classmethod
    def deaugment_grid(cls, g: Grid, op_idx: int, inv_color_perm: Optional[dict[int, int]] = None) -> Grid:
        out = g.copy()
        if inv_color_perm:
            out = cls._apply_color_perm(out, inv_color_perm)
        return DihedralTransform.invert(out, op_idx)

    @staticmethod
    def _apply_color_perm(g: Grid, perm: dict[int, int]) -> Grid:
        out = g.copy()
        for src, dst in perm.items():
            out[g == src] = dst
        return out


# ============================================================
# TRANSDUCTIVE PROMPT & SPATIAL CONSENSUS VOTING
# ============================================================

class TransductivePromptGenerator:
    """Formats tasks for direct spatial matrix completion."""

    SYSTEM_PROMPT = """You are an advanced visual intelligence system solving ARC-AGI pattern completion.
Look at the input-output demonstration pairs, understand the spatial transformation, and output ONLY the final predicted 2D test grid matrix.
Format your answer as a plain text 2D grid of digits (0-9) separated by spaces. Do not write code or explanations.
"""

    @classmethod
    def format_task_prompt(cls, task: dict) -> str:
        prompt_lines = ["Pattern Demonstrations:"]
        for idx, ex in enumerate(task["train"]):
            inp = np.array(ex["input"], dtype=np.int32)
            out = np.array(ex["output"], dtype=np.int32)
            prompt_lines.append(f"\n[Example {idx + 1} Input]\n{grid_to_str(inp)}")
            prompt_lines.append(f"[Example {idx + 1} Output]\n{grid_to_str(out)}")

        if "test" in task and task["test"]:
            test_inp = np.array(task["test"][0]["input"], dtype=np.int32)
            prompt_lines.append(f"\n[Test Input]\n{grid_to_str(test_inp)}")
            prompt_lines.append("\n[Test Output Matrix]:")

        return "\n".join(prompt_lines)


class SpatialConsensusVoter:
    """Performs spatial majority consensus voting across de-augmented transductive hypotheses."""

    @classmethod
    def vote(cls, candidate_grids: list[Grid]) -> tuple[Optional[Grid], float]:
        if not candidate_grids:
            return None, 0.0

        # Filter candidates by the most common output shape
        shapes = [g.shape for g in candidate_grids]
        shape_counts: dict[tuple[int, int], int] = {}
        for sh in shapes:
            shape_counts[sh] = shape_counts.get(sh, 0) + 1
        best_shape = max(shape_counts.keys(), key=lambda sh: shape_counts[sh])
        valid_grids = [g for g in candidate_grids if g.shape == best_shape]

        if not valid_grids:
            return None, 0.0

        h, w = best_shape
        consensus_grid = np.zeros((h, w), dtype=np.int32)
        total_confidence = 0.0

        # Majority voting per cell
        for r in range(h):
            for c in range(w):
                values = [int(g[r, c]) for g in valid_grids]
                val_counts: dict[int, int] = {}
                for v in values:
                    val_counts[v] = val_counts.get(v, 0) + 1
                winner = max(val_counts.keys(), key=lambda v: val_counts[v])
                consensus_grid[r, c] = winner
                total_confidence += val_counts[winner] / len(values)

        avg_confidence = total_confidence / (h * w)
        return consensus_grid, avg_confidence
