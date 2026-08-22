"""
Unit and Integration Tests for Probabilistic Tiny Recursive Model (PTRM)
========================================================================
Verification suite for arXiv:2605.19943v1 implementation.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from ptrm import (
    PTRMConfig,
    PTRMRolloutEngine,
    PTRMRolloutResult,
    ARCPTRMEvaluator,
    PTRMPuzzleEvaluator,
    LangevinGuidanceSampler,
)
from trm_arc import (
    TRM,
    TRMConfig,
    augment_arc_puzzle,
    inverse_augment_grid,
    grid_to_seq,
    seq_to_grid,
    SEQ_LEN,
    VOCAB_SIZE,
)


class TestPTRMCreator(unittest.TestCase):
    """Test configurations and default presets."""

    def test_presets(self):
        sudoku_cfg = PTRMConfig.for_sudoku_extreme()
        self.assertEqual(sudoku_cfg.num_rollouts, 100)
        self.assertEqual(sudoku_cfg.supervision_steps, 64)
        self.assertEqual(sudoku_cfg.noise_scale, 0.3)

        maze_cfg = PTRMConfig.for_maze_hard()
        self.assertEqual(maze_cfg.num_rollouts, 100)
        self.assertEqual(maze_cfg.noise_scale, 1.0)
        self.assertEqual(maze_cfg.selector, "mode")

        arc_cfg = PTRMConfig.for_arc_agi()
        self.assertEqual(arc_cfg.num_rollouts, 25)
        self.assertEqual(arc_cfg.noise_scale, 0.2)


class TestPTRMRolloutEngine(unittest.TestCase):
    """Test PTRM parallel rollout engine on TRM models."""

    def setUp(self):
        # Create a small lightweight TRM model for fast testing
        self.config = TRMConfig(
            hidden_size=64,
            num_heads=4,
            expansion=2.0,
            num_layers=1,
            n_latent_cycles=2,
            n_deep_cycles=2,
            halt_max_steps=4,
            puzzle_emb_dim=64,
            puzzle_emb_len=4,
            forward_dtype="float32",
        )
        self.model = TRM(self.config, num_puzzle_ids=2)
        self.model.eval()

    def test_rollout_shapes(self):
        B = 2
        K = 5
        D = 3
        seq_len = SEQ_LEN

        inputs = torch.randint(0, VOCAB_SIZE, (B, seq_len))
        puzzle_ids = torch.zeros(B, dtype=torch.long)

        cfg = PTRMConfig(
            num_rollouts=K,
            supervision_steps=D,
            noise_scale=0.2,
            selector="all",
        )
        engine = PTRMRolloutEngine(cfg)
        res = engine.run_rollouts(self.model, inputs, puzzle_ids)

        # Check tensor shapes
        self.assertEqual(res.best_q_pred.shape, (B, seq_len))
        self.assertEqual(res.best_q_score.shape, (B,))
        self.assertEqual(res.best_q_indices.shape, (B,))
        self.assertEqual(res.mode_pred.shape, (B, seq_len))
        self.assertEqual(res.all_preds.shape, (B, K, seq_len))
        self.assertEqual(res.all_q_scores.shape, (B, K))

    def test_stochasticity_vs_determinism(self):
        # Give Q-head non-zero weights to simulate a trained model
        with torch.no_grad():
            self.model.inner.q_head.weight.normal_(0, 1.0)

        B = 1
        K = 10
        inputs = torch.randint(0, VOCAB_SIZE, (B, SEQ_LEN))

        # Test deterministic mode (sigma = 0)
        cfg_det = PTRMConfig(num_rollouts=K, supervision_steps=4, noise_scale=0.0, selector="all")
        engine_det = PTRMRolloutEngine(cfg_det)
        res_det = engine_det.run_rollouts(self.model, inputs)

        # In deterministic mode, all K rollouts must produce identical predictions and Q scores
        for k in range(1, K):
            self.assertTrue(torch.equal(res_det.all_preds[0, 0], res_det.all_preds[0, k]))
            self.assertAlmostEqual(res_det.all_q_scores[0, 0].item(), res_det.all_q_scores[0, k].item(), places=5)

        # Test stochastic mode (sigma > 0)
        cfg_stoch = PTRMConfig(num_rollouts=K, supervision_steps=4, noise_scale=0.5, selector="all", seed=123)
        engine_stoch = PTRMRolloutEngine(cfg_stoch)
        res_stoch = engine_stoch.run_rollouts(self.model, inputs)

        # In stochastic mode with noise, different rollouts explore different trajectories
        unique_q_scores = set(np.round(res_stoch.all_q_scores[0].cpu().numpy(), 4))
        self.assertGreater(len(unique_q_scores), 1)

    def test_best_q_selection_logic(self):
        B = 2
        K = 4
        L = 10

        # Construct dummy results with explicit Q scores
        mock_preds = torch.randint(0, 10, (B, K, L))
        # Batch 0: rollout 2 has highest Q
        # Batch 1: rollout 1 has highest Q
        mock_q = torch.tensor([
            [-2.0, 1.5, 8.4, 0.1],
            [3.1, 7.9, -1.2, 4.0],
        ])

        best_q_indices = torch.argmax(mock_q, dim=-1)
        best_q_pred = mock_preds[torch.arange(B), best_q_indices]

        self.assertEqual(best_q_indices[0].item(), 2)
        self.assertEqual(best_q_indices[1].item(), 1)
        self.assertTrue(torch.equal(best_q_pred[0], mock_preds[0, 2]))
        self.assertTrue(torch.equal(best_q_pred[1], mock_preds[1, 1]))

    def test_mode_and_pass_at_k(self):
        B = 1
        K = 5
        L = 4

        # Rollouts: 3 copies of [1, 2, 3, 4], 2 copies of [5, 6, 7, 8]
        target = torch.tensor([[1, 2, 3, 4]])
        all_preds = torch.tensor([
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [1, 2, 3, 4],
            ]
        ])
        all_q = torch.tensor([[1.0, 2.0, 0.5, 3.0, 0.8]])

        res = PTRMRolloutResult(
            best_q_pred=all_preds[0, 3].unsqueeze(0),
            best_q_score=torch.tensor([3.0]),
            best_q_indices=torch.tensor([3]),
            mode_pred=all_preds[0, 0].unsqueeze(0),
            all_preds=all_preds,
            all_q_scores=all_q,
        )

        pass_rate, is_pass = res.compute_pass_at_k(target)
        mode_acc, is_mode = res.compute_mode_accuracy(target)

        self.assertEqual(pass_rate, 1.0)
        self.assertTrue(is_pass[0].item())
        self.assertEqual(mode_acc, 1.0)
        self.assertTrue(is_mode[0].item())


class TestTRMMLPArchitecture(unittest.TestCase):
    """Test TRM-MLP variant with PTRM."""

    def test_mlp_rollout(self):
        config = TRMConfig(
            hidden_size=64,
            num_heads=4,
            expansion=2.0,
            num_layers=1,
            n_latent_cycles=2,
            n_deep_cycles=2,
            halt_max_steps=2,
            puzzle_emb_dim=64,
            puzzle_emb_len=4,
            forward_dtype="float32",
            mlp_t=True,  # TRM-MLP mode
        )
        model = TRM(config, num_puzzle_ids=1)
        model.eval()

        inputs = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
        cfg = PTRMConfig(num_rollouts=4, supervision_steps=2, noise_scale=0.3, selector="both")
        engine = PTRMRolloutEngine(cfg)

        res = engine.run_rollouts(model, inputs)
        self.assertEqual(res.best_q_pred.shape, (2, SEQ_LEN))
        self.assertEqual(res.mode_pred.shape, (2, SEQ_LEN))


class TestARCAugmentationPipeline(unittest.TestCase):
    """Test ARC data augmentation and inverse transformations."""

    def test_dihedral_and_color_inversion(self):
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.uint8)

        pairs = [(grid, grid)]

        for trans_id in range(8):
            augmented, t_id, perm = augment_arc_puzzle(
                pairs,
                do_color_perm=True,
                do_dihedral=True,
                do_translation=False,
            )
            a_inp, a_out = augmented[0]
            inv_grid = inverse_augment_grid(a_out, t_id, perm)
            self.assertTrue(np.array_equal(grid, inv_grid), f"Failed for transform {t_id}")


class TestLangevinGuidance(unittest.TestCase):
    """Test Appendix C Langevin guidance module."""

    def test_langevin_step(self):
        config = TRMConfig(
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            puzzle_emb_dim=32,
            puzzle_emb_len=2,
            forward_dtype="float32",
        )
        model = TRM(config, num_puzzle_ids=1)
        z = torch.randn(2, 20, 32)
        y = torch.randn(2, 20, 32)

        sampler = LangevinGuidanceSampler(eta=0.01, num_steps=2)
        new_z = sampler.step(model.inner, z, y)

        self.assertEqual(new_z.shape, z.shape)
        self.assertFalse(torch.equal(new_z, z))


if __name__ == "__main__":
    unittest.main()
