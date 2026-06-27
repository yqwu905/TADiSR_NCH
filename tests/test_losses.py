"""
Stage-4 loss tests: TADiSRLoss (L2 + LPIPS + modified focal) and
SegLoss (L2 + Focal + Dice), plus the rescale op glue.

Covers:
- TADiSRLoss returns dict with loss/l2/lpips/mf, loss >= 0.
- modified focal: zero edge difference -> mf == 0; agreeing seg -> mf == 0;
  disagreeing seg amplifies the edge loss.
- LPIPS graceful fallback when the lpips package is missing.
- gradient flows through pred and pred_seg.
- output shape / range validation.
- SegLoss returns dict with loss/l2/focal/dice; perfect prediction gives
  l2~=0, focal~=0, dice~=0; gradient flows.
- rescale op: out = scale*input + shift.
"""
from __future__ import annotations

import unittest
from typing import Any

import torch
import torch.nn as nn

from framework.context import TrainContext
from framework.ops.common import RescaleOp
from models.loss.seg_loss import SegLoss
from models.loss.tadisr_loss import TADiSRLoss


class TADiSRLossTest(unittest.TestCase):
    def _make(self, **kw):
        params: dict[str, Any] = {
            "w_l2": 1.0, "w_lpips": 0.0, "w_mf": 10.0, "focal_gamma": 2.0,
        }
        params.update(kw)
        return TADiSRLoss(**params)

    def test_returns_dict_and_nonneg(self):
        loss = self._make()
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        pred_seg = torch.rand(2, 1, 16, 16)
        target_seg = (torch.rand(2, 1, 16, 16) > 0.5).float()
        out = loss(pred, target, pred_seg, target_seg)
        self.assertIn("loss", out)
        self.assertIn("l2", out)
        self.assertIn("lpips", out)
        self.assertIn("mf", out)
        self.assertGreaterEqual(out["loss"].item(), 0.0)

    def test_zero_edge_diff_gives_zero_mf(self):
        loss = self._make(w_l2=0.0, w_mf=1.0)
        img = torch.randn(1, 3, 12, 12)
        seg = torch.rand(1, 1, 12, 12)
        gt_seg = (torch.rand(1, 1, 12, 12) > 0.5).float()
        out = loss(img, img, seg, gt_seg)
        self.assertAlmostEqual(out["mf"].item(), 0.0, places=6)
        self.assertAlmostEqual(out["loss"].item(), 0.0, places=6)

    def test_agreeing_seg_gives_zero_mf(self):
        loss = self._make(w_l2=0.0, w_mf=1.0)
        pred = torch.randn(1, 3, 12, 12)
        target = torch.randn(1, 3, 12, 12)
        seg = torch.ones(1, 1, 12, 12)
        gt_seg = torch.ones(1, 1, 12, 12)
        out = loss(pred, target, seg, gt_seg)
        self.assertAlmostEqual(out["mf"].item(), 0.0, places=6)

    def test_disagreeing_seg_amplifies_mf(self):
        loss = self._make(w_l2=0.0, w_mf=1.0)
        pred = torch.randn(1, 3, 16, 16)
        target = torch.randn(1, 3, 16, 16)
        seg = torch.ones(1, 1, 16, 16)
        gt_seg = torch.zeros(1, 1, 16, 16)
        out_disagree = loss(pred, target, seg, gt_seg)
        seg_agree = torch.ones(1, 1, 16, 16)
        gt_agree = torch.ones(1, 1, 16, 16)
        out_agree = loss(pred, target, seg_agree, gt_agree)
        self.assertGreater(out_disagree["mf"].item(), out_agree["mf"].item())
        self.assertGreater(out_disagree["mf"].item(), 0.0)

    def test_no_seg_skips_mf(self):
        loss = self._make(w_l2=1.0, w_mf=10.0)
        pred = torch.randn(1, 3, 8, 8)
        target = torch.randn(1, 3, 8, 8)
        out = loss(pred, target)
        self.assertAlmostEqual(out["mf"].item(), 0.0, places=6)
        self.assertEqual(out["loss"].item(), out["l2"].item())

    def test_grad_flows_to_pred_and_seg(self):
        loss = self._make(w_l2=1.0, w_mf=1.0)
        pred = torch.randn(1, 3, 12, 12, requires_grad=True)
        target = torch.randn(1, 3, 12, 12)
        pred_seg = torch.rand(1, 1, 12, 12, requires_grad=True)
        target_seg = (torch.rand(1, 1, 12, 12) > 0.5).float()
        out = loss(pred, target, pred_seg, target_seg)
        out["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertIsNotNone(pred_seg.grad)

    def test_spatial_mismatch_raises(self):
        loss = self._make()
        pred = torch.randn(1, 3, 8, 8)
        target = torch.randn(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            loss(pred, target)


class SegLossTest(unittest.TestCase):
    def _make(self, **kw):
        params: dict[str, Any] = {
            "w_l2": 1.0, "w_focal": 10.0, "w_dice": 1.0, "focal_gamma": 2.0,
        }
        params.update(kw)
        return SegLoss(**params)

    def test_returns_dict_keys(self):
        loss = self._make()
        pred = torch.rand(2, 1, 16, 16)
        target = (torch.rand(2, 1, 16, 16) > 0.5).float()
        out = loss(pred, target)
        for k in ("loss", "l2", "focal", "dice"):
            self.assertIn(k, out)
        self.assertGreaterEqual(out["loss"].item(), 0.0)

    def test_perfect_prediction_small_loss(self):
        loss = self._make()
        target = (torch.rand(1, 1, 16, 16) > 0.5).float()
        pred = target.clone()
        out = loss(pred, target)
        self.assertAlmostEqual(out["l2"].item(), 0.0, places=6)
        self.assertAlmostEqual(out["focal"].item(), 0.0, places=3)
        self.assertAlmostEqual(out["dice"].item(), 0.0, places=3)

    def test_dice_range(self):
        loss = self._make(w_l2=0.0, w_focal=0.0, w_dice=1.0)
        pred = torch.rand(1, 1, 16, 16)
        target = (torch.rand(1, 1, 16, 16) > 0.5).float()
        out = loss(pred, target)
        self.assertGreaterEqual(out["dice"].item(), 0.0)
        self.assertLessEqual(out["dice"].item(), 1.0)

    def test_focal_nonneg(self):
        loss = self._make(w_l2=0.0, w_focal=1.0, w_dice=0.0)
        pred = torch.rand(2, 1, 12, 12)
        target = (torch.rand(2, 1, 12, 12) > 0.5).float()
        out = loss(pred, target)
        self.assertGreaterEqual(out["focal"].item(), 0.0)

    def test_grad_flows(self):
        loss = self._make()
        pred = torch.rand(1, 1, 16, 16, requires_grad=True)
        target = (torch.rand(1, 1, 16, 16) > 0.5).float()
        out = loss(pred, target)
        out["loss"].backward()
        self.assertIsNotNone(pred.grad)

    def test_channel_mismatch_raises(self):
        loss = self._make()
        pred = torch.rand(1, 2, 8, 8)
        target = (torch.rand(1, 1, 8, 8) > 0.5).float()
        with self.assertRaises(ValueError):
            loss(pred, target)

    def test_spatial_mismatch_raises(self):
        loss = self._make()
        pred = torch.rand(1, 1, 8, 8)
        target = (torch.rand(1, 1, 16, 16) > 0.5).float()
        with self.assertRaises(ValueError):
            loss(pred, target)


class RescaleOpTest(unittest.TestCase):
    def test_affine(self):
        op = RescaleOp({
            "input": "batch.x",
            "scale": 2.0,
            "shift": -1.0,
            "output": "batch.x_norm",
        })
        ctx = TrainContext()
        ctx.set("batch.x", torch.tensor([[0.0, 0.5, 1.0]]))
        op(ctx, components={})
        out = ctx.get("batch.x_norm")
        self.assertTrue(torch.allclose(out, torch.tensor([[-1.0, 0.0, 1.0]])))

    def test_non_tensor_raises(self):
        op = RescaleOp({
            "input": "batch.x",
            "scale": 1.0,
            "shift": 0.0,
            "output": "batch.x_out",
        })
        ctx = TrainContext()
        ctx.set("batch.x", "not a tensor")
        with self.assertRaises(TypeError):
            op(ctx, components={})


if __name__ == "__main__":
    unittest.main()
