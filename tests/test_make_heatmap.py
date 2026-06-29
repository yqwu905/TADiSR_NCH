"""
Tests for the ``make_heatmap`` op (``MakeHeatmapOp``).

Covers non-square token grids inferred from a 4D ``grid_source`` latent,
the square case, and error paths for shape mismatch / non-divisible
patchify. This guards against the regression where a non-square real FTSR
PNG produced ``S_img != grid_h * grid_w`` and the op's reshape failed.
"""
from __future__ import annotations

import unittest

import torch

from framework.context import TrainContext
from framework.ops.common import MakeHeatmapOp


def _make_op(cfg: dict) -> MakeHeatmapOp:
    return MakeHeatmapOp(cfg)


def _ctx_with(latent: torch.Tensor, a_tex: torch.Tensor) -> TrainContext:
    ctx = TrainContext()
    ctx.set("lq.latent", latent)
    ctx.set("pred.a_tex", a_tex)
    return ctx


class MakeHeatmapOpTest(unittest.TestCase):
    def test_non_square_grid_s192_reproducing_crash(self):
        # VAE latent [1, 64, 32, 24] -> patchify=2 -> grid 16x12 -> S=192
        latent = torch.randn(1, 64, 32, 24)
        a_tex = torch.randn(1, 192, 8)
        ctx = _ctx_with(latent, a_tex)

        op = _make_op({
            "type": "make_heatmap",
            "input": "pred.a_tex",
            "grid_source": "lq.latent",
            "patchify": 2,
            "target_h": 1024,
            "target_w": 1024,
            "reduce": "norm",
            "normalize": True,
            "output": "pred.taca_heatmap",
        })
        op(ctx, components={})

        heat = ctx.get("pred.taca_heatmap")
        self.assertEqual(heat.shape, (1, 1, 1024, 1024))
        self.assertTrue((heat >= 0).all())
        self.assertTrue((heat <= 1).all())

    def test_square_grid_s1024(self):
        # VAE latent [1, 64, 64, 64] -> patchify=2 -> grid 32x32 -> S=1024
        latent = torch.randn(1, 64, 64, 64)
        a_tex = torch.randn(1, 1024, 8)
        ctx = _ctx_with(latent, a_tex)

        op = _make_op({
            "type": "make_heatmap",
            "input": "pred.a_tex",
            "grid_source": "lq.latent",
            "patchify": 2,
            "target_h": 1024,
            "target_w": 1024,
            "reduce": "norm",
            "normalize": True,
            "output": "pred.taca_heatmap",
        })
        op(ctx, components={})

        heat = ctx.get("pred.taca_heatmap")
        self.assertEqual(heat.shape, (1, 1, 1024, 1024))

    def test_token_count_mismatch_raises(self):
        # grid 16x16 = 256, but a_tex has S=192
        latent = torch.randn(1, 64, 32, 32)
        a_tex = torch.randn(1, 192, 8)
        ctx = _ctx_with(latent, a_tex)

        op = _make_op({
            "type": "make_heatmap",
            "input": "pred.a_tex",
            "grid_source": "lq.latent",
            "patchify": 2,
            "output": "pred.taca_heatmap",
        })

        with self.assertRaisesRegex(ValueError, "token count"):
            op(ctx, components={})

    def test_patchify_not_divisible_raises(self):
        # latent 33x32 not divisible by patchify=2
        latent = torch.randn(1, 64, 33, 32)
        a_tex = torch.randn(1, 192, 8)
        ctx = _ctx_with(latent, a_tex)

        op = _make_op({
            "type": "make_heatmap",
            "input": "pred.a_tex",
            "grid_source": "lq.latent",
            "patchify": 2,
            "output": "pred.taca_heatmap",
        })

        with self.assertRaisesRegex(ValueError, "divisible by patchify"):
            op(ctx, components={})

    def test_missing_grid_source_raises(self):
        ctx = _ctx_with(torch.randn(1, 64, 32, 32), torch.randn(1, 256, 8))

        op = _make_op({
            "type": "make_heatmap",
            "input": "pred.a_tex",
            "patchify": 2,
            "output": "pred.taca_heatmap",
        })

        with self.assertRaisesRegex(ValueError, "grid_source"):
            op(ctx, components={})

    def test_reduce_mean_path(self):
        latent = torch.randn(1, 64, 32, 24)
        a_tex = torch.randn(1, 192, 8)
        ctx = _ctx_with(latent, a_tex)

        op = _make_op({
            "type": "make_heatmap",
            "input": "pred.a_tex",
            "grid_source": "lq.latent",
            "patchify": 2,
            "reduce": "mean",
            "normalize": False,
            "output": "pred.taca_heatmap",
        })
        op(ctx, components={})

        heat = ctx.get("pred.taca_heatmap")
        self.assertEqual(heat.shape, (1, 1, 16, 12))


if __name__ == "__main__":
    unittest.main()
