"""
Stage-2 TACA (Text-Aware Cross Attention) tests.

Covers:
- TACAProjection output shape, zero-initialization, and gradient flow.
- NCHAttnProcessor2_0 manual attention numerical equivalence with SDPA
  and attention-weight caching when ``store_attn_weights`` is enabled.
- NCHTransformer2DModel end-to-end ``extract_a_tex`` produces a correctly
  shaped, zero-initialized ``a_tex`` without breaking the baseline path.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from framework.context import TrainContext
from framework.ops.diffusion import NCHMMDiTSROp
from models.dit.nch.ldm.taca import TACAProjection
from models.dit.nch.ldm.attention_processor_v3_split import (
    Attention,
    NCHAttnProcessor2_0,
)
from models.dit.nch.ldm.transformer_nch_v3_split import NCHTransformer2DModel


class TACAProjectionTest(unittest.TestCase):
    def test_shape_zero_init_and_grad(self):
        torch.manual_seed(0)
        num_layers, num_heads, n_text, out_dim = 2, 4, 1, 32
        taca = TACAProjection(num_layers, num_heads, n_text, out_dim)

        B, H, S_text, S_img = 2, 4, 10, 20
        S = S_text + S_img
        weights = [w.softmax(dim=-1) for w in (
            torch.randn(B, H, S, S), torch.randn(B, H, S, S)
        )]

        a_tex = taca(weights, [3], S_text)

        self.assertEqual(a_tex.shape, (B, S_img, out_dim))
        # zero-initialized projection => a_tex is exactly zero
        self.assertEqual(a_tex.abs().max().item(), 0.0)

        a_tex.sum().backward()
        self.assertIsNotNone(taca.proj.weight.grad)
        # bias=False by default
        self.assertIsNone(taca.proj.bias)

    def test_wrong_num_layers_raises(self):
        taca = TACAProjection(num_layers=3, num_heads=2, num_text_tokens=1, out_dim=4)
        with self.assertRaises(ValueError):
            taca([torch.randn(1, 2, 5, 5)], [0], 2)


class NCHAttnProcessorTest(unittest.TestCase):
    def _make_attn(self):
        return Attention(
            query_dim=32,
            heads=4,
            dim_head=8,
            added_kv_proj_dim=32,
            out_dim=32,
            context_pre_only=False,
            bias=True,
            processor=NCHAttnProcessor2_0(),
            qk_norm="rms_norm",
        )

    def test_manual_matches_sdpa(self):
        torch.manual_seed(1)
        attn = self._make_attn().eval()
        hidden = torch.randn(2, 16, 32)
        enc = torch.randn(2, 10, 32)

        with torch.no_grad():
            attn.store_attn_weights = False
            h_sdpa, e_sdpa = attn(hidden_states=hidden, encoder_hidden_states=enc)

            attn.store_attn_weights = True
            h_manual, e_manual = attn(hidden_states=hidden, encoder_hidden_states=enc)

        self.assertTrue(torch.allclose(h_sdpa, h_manual, atol=1e-4),
                        msg=f"hidden mismatch max={(h_sdpa-h_manual).abs().max()}")
        self.assertTrue(torch.allclose(e_sdpa, e_manual, atol=1e-4),
                        msg=f"encoder mismatch max={(e_sdpa-e_manual).abs().max()}")

    def test_stores_weights_and_softmax_rows(self):
        torch.manual_seed(2)
        attn = self._make_attn().eval()
        hidden = torch.randn(2, 16, 32)
        enc = torch.randn(2, 10, 32)

        with torch.no_grad():
            attn.store_attn_weights = False
            attn(hidden_states=hidden, encoder_hidden_states=enc)
            self.assertIsNone(attn.last_attn_weights)

            attn.store_attn_weights = True
            attn(hidden_states=hidden, encoder_hidden_states=enc)

        w = attn.last_attn_weights
        self.assertIsNotNone(w)
        # [B, H, S_text+S_img, S_text+S_img]
        self.assertEqual(w.shape, (2, 4, 26, 26))
        # each row sums to 1 (softmax)
        row_sums = w.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))


class DiTExtractATexTest(unittest.TestCase):
    def _make_model(self):
        return NCHTransformer2DModel(
            patch_size=1,
            in_channels=32,        # input C=8 * patch^2 (patchify 2x2)
            out_channels=8,        # proj_out dim = ps^2*out_ch = 8; unrearrange /4 => 2ch
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=64,
            guidance_embeds=False,
            axes_dims_rope=[8, 4, 4],   # sum == attention_head_dim
            processor_type="default",
            ffn_ratio=4,
            adaln_dim=32,
            layers_to_retained={"100": {"transformer_blocks": [0, 1]}},
            taca_cfg={
                "enabled": True,
                "taca_layers": [0, 1],
                "text_token_indices": [2],
                "out_dim": 32,
            },
        )

    def test_extract_a_tex_shape_and_zero_init(self):
        torch.manual_seed(3)
        model = self._make_model().eval()
        hidden = torch.randn(1, 8, 32, 32)        # [B, C, H, W]
        enc = torch.randn(1, 10, 32)              # [B, S_text, joint_dim]
        t = torch.tensor([1.0])

        with torch.no_grad():
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                timestep=t,
                enable_skip_level="100",
                extract_a_tex=True,
            )

        self.assertIsInstance(out, dict)
        self.assertIn("sample", out)
        self.assertIn("a_tex", out)
        self.assertEqual(out["sample"].shape, (1, 2, 32, 32))
        # S_img = (32/2)*(32/2) = 256
        self.assertEqual(out["a_tex"].shape, (1, 256, 32))
        # zero-init projection => a_tex is zero at init
        self.assertEqual(out["a_tex"].abs().max().item(), 0.0)

    def test_baseline_path_unaffected(self):
        torch.manual_seed(4)
        model = self._make_model().eval()
        hidden = torch.randn(1, 8, 32, 32)
        enc = torch.randn(1, 10, 32)
        t = torch.tensor([1.0])

        with torch.no_grad():
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                timestep=t,
                enable_skip_level="100",
            )

        # default (non-extract) returns Transformer2DModelOutput with .sample
        self.assertTrue(hasattr(out, "sample"))
        self.assertEqual(out.sample.shape, (1, 2, 32, 32))
        # harvesting was reset, processors back to SDPA mode
        for block in model.transformer_blocks:
            self.assertFalse(block.attn.store_attn_weights)


class NCHMMDiTSROpATexTest(unittest.TestCase):
    """Verifies the op glue: extract_a_tex wires the DiT result's a_tex
    into TrainContext alongside the denoised latent."""

    def test_op_writes_a_tex_to_ctx(self):
        class FakeModel(nn.Module):
            def forward(self, hidden_states, encoder_hidden_states,
                        timestep, enable_skip_level=None, extract_a_tex=False):
                bs = hidden_states.shape[0]
                out = {"sample": torch.zeros(bs, 2, 32, 32)}
                if extract_a_tex:
                    out["a_tex"] = torch.zeros(bs, 256, 32)
                return out

        ctx = TrainContext(global_step=0, batch={})
        ctx.set("lq.latent", torch.randn(1, 8, 32, 32))
        ctx.set("enc", torch.randn(1, 10, 32))

        op = NCHMMDiTSROp({
            "component": "dit", "input_type": "lq", "timestep": 1.0,
            "mask_repeat": 2, "enable_skip_level": "100", "denoise": False,
            "extract_a_tex": True,
            "inputs": {"latent": "lq.latent", "encoder_hidden_states": "enc"},
            "outputs": {
                "latent_denoised": "pred.latent_denoised",
                "a_tex": "pred.a_tex",
            },
        })
        op(ctx, {"dit": FakeModel()})

        self.assertEqual(ctx.get("pred.latent_denoised").shape, (1, 2, 32, 32))
        self.assertEqual(ctx.get("pred.a_tex").shape, (1, 256, 32))


if __name__ == "__main__":
    unittest.main()
