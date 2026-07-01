"""
Stage-2 TACA (Text-Aware Cross Attention) tests.

Covers:
- TACAProjection output shape, zero-initialization, and gradient flow
  with the new pre-extracted slice interface.
- NCHAttnProcessor2_0 legacy full-matrix path (store_attn_weights)
  numerical equivalence with SDPA and weight caching.
- NCHAttnProcessor2_0 chunked-logsumexp TACA side path: numerical
  equivalence with the full-matrix slice, and main-path (SDPA)
  output unchanged when extraction is enabled.
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

        B, S_img = 2, 20
        slices = [torch.randn(B, num_heads, S_img, n_text) for _ in range(num_layers)]

        a_tex = taca(slices)

        self.assertEqual(a_tex.shape, (B, S_img, out_dim))
        # zero-initialized projection => a_tex is exactly zero
        self.assertEqual(a_tex.abs().max().item(), 0.0)

        a_tex.sum().backward()
        self.assertIsNotNone(taca.proj.weight.grad)
        # bias=False by default
        self.assertIsNone(taca.proj.bias)

    def test_wrong_num_slices_raises(self):
        taca = TACAProjection(num_layers=3, num_heads=2, num_text_tokens=1, out_dim=4)
        with self.assertRaises(ValueError):
            taca([torch.randn(1, 2, 5, 1)])


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
        """Legacy full-matrix path still matches SDPA (SparseAttnProcessor compat)."""
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
        """Legacy path caches full [B,H,S,S] weights."""
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
        self.assertEqual(w.shape, (2, 4, 26, 26))
        row_sums = w.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    # ---- New chunked-logsumexp TACA side-path tests ----

    def test_taca_slice_matches_full_matrix(self):
        """Chunked logsumexp slice == full-matrix image->text extraction."""
        torch.manual_seed(3)
        attn = self._make_attn().eval()
        hidden = torch.randn(2, 16, 32)
        enc = torch.randn(2, 10, 32)
        text_indices = [3]
        text_seq_len = 10

        with torch.no_grad():
            # Full-matrix reference
            attn.store_attn_weights = True
            attn.taca_config = None
            attn(hidden_states=hidden, encoder_hidden_states=enc)
            full_w = attn.last_attn_weights  # [B, H, S, S]
            # image->text: image rows x text cols
            ref_slice = full_w[:, :, text_seq_len:, text_indices]  # [B,H,S_img,n_text]

            # Chunked side path
            attn.store_attn_weights = False
            attn.taca_config = {
                "text_seq_len": text_seq_len,
                "text_token_indices": text_indices,
                "query_chunk_size": 4,  # small chunk to exercise looping
                "use_checkpoint": False,
            }
            attn.taca_slice = None
            attn(hidden_states=hidden, encoder_hidden_states=enc)
            taca_slice = attn.taca_slice  # [B, H, S_img, n_text]

        self.assertIsNotNone(taca_slice)
        self.assertEqual(taca_slice.shape, ref_slice.shape)
        self.assertTrue(
            torch.allclose(taca_slice, ref_slice, atol=1e-5),
            msg=f"slice mismatch max={(taca_slice-ref_slice).abs().max()}",
        )

    def test_taca_text_only_normalization_matches_cross_attention(self):
        """Paper-style TACA normalizes image queries over text keys only."""
        torch.manual_seed(8)
        attn = self._make_attn().eval()
        hidden = torch.randn(2, 16, 32)
        enc = torch.randn(2, 10, 32)
        text_indices = [3]
        text_seq_len = 10

        with torch.no_grad():
            query = attn.to_q(hidden)
            key = attn.to_k(hidden)
            encoder_query = attn.add_q_proj(enc)
            encoder_key = attn.add_k_proj(enc)

            batch_size = hidden.shape[0]
            head_dim = key.shape[-1] // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()

            q_img = query[:, :, text_seq_len:, :]
            k_text = key[:, :, :text_seq_len, :]
            scores = torch.matmul(q_img.float(), k_text.float().transpose(-1, -2))
            scores = scores * attn.scale
            ref_slice = torch.softmax(scores, dim=-1).index_select(
                -1, torch.tensor(text_indices)
            ).to(query.dtype)

            attn.store_attn_weights = False
            attn.taca_config = {
                "text_seq_len": text_seq_len,
                "text_token_indices": text_indices,
                "query_chunk_size": 4,
                "use_checkpoint": False,
                "text_only_normalization": True,
            }
            attn.taca_slice = None
            attn(hidden_states=hidden, encoder_hidden_states=enc)
            taca_slice = attn.taca_slice

        self.assertIsNotNone(taca_slice)
        self.assertEqual(taca_slice.shape, ref_slice.shape)
        self.assertTrue(
            torch.allclose(taca_slice, ref_slice, atol=1e-5),
            msg=f"slice mismatch max={(taca_slice-ref_slice).abs().max()}",
        )

    def test_main_path_sdpa_unchanged_with_taca(self):
        """SDPA output is identical whether or not taca_config is set."""
        torch.manual_seed(4)
        attn = self._make_attn().eval()
        hidden = torch.randn(2, 16, 32)
        enc = torch.randn(2, 10, 32)

        with torch.no_grad():
            attn.taca_config = None
            h_ref, e_ref = attn(hidden_states=hidden, encoder_hidden_states=enc)

            attn.taca_config = {
                "text_seq_len": 10,
                "text_token_indices": [3],
                "query_chunk_size": 8,
                "use_checkpoint": False,
            }
            h_taca, e_taca = attn(hidden_states=hidden, encoder_hidden_states=enc)

        self.assertTrue(torch.allclose(h_ref, h_taca, atol=1e-6),
                        msg=f"hidden changed max={(h_ref-h_taca).abs().max()}")
        self.assertTrue(torch.allclose(e_ref, e_taca, atol=1e-6),
                        msg=f"encoder changed max={(e_ref-e_taca).abs().max()}")

    def test_taca_slice_shape_and_nonneg(self):
        """TACA slice has correct shape and is valid probability [0,1]."""
        torch.manual_seed(5)
        attn = self._make_attn().eval()
        hidden = torch.randn(1, 16, 32)
        enc = torch.randn(1, 10, 32)

        with torch.no_grad():
            attn.taca_config = {
                "text_seq_len": 10,
                "text_token_indices": [2, 5],  # 2 text tokens
                "query_chunk_size": 0,  # full (no chunking)
                "use_checkpoint": False,
            }
            attn(hidden_states=hidden, encoder_hidden_states=enc)

        s = attn.taca_slice
        self.assertEqual(s.shape, (1, 4, 16, 2))  # [B, H, S_img, n_text]
        self.assertTrue((s >= 0).all())
        self.assertTrue((s <= 1).all())


class DiTExtractATexTest(unittest.TestCase):
    def _make_model(self):
        return NCHTransformer2DModel(
            patch_size=1,
            in_channels=32,
            out_channels=8,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=64,
            guidance_embeds=False,
            axes_dims_rope=[8, 4, 4],
            processor_type="default",
            ffn_ratio=4,
            adaln_dim=32,
            layers_to_retained={"100": {"transformer_blocks": [0, 1]}},
            taca_cfg={
                "enabled": True,
                "taca_layers": [0, 1],
                "text_token_indices": [2],
                "out_dim": 32,
                "query_chunk_size": 64,
            },
        )

    def test_extract_a_tex_shape_and_zero_init(self):
        torch.manual_seed(6)
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
                extract_a_tex=True,
            )

        self.assertIsInstance(out, dict)
        self.assertIn("sample", out)
        self.assertIn("a_tex", out)
        self.assertIn("taca_attention", out)
        self.assertEqual(out["sample"].shape, (1, 2, 32, 32))
        # S_img = (32/2)*(32/2) = 256
        self.assertEqual(out["a_tex"].shape, (1, 256, 32))
        self.assertEqual(out["taca_attention"].shape, (1, 256, 1))
        # zero-init projection => a_tex is zero at init
        self.assertEqual(out["a_tex"].abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(out["taca_attention"]).all())

    def test_baseline_path_unaffected(self):
        torch.manual_seed(7)
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

        self.assertTrue(hasattr(out, "sample"))
        self.assertEqual(out.sample.shape, (1, 2, 32, 32))
        # extraction was reset, processors back to baseline
        for block in model.transformer_blocks:
            self.assertIsNone(block.attn.taca_config)

    def test_taca_sequence_order_restores_row_major_grid(self):
        """TACA attention is collected in DiT block order, but JSD expects
        row-major spatial order."""
        model = self._make_model().eval()
        values = torch.arange(24, dtype=torch.float32).view(1, 24, 1)
        block_order = model.reshape1D(
            values, new_H=3, new_W=4, block_lenth_2D=2, new_img_len=24
        )

        # The block order is interleaved and would show periodic grid artifacts
        # if directly reshaped as HxW.
        self.assertFalse(torch.equal(block_order, values))

        row_major = model.unreshape1D(
            block_order, new_H=3, new_W=4, block_lenth_2D=2, new_img_len=24
        )
        self.assertTrue(torch.equal(row_major, values))


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
                    out["taca_attention"] = torch.zeros(bs, 256, 1)
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
                "taca_attention": "pred.taca_attention",
            },
        })
        op(ctx, {"dit": FakeModel()})

        self.assertEqual(ctx.get("pred.latent_denoised").shape, (1, 2, 32, 32))
        self.assertEqual(ctx.get("pred.a_tex").shape, (1, 256, 32))
        self.assertEqual(ctx.get("pred.taca_attention").shape, (1, 256, 1))


if __name__ == "__main__":
    unittest.main()
