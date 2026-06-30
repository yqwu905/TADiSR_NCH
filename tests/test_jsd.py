"""
Stage-3 JSD (Joint Segmentation Decoder) tests.

Covers:
- CDIB zero-initialized residual is identity at start, with gradient flow
  to the residual scale parameters.
- CDIB shape preservation and cross-branch interaction.
- JointSegDecoder output shapes (recon RGB + seg mask), seg range [0,1],
  a_tex seq->spatial reshape, and gradient flow to both branches.
- F8C32 VAE wrapper and JSD decoder-target compatibility.
- jsd_decode op glue: reads context, calls component, writes outputs.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from framework.context import TrainContext
from framework.ops.diffusion import JSDDecodeOp
from models.vae.npu.f8c32_swin import VaeDecoder as F8C32Decoder
from models.vae.npu.f8c32_swin import VaeEncoder as F8C32Encoder
from models.vae.npu.jsd import CDIB, JointSegDecoder


class CDIBTest(unittest.TestCase):
    def test_zero_init_is_identity(self):
        torch.manual_seed(0)
        cdib = CDIB(channels=64)
        z = torch.randn(2, 64, 8, 8, requires_grad=True)
        a = torch.randn(2, 64, 8, 8, requires_grad=True)
        z_out, a_out = cdib(z, a)
        self.assertTrue(torch.allclose(z_out, z, atol=1e-6))
        self.assertTrue(torch.allclose(a_out, a, atol=1e-6))

    def test_shape_preserved(self):
        cdib = CDIB(channels=128)
        z = torch.randn(1, 128, 4, 4)
        a = torch.randn(1, 128, 4, 4)
        z_out, a_out = cdib(z, a)
        self.assertEqual(z_out.shape, z.shape)
        self.assertEqual(a_out.shape, a.shape)

    def test_scale_grad_flows(self):
        cdib = CDIB(channels=64)
        z = torch.randn(1, 64, 4, 4)
        a = torch.randn(1, 64, 4, 4)
        z_out, a_out = cdib(z, a)
        (z_out.sum() + a_out.sum()).backward()
        self.assertIsNotNone(cdib.scale_img.grad)
        self.assertIsNotNone(cdib.scale_seg.grad)


class JointSegDecoderTest(unittest.TestCase):
    def _make(self, resolution=128, a_tex_channels=64, cdib_levels=None):
        return JointSegDecoder(
            z_channels=64,
            a_tex_channels=a_tex_channels,
            resolution=resolution,
            ch=128,
            in_channels=3,
            seg_channels=1,
            cdib_levels=cdib_levels,
        )

    def test_output_shapes_and_seg_range(self):
        torch.manual_seed(0)
        jsd = self._make(resolution=128)
        latent = torch.randn(1, 64, 8, 8)
        a_tex = torch.randn(1, 16, 64)
        out = jsd(latent, a_tex)
        self.assertEqual(out["recon"].shape, (1, 3, 128, 128))
        self.assertEqual(out["seg"].shape, (1, 1, 128, 128))
        self.assertGreaterEqual(out["seg"].min().item(), 0.0)
        self.assertLessEqual(out["seg"].max().item(), 1.0)

    def test_a_tex_channels_diverging_from_z(self):
        jsd = self._make(resolution=128, a_tex_channels=128)
        latent = torch.randn(1, 64, 8, 8)
        a_tex = torch.randn(1, 16, 128)
        out = jsd(latent, a_tex)
        self.assertEqual(out["recon"].shape, (1, 3, 128, 128))
        self.assertEqual(out["seg"].shape, (1, 1, 128, 128))

    def test_grad_flows_to_both_branches(self):
        jsd = self._make(resolution=128)
        latent = torch.randn(1, 64, 8, 8, requires_grad=True)
        a_tex = torch.randn(1, 16, 64, requires_grad=True)
        out = jsd(latent, a_tex)
        (out["recon"].sum() + out["seg"].sum()).backward()
        self.assertIsNotNone(latent.grad)
        self.assertIsNotNone(a_tex.grad)
        img_param = next(jsd.image_decoder.parameters())
        seg_param = next(jsd.seg_decoder.parameters())
        self.assertIsNotNone(img_param.grad)
        self.assertIsNotNone(seg_param.grad)
        for key, mod in jsd.cdib.items():
            self.assertIsNotNone(mod.scale_img.grad, msg=f"cdib[{key}].scale_img")
            self.assertIsNotNone(mod.scale_seg.grad, msg=f"cdib[{key}].scale_seg")

    def test_bad_a_tex_seq_len_raises(self):
        jsd = self._make(resolution=128)
        latent = torch.randn(1, 64, 8, 8)
        bad = torch.randn(1, 10, 64)
        with self.assertRaises(ValueError):
            jsd(latent, bad)

    def test_cdib_levels_default_excludes_level0(self):
        jsd = self._make(resolution=128)
        self.assertEqual(jsd.cdib_levels, [4, 3, 2, 1])

    def test_f8c32_decoder_layout(self):
        torch.manual_seed(2)
        jsd = JointSegDecoder(
            z_channels=32,
            a_tex_channels=32,
            resolution=64,
            ch=32,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=[1, 1, 1, 1],
            decoder_target="models.vae.npu.f8c32_swin.Decoder",
            decoder_activation="swish",
            image_post_quant_conv=True,
            image_embed_dim=32,
            image_latent_shift_factor=0.07050679,
            image_latent_scaling_factor=0.2517327,
            image_output_scale=1.0,
            image_output_shift=0.0,
            cdib_levels=[3, 2, 1],
            swin_depths=[1],
            swin_num_heads=[4],
            swin_window_size=4,
        )
        latent = torch.randn(1, 32, 8, 8)
        a_tex = torch.randn(1, 16, 32)
        out = jsd(latent, a_tex)
        self.assertEqual(out["recon"].shape, (1, 3, 64, 64))
        self.assertEqual(out["seg"].shape, (1, 1, 64, 64))
        self.assertEqual(jsd.cdib_levels, [3, 2, 1])

    def test_image_branch_identity_without_cdib(self):
        # With CDIB at no levels, image branch == plain VAE decode path.
        from models.vae.npu.f16c64 import SHIFTING_FACTOR, SCALING_FACTOR
        from models.vae.npu.mj64_vae import Decoder

        torch.manual_seed(1)
        jsd = self._make(resolution=128, cdib_levels=[])
        torch.manual_seed(1)
        ref = Decoder(z_channels=64, resolution=128)
        ref.load_state_dict(jsd.image_decoder.state_dict())
        latent = torch.randn(1, 64, 8, 8)
        a_tex = torch.randn(1, 16, 64)
        jsd_recon = jsd(latent, a_tex)["recon"]
        expected = 1.0 / SCALING_FACTOR * ref(latent) + SHIFTING_FACTOR
        self.assertTrue(torch.allclose(jsd_recon, expected, atol=1e-6))


class JSDDecodeOpTest(unittest.TestCase):
    def test_op_writes_recon_and_seg(self):
        torch.manual_seed(0)
        jsd = JointSegDecoder(
            z_channels=64, a_tex_channels=64, resolution=128, ch=128
        )
        jsd.eval()

        ctx = TrainContext(global_step=0, batch={})
        latent = torch.randn(1, 64, 8, 8)
        a_tex = torch.randn(1, 16, 64)
        ctx.set("pred.latent_denoised", latent)
        ctx.set("pred.a_tex", a_tex)

        op = JSDDecodeOp(
            {
                "component": "jsd",
                "inputs": {
                    "latent": "pred.latent_denoised",
                    "a_tex": "pred.a_tex",
                },
                "outputs": {"recon": "pred.rgb", "seg": "pred.seg"},
            }
        )
        op(ctx, {"jsd": jsd})

        self.assertEqual(ctx.get("pred.rgb").shape, (1, 3, 128, 128))
        self.assertEqual(ctx.get("pred.seg").shape, (1, 1, 128, 128))

    def test_op_missing_component_raises(self):
        with self.assertRaises(ValueError):
            JSDDecodeOp({"inputs": {}})

    def test_op_missing_a_tex_key_raises(self):
        op = JSDDecodeOp({"component": "jsd", "inputs": {"latent": "x"}})
        ctx = TrainContext(global_step=0, batch={})
        ctx.set("x", torch.randn(1, 64, 8, 8))
        with self.assertRaises(KeyError):
            op(ctx, {"jsd": nn.Identity()})


class F8C32VaeWrapperTest(unittest.TestCase):
    def _params(self):
        return {
            "embed_dim": 32,
            "ddconfig": {
                "double_z": True,
                "z_channels": 32,
                "resolution": 64,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 32,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": [1, 1, 1, 1],
                "attn_resolutions": [],
                "dropout": 0.0,
                "swin_depths": [1],
                "swin_num_heads": [4],
                "swin_window_size": 4,
            },
        }

    def test_encoder_decoder_shapes(self):
        torch.manual_seed(3)
        params = self._params()
        encoder = F8C32Encoder(**params)
        decoder = F8C32Decoder(**params)
        x = torch.randn(1, 3, 64, 64)
        latent = encoder(x)["latent"]
        self.assertEqual(latent.shape, (1, 32, 8, 8))
        recon = decoder(latent)["recon"]
        self.assertEqual(recon.shape, (1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
