"""Joint Segmentation Decoder (JSD) for TADiSR.

Replaces the standard VAE image decoder with a dual-branch decoder that
jointly produces a super-resolved image and a text segmentation mask.
The image branch reuses the VAE ``Decoder`` structure (and optionally its
weights); the segmentation branch is a symmetric decoder initialized from
scratch. The two branches interact through Cross-Decoder Interaction
Blocks (CDIB) inserted at selected decoder up-sampling levels.

Reference: TADiSR (Text-Aware Real-World Image Super-Resolution,
NeurIPS 2025, arXiv:2506.04641), Section 3.3.
"""
from __future__ import annotations

import math
from importlib import import_module
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from .mj64_vae import Decoder, Normalize, ResnetBlock

SHIFTING_FACTOR = -0.02453034371137619
SCALING_FACTOR = 1 / 0.7610968947410583

_CH_MULT = (1, 2, 4, 4, 4)
_NUM_RES_BLOCKS = (2, 3, 2, 2, 2)


def _locate(target: str):
    module_name, attr_name = target.rsplit(".", 1)
    return getattr(import_module(module_name), attr_name)


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "swish":
        return x * torch.sigmoid(x)
    raise ValueError(f"unknown decoder_activation: {name!r}")


class CDIB(nn.Module):
    """Cross-Decoder Interaction Block."""

    def __init__(self, channels: int, num_groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.channels = int(channels)
        self.resblock_img = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            temb_channels=0,
            dropout=dropout,
        )
        self.resblock_seg = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            temb_channels=0,
            dropout=dropout,
        )
        self.proj_img = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.proj_seg = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.norm_img = Normalize(channels, num_groups=num_groups)
        self.norm_seg = Normalize(channels, num_groups=num_groups)
        self.out_img = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_seg = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale_img = nn.Parameter(torch.zeros(1))
        self.scale_seg = nn.Parameter(torch.zeros(1))

    def forward(
        self, z_img: torch.Tensor, a_seg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_img = self.resblock_img(z_img, None)
        h_seg = self.resblock_seg(a_seg, None)

        fwd_img, exch_img = torch.chunk(self.proj_img(h_img), 2, dim=1)
        fwd_seg, exch_seg = torch.chunk(self.proj_seg(h_seg), 2, dim=1)

        gate_img = fwd_img * torch.sigmoid(exch_seg)
        gate_seg = fwd_seg * torch.sigmoid(exch_img)

        gate_img = self.out_img(F.silu(self.norm_img(gate_img)))
        gate_seg = self.out_seg(F.silu(self.norm_seg(gate_seg)))

        z_img_out = z_img + self.scale_img * gate_img
        a_seg_out = a_seg + self.scale_seg * gate_seg
        return z_img_out, a_seg_out


class JointSegDecoder(nn.Module):
    """Joint Segmentation Decoder.

    The default parameters preserve the original F16C64 JSD behavior. For
    F8C32, pass ``decoder_target='models.vae.npu.f8c32_swin.Decoder'``,
    ``ch_mult=[1, 2, 4, 4]``, ``num_res_blocks=[2, 3, 2, 2]``,
    ``image_post_quant_conv=True``, and the VAE latent shift/scaling
    factors.
    """

    def __init__(
        self,
        z_channels: int,
        a_tex_channels: int,
        resolution: int = 512,
        ch: int = 128,
        ch_mult: Sequence[int] = _CH_MULT,
        num_res_blocks: Sequence[int] = _NUM_RES_BLOCKS,
        in_channels: int = 3,
        seg_channels: int = 1,
        decoder_target: str = "models.vae.npu.mj64_vae.Decoder",
        decoder_activation: str = "silu",
        image_post_quant_conv: bool = False,
        image_embed_dim: Optional[int] = None,
        image_latent_shift_factor: Optional[float] = None,
        image_latent_scaling_factor: Optional[float] = None,
        image_output_scale: float = 1.0 / SCALING_FACTOR,
        image_output_shift: float = SHIFTING_FACTOR,
        cdib_levels: Optional[Sequence[int]] = None,
        image_checkpoint: Optional[str] = None,
        **decoder_kwargs,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.a_tex_channels = int(a_tex_channels)
        self.in_channels = int(in_channels)
        self.seg_channels = int(seg_channels)
        self.ch = int(ch)
        self.ch_mult = tuple(int(x) for x in ch_mult)
        self.num_res_blocks = tuple(int(x) for x in num_res_blocks)
        if len(self.ch_mult) != len(self.num_res_blocks):
            raise ValueError("ch_mult and num_res_blocks must have the same length")

        self.decoder_activation = str(decoder_activation)
        self.image_latent_shift_factor = image_latent_shift_factor
        self.image_latent_scaling_factor = image_latent_scaling_factor
        self.image_output_scale = float(image_output_scale)
        self.image_output_shift = float(image_output_shift)
        self.gradient_checkpointing = False

        block_in = self.ch * self.ch_mult[-1]
        if block_in % self.a_tex_channels != 0:
            raise ValueError(
                f"a_tex_channels={self.a_tex_channels} must evenly divide "
                f"decoder block_in={block_in} (ch*ch_mult[-1])"
            )

        decoder_cls = _locate(decoder_target)
        decoder_common = dict(decoder_kwargs)
        decoder_common.update(
            {
                "resolution": resolution,
                "ch": ch,
                "ch_mult": self.ch_mult,
                "num_res_blocks": self.num_res_blocks,
            }
        )

        self.image_post_quant_conv = None
        if image_post_quant_conv:
            embed_dim = int(image_embed_dim if image_embed_dim is not None else z_channels)
            self.image_post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)

        self.image_decoder = decoder_cls(
            z_channels=z_channels,
            in_channels=in_channels,
            out_ch=in_channels,
            **decoder_common,
        )
        self.seg_decoder = decoder_cls(
            z_channels=a_tex_channels,
            in_channels=seg_channels,
            out_ch=seg_channels,
            **decoder_common,
        )

        num_resolutions = len(self.ch_mult)
        if cdib_levels is None:
            cdib_levels = list(range(num_resolutions - 1, 0, -1))
        self.cdib_levels = [int(x) for x in cdib_levels]
        dropout = float(decoder_kwargs.get("dropout", 0.0))
        self.cdib = nn.ModuleDict(
            {
                str(lvl): CDIB(ch * self.ch_mult[lvl], dropout=dropout)
                for lvl in self.cdib_levels
            }
        )

        if image_checkpoint:
            self._load_image_checkpoint(image_checkpoint)

    def gradient_checkpointing_enable(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _ckpt(self, fn: Any, *args: Any) -> Any:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return _torch_checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _load_image_checkpoint(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        prefixed = {}
        for key, value in sd.items():
            key = _strip_module_prefix(str(key))
            if key.startswith("image_decoder."):
                prefixed[key] = value
            elif key.startswith("decoder."):
                prefixed[f"image_decoder.{key[len('decoder.'):]}"] = value
            elif key.startswith("model."):
                prefixed[f"image_decoder.{key[len('model.'):]}"] = value
            elif (
                key.startswith("post_quant_conv.")
                and self.image_post_quant_conv is not None
            ):
                prefixed[
                    f"image_post_quant_conv.{key[len('post_quant_conv.'):]}"
                ] = value
            elif not key.startswith(("encoder.", "quant_conv.")):
                prefixed[f"image_decoder.{key}"] = value

        missing, unexpected = self.load_state_dict(prefixed, strict=False)
        print(
            f"[JointSegDecoder] loaded image checkpoint from {path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    @staticmethod
    def _a_tex_to_spatial(a_tex: torch.Tensor, latent_shape: torch.Size) -> torch.Tensor:
        """Reshape ``a_tex`` [B, S_img, C] to spatial [B, C, H, W]."""
        b, s_img, c = a_tex.shape
        h_lat, w_lat = latent_shape[-2], latent_shape[-1]
        total = h_lat * w_lat
        if total % s_img != 0:
            raise ValueError(
                f"latent grid {h_lat}x{w_lat}={total} not divisible by "
                f"a_tex seq len {s_img}"
            )
        scale = total // s_img
        patch = int(math.isqrt(scale))
        if patch * patch != scale:
            raise ValueError(
                f"patch scale {scale} is not a perfect square; cannot infer "
                f"patch grid for a_tex seq len {s_img}"
            )
        h_patch = h_lat // patch
        w_patch = w_lat // patch
        if h_patch * w_patch != s_img:
            raise ValueError(
                f"inferred patch grid {h_patch}x{w_patch}={h_patch * w_patch} "
                f"!= a_tex seq len {s_img}"
            )
        a_spatial = a_tex.permute(0, 2, 1).reshape(b, c, h_patch, w_patch)
        if (h_patch, w_patch) != (h_lat, w_lat):
            a_spatial = F.interpolate(
                a_spatial, size=(h_lat, w_lat), mode="bilinear", align_corners=False
            )
        return a_spatial.contiguous()

    def _prepare_image_latent(self, latent: torch.Tensor) -> torch.Tensor:
        image_latent = latent
        if self.image_latent_scaling_factor is not None:
            image_latent = image_latent / float(self.image_latent_scaling_factor)
        if self.image_latent_shift_factor is not None:
            image_latent = image_latent + float(self.image_latent_shift_factor)
        if self.image_post_quant_conv is not None:
            image_latent = self.image_post_quant_conv(image_latent)
        return image_latent

    def forward(self, latent: torch.Tensor, a_tex: torch.Tensor) -> dict[str, torch.Tensor]:
        a_spatial = self._a_tex_to_spatial(a_tex, latent.shape)

        h_img = self.image_decoder.conv_in(self._prepare_image_latent(latent))
        h_img = self._ckpt(self.image_decoder.mid.block_1, h_img, None)
        h_img = self._ckpt(self.image_decoder.mid.attn_1, h_img)
        h_img = self._ckpt(self.image_decoder.mid.block_2, h_img, None)

        h_seg = self.seg_decoder.conv_in(a_spatial)
        h_seg = self._ckpt(self.seg_decoder.mid.block_1, h_seg, None)
        h_seg = self._ckpt(self.seg_decoder.mid.attn_1, h_seg)
        h_seg = self._ckpt(self.seg_decoder.mid.block_2, h_seg, None)

        num_res = self.image_decoder.num_resolutions
        for i_level in reversed(range(num_res)):
            n_blocks = self.image_decoder.num_res_blocks[i_level] + 1
            for i_block in range(n_blocks):
                h_img = self._ckpt(
                    self.image_decoder.up[i_level].block[i_block], h_img, None
                )
                if len(self.image_decoder.up[i_level].attn) > 0:
                    h_img = self._ckpt(
                        self.image_decoder.up[i_level].attn[i_block], h_img
                    )
                h_seg = self._ckpt(
                    self.seg_decoder.up[i_level].block[i_block], h_seg, None
                )
                if len(self.seg_decoder.up[i_level].attn) > 0:
                    h_seg = self._ckpt(
                        self.seg_decoder.up[i_level].attn[i_block], h_seg
                    )

            key = str(i_level)
            if key in self.cdib:
                h_img, h_seg = self._ckpt(self.cdib[key], h_img, h_seg)

            if i_level != 0:
                h_img = self.image_decoder.up[i_level].upsample(h_img)
                h_seg = self.seg_decoder.up[i_level].upsample(h_seg)

        h_img = self.image_decoder.norm_out(h_img)
        h_img = _activation(self.decoder_activation, h_img)
        h_img = self.image_decoder.conv_out(h_img)
        recon = self.image_output_scale * h_img + self.image_output_shift

        h_seg = self.seg_decoder.norm_out(h_seg)
        h_seg = _activation(self.decoder_activation, h_seg)
        h_seg = self.seg_decoder.conv_out(h_seg)
        seg = torch.sigmoid(h_seg)

        return {"recon": recon, "seg": seg}

    def get_fsdp_wrap_module_list(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        for dec in (self.image_decoder, self.seg_decoder):
            modules.append(dec)
            for level in dec.up:
                modules.extend(level.block)
                modules.extend(level.attn)
            modules.extend([dec.mid.block_1, dec.mid.attn_1, dec.mid.block_2])
        modules.extend(list(self.cdib.values()))
        return modules
