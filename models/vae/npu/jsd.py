"""Joint Segmentation Decoder (JSD) for TADiSR.

Replaces the standard VAE image decoder with a dual-branch decoder that
jointly produces a super-resolved image and a text segmentation mask.
The image branch reuses the VAE ``Decoder`` structure (and optionally its
weights); the segmentation branch is a symmetric decoder initialized from
scratch.  The two branches interact through Cross-Decoder Interaction
Blocks (CDIB) inserted at selected decoder up-sampling levels.

Reference: TADiSR (Text-Aware Real-World Image Super-Resolution,
NeurIPS 2025, arXiv:2506.04641), Section 3.3.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from .mj64_vae import Decoder, Normalize, ResnetBlock, nonlinearity

SHIFTING_FACTOR = -0.02453034371137619
SCALING_FACTOR = 1 / 0.7610968947410583

_CH_MULT = (1, 2, 4, 4, 4)
_NUM_RES_BLOCKS = (2, 3, 2, 2, 2)


class CDIB(nn.Module):
    """Cross-Decoder Interaction Block.

    Two-branch interaction module inserted between the image and
    segmentation decoder streams.  Each branch passes its feature through
    a ResBlock, then a 1x1 conv that doubles the channels and is split
    into a within-branch half and a cross-branch half.  The cross-branch
    half is sigmoid-gated and multiplied (Hadamard) with the within-branch
    half, then processed by GroupNorm + SiLU + 1x1 conv.  A learnable
    residual scale initialized to zero keeps the block identity at start
    so training begins without disturbing the pretrained image branch.
    """

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

    Parameters
    ----------
    z_channels : int
        Latent channels of the image branch input (VAE latent dim).
    a_tex_channels : int
        Channel dimension of the text-aware attention feature ``a_tex``
        (TACA projection ``out_dim``).  Must evenly divide the decoder
        block-in width (``ch * ch_mult[-1]``) so the seg-branch
        ``conv_in`` shortcut stays valid.
    resolution : int
        Target image resolution forwarded to ``Decoder``.
    ch : int
        Base channel width of the decoder.
    in_channels : int
        Output channels of the image branch (RGB, default 3).
    seg_channels : int
        Output channels of the segmentation branch (1 for binary mask).
    cdib_levels : sequence of int, optional
        Decoder up-sampling level indices (0 = highest res,
        ``num_resolutions-1`` = lowest res) at which a CDIB is inserted.
        Default ``[num_resolutions-1, ..., 1]`` (all but the final
        output level).
    image_checkpoint : str or None
        Optional path to a VAE checkpoint loaded into the image branch
        only (non-strict, so seg-branch keys are ignored).
    **decoder_kwargs
        Forwarded to ``Decoder`` (``attn_resolutions``, ``dropout``, ...).
    """

    def __init__(
        self,
        z_channels: int,
        a_tex_channels: int,
        resolution: int = 512,
        ch: int = 128,
        in_channels: int = 3,
        seg_channels: int = 1,
        cdib_levels: Optional[Sequence[int]] = None,
        image_checkpoint: Optional[str] = None,
        **decoder_kwargs,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.a_tex_channels = int(a_tex_channels)
        self.in_channels = int(in_channels)
        self.seg_channels = int(seg_channels)
        self.gradient_checkpointing = False

        block_in = ch * _CH_MULT[-1]
        if block_in % self.a_tex_channels != 0:
            raise ValueError(
                f"a_tex_channels={self.a_tex_channels} must evenly divide "
                f"decoder block_in={block_in} (ch*ch_mult[-1])"
            )

        self.image_decoder = Decoder(
            z_channels=z_channels,
            resolution=resolution,
            ch=ch,
            in_channels=in_channels,
            **decoder_kwargs,
        )
        self.seg_decoder = Decoder(
            z_channels=a_tex_channels,
            resolution=resolution,
            ch=ch,
            in_channels=seg_channels,
            **decoder_kwargs,
        )

        num_resolutions = len(_CH_MULT)
        if cdib_levels is None:
            cdib_levels = list(range(num_resolutions - 1, 0, -1))
        self.cdib_levels = [int(x) for x in cdib_levels]
        dropout = float(decoder_kwargs.get("dropout", 0.0))
        self.cdib = nn.ModuleDict(
            {str(lvl): CDIB(ch * _CH_MULT[lvl], dropout=dropout) for lvl in self.cdib_levels}
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
        prefixed = {f"image_decoder.{k}": v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(prefixed, strict=False)
        print(
            f"[JointSegDecoder] loaded image checkpoint from {path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    @staticmethod
    def _a_tex_to_spatial(a_tex: torch.Tensor, latent_shape: torch.Size) -> torch.Tensor:
        """Reshape ``a_tex`` ``[B, S_img, C]`` to spatial ``[B, C, H, W]``
        matching the latent grid, interpolating when the DiT patch grid
        is coarser than the latent grid."""
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

    def forward(self, latent: torch.Tensor, a_tex: torch.Tensor) -> dict[str, torch.Tensor]:
        a_spatial = self._a_tex_to_spatial(a_tex, latent.shape)

        h_img = self.image_decoder.conv_in(latent)
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
                h_img = self._ckpt(self.image_decoder.up[i_level].block[i_block], h_img, None)
                if len(self.image_decoder.up[i_level].attn) > 0:
                    h_img = self._ckpt(self.image_decoder.up[i_level].attn[i_block], h_img)
                h_seg = self._ckpt(self.seg_decoder.up[i_level].block[i_block], h_seg, None)
                if len(self.seg_decoder.up[i_level].attn) > 0:
                    h_seg = self._ckpt(self.seg_decoder.up[i_level].attn[i_block], h_seg)

            key = str(i_level)
            if key in self.cdib:
                h_img, h_seg = self._ckpt(self.cdib[key], h_img, h_seg)

            if i_level != 0:
                h_img = self.image_decoder.up[i_level].upsample(h_img)
                h_seg = self.seg_decoder.up[i_level].upsample(h_seg)

        h_img = self.image_decoder.norm_out(h_img)
        h_img = nonlinearity(h_img)
        h_img = self.image_decoder.conv_out(h_img)
        recon = 1.0 / SCALING_FACTOR * h_img + SHIFTING_FACTOR

        h_seg = self.seg_decoder.norm_out(h_seg)
        h_seg = nonlinearity(h_seg)
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
