"""Joint Segmentation Decoder (JSD) implementations for TADiSR.

The F16C64 and F8C32 VAE families use different decoder internals, so they
are exposed as separate JSD components instead of one parameter-switched class.
Both implementations keep the TADiSR dual-branch shape: an image decoder, a
segmentation decoder, and Cross-Decoder Interaction Blocks (CDIB) between
matching up-sampling levels.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from . import f8c32_swin
from . import mj64_vae

SHIFTING_FACTOR = -0.02453034371137619
SCALING_FACTOR = 1 / 0.7610968947410583

_F16_CH_MULT = (1, 2, 4, 4, 4)
_F16_NUM_RES_BLOCKS = (2, 3, 2, 2, 2)
_F8_CH_MULT = (1, 2, 4, 4)
_F8_NUM_RES_BLOCKS = (2, 3, 2, 2)


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "swish":
        return x * torch.sigmoid(x)
    raise ValueError(f"unknown decoder_activation: {name!r}")


def _as_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc


def _reject_params(kwargs: dict[str, Any], names: set[str], owner: str) -> None:
    present = sorted(name for name in names if name in kwargs)
    if present:
        joined = ", ".join(present)
        raise TypeError(f"{owner} does not accept shared-JSD switch params: {joined}")


class _CDIBBase(nn.Module):
    """Cross-Decoder Interaction Block shared by concrete VAE layouts."""

    def __init__(
        self,
        channels: int,
        *,
        resblock_cls: type[nn.Module],
        normalize_fn: Any,
        num_groups: int = 32,
        dropout: float = 0.0,
        zero_init_outputs: bool = True,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.resblock_img = resblock_cls(
            in_channels=channels,
            out_channels=channels,
            temb_channels=0,
            dropout=dropout,
        )
        self.resblock_seg = resblock_cls(
            in_channels=channels,
            out_channels=channels,
            temb_channels=0,
            dropout=dropout,
        )
        self.proj_img = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.proj_seg = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.norm_img = normalize_fn(channels, num_groups=num_groups)
        self.norm_seg = normalize_fn(channels, num_groups=num_groups)
        self.out_img = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_seg = nn.Conv2d(channels, channels, kernel_size=1)
        if zero_init_outputs:
            nn.init.zeros_(self.out_img.weight)
            nn.init.zeros_(self.out_img.bias)
            nn.init.zeros_(self.out_seg.weight)
            nn.init.zeros_(self.out_seg.bias)
        scale = torch.full((1,), float(initial_scale))
        self.scale_img = nn.Parameter(scale.clone())
        self.scale_seg = nn.Parameter(scale.clone())

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


class CDIBF16C64(_CDIBBase):
    """CDIB using the F16C64 VAE ResNet/normalization blocks."""

    def __init__(self, channels: int, **kwargs: Any):
        super().__init__(
            channels,
            resblock_cls=mj64_vae.ResnetBlock,
            normalize_fn=mj64_vae.Normalize,
            **kwargs,
        )


class CDIBF8C32(_CDIBBase):
    """CDIB using the F8C32 Swin VAE ResNet/normalization blocks."""

    def __init__(self, channels: int, **kwargs: Any):
        super().__init__(
            channels,
            resblock_cls=f8c32_swin.ResnetBlock,
            normalize_fn=f8c32_swin.Normalize,
            **kwargs,
        )


class _JointSegDecoderBase(nn.Module):
    def __init__(
        self,
        *,
        z_channels: int,
        a_tex_channels: int,
        resolution: int,
        ch: int,
        ch_mult: Sequence[int],
        num_res_blocks: Sequence[int],
        in_channels: int,
        seg_channels: int,
        decoder_activation: str,
        image_output_scale: float,
        image_output_shift: float,
        cdib_cls: type[nn.Module],
        cdib_levels: Optional[Sequence[int]],
        cdib_dropout: float,
        seg_init_from_image: bool,
        seg_output_bias: float,
        zero_init_cdib_outputs: bool,
        cdib_initial_scale: float,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.a_tex_channels = int(a_tex_channels)
        self.in_channels = int(in_channels)
        self.seg_channels = int(seg_channels)
        self.resolution = int(resolution)
        self.ch = int(ch)
        self.ch_mult = _as_tuple(ch_mult, "ch_mult")
        self.num_res_blocks = _as_tuple(num_res_blocks, "num_res_blocks")
        if len(self.ch_mult) != len(self.num_res_blocks):
            raise ValueError("ch_mult and num_res_blocks must have the same length")

        self.decoder_activation = str(decoder_activation)
        self.image_output_scale = float(image_output_scale)
        self.image_output_shift = float(image_output_shift)
        self.seg_init_from_image = bool(seg_init_from_image)
        self.seg_output_bias = float(seg_output_bias)
        self.zero_init_cdib_outputs = bool(zero_init_cdib_outputs)
        self.cdib_initial_scale = float(cdib_initial_scale)
        self.gradient_checkpointing = False

        block_in = self.ch * self.ch_mult[-1]
        if block_in % self.a_tex_channels != 0:
            raise ValueError(
                f"a_tex_channels={self.a_tex_channels} must evenly divide "
                f"decoder block_in={block_in} (ch*ch_mult[-1])"
            )

        num_resolutions = len(self.ch_mult)
        if cdib_levels is None:
            cdib_levels = list(range(num_resolutions - 1, 0, -1))
        self.cdib_levels = [int(x) for x in cdib_levels]
        self.cdib = nn.ModuleDict(
            {
                str(lvl): cdib_cls(
                    self.ch * self.ch_mult[lvl],
                    dropout=cdib_dropout,
                    zero_init_outputs=self.zero_init_cdib_outputs,
                    initial_scale=self.cdib_initial_scale,
                )
                for lvl in self.cdib_levels
            }
        )

    def _finish_initialization(self, image_checkpoint: Optional[str]) -> None:
        if image_checkpoint:
            self._load_image_checkpoint(image_checkpoint)
        self._init_segmentation_branch()

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
            elif not key.startswith(
                (
                    "encoder.",
                    "quant_conv.",
                    "post_quant_conv.",
                    "image_post_quant_conv.",
                    "seg_decoder.",
                    "cdib.",
                )
            ):
                prefixed[f"image_decoder.{key}"] = value

        missing, unexpected = self.load_state_dict(prefixed, strict=False)
        print(
            f"[{type(self).__name__}] loaded image checkpoint from {path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    @staticmethod
    def _copy_shape_matched_state(src: nn.Module, dst: nn.Module) -> int:
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        matched = {
            key: value
            for key, value in src_state.items()
            if key in dst_state and tuple(value.shape) == tuple(dst_state[key].shape)
        }
        if matched:
            dst.load_state_dict(matched, strict=False)
        return len(matched)

    def _init_segmentation_branch(self) -> None:
        if self.seg_init_from_image:
            self._copy_shape_matched_state(self.image_decoder, self.seg_decoder)

        conv_out = getattr(self.seg_decoder, "conv_out", None)
        if conv_out is None:
            return
        with torch.no_grad():
            if getattr(conv_out, "weight", None) is not None:
                conv_out.weight.zero_()
            if getattr(conv_out, "bias", None) is not None:
                conv_out.bias.fill_(self.seg_output_bias)

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
        return latent

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


class JointSegDecoderF16C64(_JointSegDecoderBase):
    """JSD for the F16C64 NPU VAE decoder."""

    def __init__(
        self,
        z_channels: int,
        a_tex_channels: int,
        resolution: int = 512,
        ch: int = 128,
        ch_mult: Sequence[int] = _F16_CH_MULT,
        num_res_blocks: Sequence[int] = _F16_NUM_RES_BLOCKS,
        in_channels: int = 3,
        seg_channels: int = 1,
        decoder_activation: str = "silu",
        image_output_scale: float = 1.0 / SCALING_FACTOR,
        image_output_shift: float = SHIFTING_FACTOR,
        cdib_levels: Optional[Sequence[int]] = None,
        image_checkpoint: Optional[str] = None,
        seg_init_from_image: bool = True,
        seg_output_bias: float = -4.0,
        zero_init_cdib_outputs: bool = True,
        cdib_initial_scale: float = 1.0,
        **decoder_kwargs: Any,
    ):
        decoder_kwargs = dict(decoder_kwargs)
        _reject_params(
            decoder_kwargs,
            {
                "decoder_target",
                "image_post_quant_conv",
                "image_embed_dim",
                "image_latent_shift_factor",
                "image_latent_scaling_factor",
                "embed_dim",
                "shift_factor",
                "scaling_factor",
            },
            type(self).__name__,
        )
        ch_mult_tuple = _as_tuple(ch_mult, "ch_mult")
        num_res_blocks_tuple = _as_tuple(num_res_blocks, "num_res_blocks")
        if ch_mult_tuple != _F16_CH_MULT or num_res_blocks_tuple != _F16_NUM_RES_BLOCKS:
            raise ValueError(
                "JointSegDecoderF16C64 uses the fixed F16C64 VAE layout: "
                f"ch_mult={_F16_CH_MULT}, num_res_blocks={_F16_NUM_RES_BLOCKS}"
            )

        super().__init__(
            z_channels=z_channels,
            a_tex_channels=a_tex_channels,
            resolution=resolution,
            ch=ch,
            ch_mult=ch_mult_tuple,
            num_res_blocks=num_res_blocks_tuple,
            in_channels=in_channels,
            seg_channels=seg_channels,
            decoder_activation=decoder_activation,
            image_output_scale=image_output_scale,
            image_output_shift=image_output_shift,
            cdib_cls=CDIBF16C64,
            cdib_levels=cdib_levels,
            cdib_dropout=float(decoder_kwargs.get("dropout", 0.0)),
            seg_init_from_image=seg_init_from_image,
            seg_output_bias=seg_output_bias,
            zero_init_cdib_outputs=zero_init_cdib_outputs,
            cdib_initial_scale=cdib_initial_scale,
        )
        self.image_decoder = mj64_vae.Decoder(
            z_channels=z_channels,
            resolution=resolution,
            ch=ch,
            in_channels=in_channels,
            **decoder_kwargs,
        )
        self.seg_decoder = mj64_vae.Decoder(
            z_channels=a_tex_channels,
            resolution=resolution,
            ch=ch,
            in_channels=seg_channels,
            **decoder_kwargs,
        )
        self._finish_initialization(image_checkpoint)


class JointSegDecoderF8C32(_JointSegDecoderBase):
    """JSD for the F8C32 Swin VAE decoder.

    The public initialization shape mirrors ``f8c32_swin.VaeDecoder``:
    VAE structure lives in ``ddconfig`` or the same flat ddconfig keys, while
    JSD-specific knobs stay as explicit arguments.
    """

    def __init__(
        self,
        a_tex_channels: int,
        embed_dim: int = 32,
        shift_factor: float = f8c32_swin.SHIFT_FACTOR,
        scaling_factor: float = f8c32_swin.SCALING_FACTOR,
        seg_channels: int = 1,
        decoder_activation: str = "swish",
        image_output_scale: float = 1.0,
        image_output_shift: float = 0.0,
        cdib_levels: Optional[Sequence[int]] = None,
        image_checkpoint: Optional[str] = None,
        seg_init_from_image: bool = True,
        seg_output_bias: float = -4.0,
        zero_init_cdib_outputs: bool = True,
        cdib_initial_scale: float = 1.0,
        **decoder_kwargs: Any,
    ):
        decoder_kwargs = dict(decoder_kwargs)
        _reject_params(
            decoder_kwargs,
            {
                "decoder_target",
                "image_post_quant_conv",
                "image_embed_dim",
                "image_latent_shift_factor",
                "image_latent_scaling_factor",
            },
            type(self).__name__,
        )
        ddconfig, rest = f8c32_swin._split_config(decoder_kwargs)
        self.embed_dim = int(embed_dim)
        self.shift_factor = float(rest.pop("shift_factor", shift_factor))
        self.scaling_factor = float(rest.pop("scaling_factor", scaling_factor))
        if rest:
            unknown = ", ".join(sorted(rest))
            raise TypeError(f"{type(self).__name__} got unexpected params: {unknown}")
        ddconfig = dict(ddconfig)
        z_channels = int(ddconfig.get("z_channels", self.embed_dim))
        if self.embed_dim != z_channels:
            raise ValueError(
                "JointSegDecoderF8C32 no longer owns a post-quant conv, so "
                f"embed_dim ({self.embed_dim}) must equal ddconfig.z_channels "
                f"({z_channels})"
            )
        ddconfig["z_channels"] = z_channels
        ch = int(ddconfig.get("ch", 128))
        ch_mult = _as_tuple(ddconfig.get("ch_mult", _F8_CH_MULT), "ddconfig.ch_mult")
        num_res_blocks = _as_tuple(
            ddconfig.get("num_res_blocks", _F8_NUM_RES_BLOCKS),
            "ddconfig.num_res_blocks",
        )
        resolution = int(ddconfig.get("resolution", 256))
        in_channels = int(ddconfig.get("in_channels", 3))

        super().__init__(
            z_channels=z_channels,
            a_tex_channels=a_tex_channels,
            resolution=resolution,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            seg_channels=seg_channels,
            decoder_activation=decoder_activation,
            image_output_scale=image_output_scale,
            image_output_shift=image_output_shift,
            cdib_cls=CDIBF8C32,
            cdib_levels=cdib_levels,
            cdib_dropout=float(decoder_kwargs.get("dropout", 0.0)),
            seg_init_from_image=seg_init_from_image,
            seg_output_bias=seg_output_bias,
            zero_init_cdib_outputs=zero_init_cdib_outputs,
            cdib_initial_scale=cdib_initial_scale,
        )
        self.ddconfig = ddconfig
        self.image_decoder = f8c32_swin.Decoder(**ddconfig)
        seg_ddconfig = dict(ddconfig)
        seg_ddconfig["z_channels"] = a_tex_channels
        seg_ddconfig["in_channels"] = seg_channels
        seg_ddconfig["out_ch"] = seg_channels
        self.seg_decoder = f8c32_swin.Decoder(**seg_ddconfig)
        self._finish_initialization(image_checkpoint)

    def _prepare_image_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent / self.scaling_factor + self.shift_factor


# Backward-compatible F16 name. New configs should use the explicit classes.
CDIB = CDIBF16C64
JointSegDecoder = JointSegDecoderF16C64
