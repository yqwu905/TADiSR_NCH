from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:  # pragma: no cover - older timm compatibility
    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    except ImportError:  # pragma: no cover - test environments may omit timm
        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(
                    shape, dtype=x.dtype, device=x.device
                )
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor

        def to_2tuple(x):
            return x if isinstance(x, tuple) else (x, x)

        def trunc_normal_(tensor, std=0.02):
            return nn.init.trunc_normal_(tensor, std=std)


SHIFT_FACTOR = 0.07050679
SCALING_FACTOR = 0.2517327


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(
        b,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, c)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, h: int, w: int
) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b,
        h // window_size,
        w // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads,
            )
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        self.register_buffer(
            "attn_mask",
            self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None,
        )

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, _l, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_mask = (
            self.attn_mask
            if self.input_resolution == x_size
            else self.calculate_mask(x_size).to(x.device)
        )
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        b, _hw, _c = x.shape
        return x.transpose(1, 2).view(b, self.embed_dim, x_size[0], x_size[1])


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, Sequence)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            if downsample is not None
            else None
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
        img_size: int = 224,
        patch_size: int = 4,
        resi_connection: str = "1conv",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"unknown resi_connection: {resi_connection}")
        self.patch_embed = PatchEmbed(img_size, patch_size, 0, dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, 0, dim, norm_layer=None)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h = self.residual_group(x, x_size)
        h = self.patch_unembed(h, x_size)
        h = self.conv(h)
        h = self.patch_embed(h)
        return h + x


class SwinAttn(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 1,
        embed_dim: int = 512,
        depths: Sequence[int] = (6, 6),
        num_heads: Sequence[int] = (8, 8),
        window_size: int = 16,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.01,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        resi_connection: str = "1conv",
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"unknown resi_connection: {resi_connection}")

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        return {"relative_position_bias_table"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_after_body(self.forward_features(x)) + x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        if in_channels * factor**2 % out_channels != 0:
            raise ValueError("pixel-unshuffle channel average requires divisible channels")
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        b, _c, h, w = x.shape
        x = x.view(b, self.out_channels, self.group_size, h, w)
        return x.mean(dim=2)


class PixelUnshuffleChannelAveragingDownSampleLayerConvOut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        if in_channels * factor**2 % out_channels != 0:
            raise ValueError("channel average shortcut requires divisible channels")
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        x = x.view(b, self.out_channels, self.group_size, h, w)
        return x.mean(dim=2)


class ChannelDuplicatingPixelUnshuffleUpSampleLayer2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        if out_channels * factor**2 % in_channels != 0:
            raise ValueError("channel duplicate shortcut requires divisible channels")
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        return F.pixel_shuffle(x, self.factor)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv,
            "b (qkv heads c) h w -> qkv b heads c (h w)",
            heads=self.heads,
            qkv=3,
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", h=h, w=w)
        return self.to_out(out)


class EncodeConvOutShortcut(nn.Module):
    def __init__(self, in_channels: int, z_channels: int, double_z: bool = True):
        super().__init__()
        out_channels = 2 * z_channels if double_z else z_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayerConvOut(
            in_channels, out_channels, factor=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class DecodeConvInShortcut(nn.Module):
    def __init__(self, in_channels: int, z_channels: int, double_z: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(z_channels, in_channels, 3, 1, 1)
        self.shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer2D(
            z_channels, in_channels, factor=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            nn.Conv2d(in_channels, in_channels, 3, 1, 1) if self.with_conv else None
        )
        self.shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer2D(
            in_channels, in_channels, factor=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = self.shortcut(x)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.conv is not None:
            x = self.conv(x) + sh
        return x


class DownsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            nn.Conv2d(in_channels, in_channels, 3, 2, 0) if self.with_conv else None
        )
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels, in_channels, factor=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            sh = self.shortcut(x)
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x) + sh
        return F.avg_pool2d(x, kernel_size=2, stride=2) + self.shortcut(x)


class DownsampleShortcutEnlargeCh(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            nn.Conv2d(in_channels, in_channels * 2, 3, 2, 0)
            if self.with_conv
            else None
        )
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels, in_channels * 2, factor=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            sh = self.shortcut(x)
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x) + sh
        return F.avg_pool2d(x, kernel_size=2, stride=2) + self.shortcut(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class LinAttnBlock(LinearAttention):
    def __init__(self, in_channels: int):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** -0.5)
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


def make_attn(in_channels: int, attn_type: str = "vanilla") -> nn.Module:
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    if attn_type == "linear":
        return LinAttnBlock(in_channels)
    if attn_type == "none":
        return nn.Identity()
    raise ValueError(f"unknown attn_type: {attn_type}")


def _as_tuple(values: Sequence[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    return tuple(default if values is None else values)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: Sequence[int] = (2, 3, 2, 2),
        attn_resolutions: Sequence[int] = (),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 32,
        double_z: bool = True,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        num_resize: int = 4,
        swin_depths: Sequence[int] = (6, 6),
        swin_num_heads: Sequence[int] = (8, 8),
        swin_window_size: int = 16,
        **ignore_kwargs: Any,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"

        self.ch = ch
        self.temb_ch = 0
        self.ch_mult = _as_tuple(ch_mult, (1, 2, 4, 4))
        self.num_res_blocks = _as_tuple(num_res_blocks, (2, 3, 2, 2))
        if len(self.ch_mult) != len(self.num_res_blocks):
            raise ValueError("ch_mult and num_res_blocks must have the same length")
        self.num_resolutions = len(self.ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.downsampling_factor = 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)

        curr_res = resolution
        curr_channels = ch
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * self.ch_mult[i_level]
            for _i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=curr_channels,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                curr_channels = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(curr_channels, attn_type=attn_type))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in (0, 1) and i_level != self.num_resolutions - 1:
                down.downsample = DownsampleShortcutEnlargeCh(
                    curr_channels, resamp_with_conv
                )
                curr_channels *= 2
                curr_res //= 2
            elif i_level != self.num_resolutions - 1:
                down.downsample = DownsampleShortcut(curr_channels, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        block_in = curr_channels
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = SwinAttn(
            img_size=curr_res,
            embed_dim=block_in,
            depths=swin_depths,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = EncodeConvOutShortcut(
            in_channels=block_in,
            z_channels=z_channels,
            double_z=double_z,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temb = None
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = swish(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: Sequence[int] = (2, 3, 2, 2),
        attn_resolutions: Sequence[int] = (),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 32,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        use_linear_attn: bool = False,
        num_resize: int = 4,
        attn_type: str = "vanilla",
        swin_depths: Sequence[int] = (6, 6),
        swin_num_heads: Sequence[int] = (8, 8),
        swin_window_size: int = 16,
        **ignore_kwargs: Any,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"

        self.ch = ch
        self.temb_ch = 0
        self.ch_mult = _as_tuple(ch_mult, (1, 2, 4, 4))
        self.num_res_blocks = _as_tuple(num_res_blocks, (2, 3, 2, 2))
        if len(self.ch_mult) != len(self.num_res_blocks):
            raise ValueError("ch_mult and num_res_blocks must have the same length")
        self.num_resolutions = len(self.ch_mult)
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.downsampling_factor = 2 ** (self.num_resolutions - 1)

        block_in = ch * self.ch_mult[self.num_resolutions - 1]
        curr_res = resolution // self.downsampling_factor
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = DecodeConvInShortcut(
            in_channels=block_in,
            z_channels=z_channels,
        )

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = SwinAttn(
            img_size=curr_res,
            embed_dim=block_in,
            depths=swin_depths,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        curr_channels = block_in
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * self.ch_mult[i_level]
            for _i_block in range(self.num_res_blocks[i_level] + 1):
                block.append(
                    ResnetBlock(
                        in_channels=curr_channels,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                curr_channels = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(curr_channels, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpsampleShortcut(curr_channels, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = Normalize(curr_channels)
        self.conv_out = nn.Conv2d(curr_channels, out_ch, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level] + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


def _split_config(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs = dict(kwargs)
    ddconfig = dict(kwargs.pop("ddconfig", {}) or {})
    if not ddconfig:
        dd_keys = {
            "ch",
            "out_ch",
            "ch_mult",
            "num_res_blocks",
            "attn_resolutions",
            "dropout",
            "resamp_with_conv",
            "in_channels",
            "resolution",
            "z_channels",
            "double_z",
            "use_linear_attn",
            "attn_type",
            "num_resize",
            "swin_depths",
            "swin_num_heads",
            "swin_window_size",
        }
        for key in list(kwargs):
            if key in dd_keys:
                ddconfig[key] = kwargs.pop(key)
    return ddconfig, kwargs


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig: Mapping[str, Any] | None = None,
        embed_dim: int = 32,
        monitor: str | None = None,
        only_encoder: bool = False,
        shift_factor: float = SHIFT_FACTOR,
        scaling_factor: float = SCALING_FACTOR,
        **kwargs: Any,
    ):
        super().__init__()
        merged_ddconfig = dict(ddconfig or {})
        merged_ddconfig.update(kwargs)
        self.encoder = Encoder(**merged_ddconfig)
        self.decoder = None if only_encoder else Decoder(**merged_ddconfig)
        z_channels = int(merged_ddconfig.get("z_channels", embed_dim))
        double_z = bool(merged_ddconfig.get("double_z", True))
        quant_in_channels = 2 * z_channels if double_z else z_channels
        self.quant_conv = nn.Conv2d(quant_in_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = int(embed_dim)
        self.monitor = monitor
        self.shift_factor = float(shift_factor)
        self.scaling_factor = float(scaling_factor)
        self.downsampling_factor = 2 ** (len(merged_ddconfig.get("ch_mult", (1, 2, 4, 4))) - 1)

    @staticmethod
    def gaussian_sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def encode_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self.quant_conv(self.encoder(x))
        return torch.chunk(moments, chunks=2, dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("AutoencoderKL was created with only_encoder=True")
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, logvar = self.encode_moments(x)
        mean, logvar = mean.float(), logvar.float()
        z = self.gaussian_sample(mean, logvar)
        x_rec = self.decode(z.to(dtype=x.dtype))
        return {"x_rec": x_rec, "z": z, "mean": mean, "logvar": logvar}


class VaeEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        shift_factor: float = SHIFT_FACTOR,
        scaling_factor: float = SCALING_FACTOR,
        sample: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        ddconfig, rest = _split_config(kwargs)
        self.encoder = Encoder(**ddconfig)
        z_channels = int(ddconfig.get("z_channels", embed_dim))
        double_z = bool(ddconfig.get("double_z", True))
        quant_in_channels = 2 * z_channels if double_z else z_channels
        self.quant_conv = nn.Conv2d(quant_in_channels, 2 * embed_dim, 1)
        self.embed_dim = int(embed_dim)
        self.shift_factor = float(rest.pop("shift_factor", shift_factor))
        self.scaling_factor = float(rest.pop("scaling_factor", scaling_factor))
        self.sample = bool(rest.pop("sample", sample))
        self.downsampling_factor = 2 ** (len(ddconfig.get("ch_mult", (1, 2, 4, 4))) - 1)

    @staticmethod
    def gaussian_sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        moments = self.quant_conv(self.encoder(x))
        mean, logvar = torch.chunk(moments, chunks=2, dim=1)
        latent = self.gaussian_sample(mean, logvar) if self.sample else mean
        latent = (latent - self.shift_factor) * self.scaling_factor
        return {"latent": latent}

    def get_fsdp_wrap_module_list(self) -> list[nn.Module]:
        encoder = self.encoder
        modules: list[nn.Module] = [encoder]
        for level in encoder.down:
            modules.extend(level.block)
            modules.extend(level.attn)
        modules.extend([encoder.mid.block_1, encoder.mid.attn_1, encoder.mid.block_2])
        return modules


class VaeDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        shift_factor: float = SHIFT_FACTOR,
        scaling_factor: float = SCALING_FACTOR,
        **kwargs: Any,
    ):
        super().__init__()
        ddconfig, rest = _split_config(kwargs)
        z_channels = int(ddconfig.get("z_channels", embed_dim))
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder(**ddconfig)
        self.shift_factor = float(rest.pop("shift_factor", shift_factor))
        self.scaling_factor = float(rest.pop("scaling_factor", scaling_factor))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = x / self.scaling_factor + self.shift_factor
        z = self.post_quant_conv(z)
        return {"recon": self.decoder(z)}

    def get_fsdp_wrap_module_list(self) -> list[nn.Module]:
        decoder = self.decoder
        modules: list[nn.Module] = [decoder]
        for level in decoder.up:
            modules.extend(level.block)
            modules.extend(level.attn)
        modules.extend([decoder.mid.block_1, decoder.mid.attn_1, decoder.mid.block_2])
        return modules


def f8c32_swin(pretrained: str | None = None, **kwargs: Any) -> AutoencoderKL:
    ddconfig = {
        "double_z": True,
        "z_channels": 32,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1, 2, 4, 4),
        "num_res_blocks": (2, 3, 2, 2),
        "attn_resolutions": (),
        "dropout": 0.0,
    }
    ddconfig.update(kwargs.pop("ddconfig", {}) or {})
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=32, **kwargs)
    if pretrained is not None:
        state = torch.load(pretrained, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    return model
