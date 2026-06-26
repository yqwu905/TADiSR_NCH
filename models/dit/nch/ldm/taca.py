"""
Text-Aware Cross Attention (TACA) projection head for TADiSR.

Extracts attention weights corresponding to text tokens from multiple DiT
joint self-attention blocks, concatenates them across layers, and projects
to a per-image-patch text-aware feature ``a_tex``. The projection is
zero-initialized so that, at the start of training, ``a_tex`` is exactly
zero and does not disturb the pretrained DiT / SR path.

Reference: TADiSR (Text-Aware Real-World Image Super-Resolution,
NeurIPS 2025, arXiv:2506.04641).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


class TACAProjection(nn.Module):
    """Text-Aware Cross Attention projection head.

    For each collected DiT block the attention weight tensor has shape
    ``[B, H, S, S]`` with ``S = text_seq_len + img_seq_len`` and text tokens
    occupying the first ``text_seq_len`` positions. The rows of the text
    tokens (restricted to the image columns) are extracted, flattened over
    heads and text tokens, concatenated across layers, and linearly
    projected to ``a_tex`` of shape ``[B, img_seq_len, out_dim]``.

    Parameters
    ----------
    num_layers : int
        Number of DiT blocks whose attention weights are collected (must
        match the length of the weights list passed to ``forward``).
    num_heads : int
        Number of attention heads per block.
    num_text_tokens : int
        Number of text-token rows extracted per block
        (``len(text_token_indices)``).
    out_dim : int
        Output channel dimension of ``a_tex`` per image patch.
    bias : bool
        Whether the projection linear uses a bias term.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_text_tokens: int,
        out_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_text_tokens = int(num_text_tokens)
        self.out_dim = int(out_dim)

        concat_dim = self.num_layers * self.num_heads * self.num_text_tokens
        self.proj = nn.Linear(concat_dim, self.out_dim, bias=bias)
        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        attn_weights_list: List[torch.Tensor],
        text_token_indices: Sequence[int],
        text_seq_len: int,
    ) -> torch.Tensor:
        """Project multi-layer attention weights into ``a_tex``.

        Parameters
        ----------
        attn_weights_list : list of Tensor
            Each tensor has shape ``[B, H, S, S]`` where
            ``S = text_seq_len + img_seq_len``. Text tokens occupy the first
            ``text_seq_len`` positions.
        text_token_indices : sequence of int
            Indices (within the text portion) of the text tokens to extract.
        text_seq_len : int
            Length of the text portion at the front of the attention seq.

        Returns
        -------
        Tensor
            ``a_tex`` of shape ``[B, img_seq_len, out_dim]``.
        """
        if len(attn_weights_list) != self.num_layers:
            raise ValueError(
                f"TACAProjection expected {self.num_layers} attention "
                f"weight tensors, got {len(attn_weights_list)}"
            )

        idx = torch.as_tensor(list(text_token_indices), dtype=torch.long)
        idx = idx.to(attn_weights_list[0].device)

        feats = []
        for weights in attn_weights_list:
            # weights: [B, H, S, S]; image columns start at text_seq_len
            img_cols = weights[:, :, :, text_seq_len:]  # [B, H, S, S_img]
            # text-token rows -> [B, H, n_text, S_img]
            rows = img_cols.index_select(2, idx)
            # -> [B, S_img, H, n_text] -> [B, S_img, H*n_text]
            rows = rows.permute(0, 3, 1, 2).contiguous()
            feats.append(rows.view(rows.shape[0], rows.shape[1], -1))

        concat = torch.cat(feats, dim=-1)  # [B, S_img, L*H*n_text]
        a_tex = self.proj(concat)  # [B, S_img, out_dim]
        return a_tex
