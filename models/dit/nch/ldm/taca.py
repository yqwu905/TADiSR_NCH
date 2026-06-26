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

    For each collected DiT block the attention processor pre-extracts an
    image→text attention slice of shape ``[B, H, S_img, n_text]`` — the
    softmax probability of every image patch attending to the selected
    text tokens — via a chunked logsumexp side path that avoids
    materializing the full ``[B, H, S, S]`` weight matrix. The slices are
    flattened over heads and text tokens, concatenated across layers, and
    linearly projected to ``a_tex`` of shape ``[B, img_seq_len, out_dim]``.

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

    def forward(self, slices: List[torch.Tensor]) -> torch.Tensor:
        """Project pre-extracted text-attention slices into ``a_tex``.

        Parameters
        ----------
        slices : list of Tensor
            Each tensor has shape ``[B, H, S_img, n_text]`` — the
            image→text attention probabilities for every image patch
            towards the selected text tokens, pre-extracted by the
            attention processor via chunked logsumexp.

        Returns
        -------
        Tensor
            ``a_tex`` of shape ``[B, S_img, out_dim]``.
        """
        if len(slices) != self.num_layers:
            raise ValueError(
                f"TACAProjection expected {self.num_layers} attention "
                f"slices, got {len(slices)}"
            )

        feats = []
        for s in slices:
            # s: [B, H, S_img, n_text] -> [B, S_img, H, n_text] -> [B, S_img, H*n_text]
            s = s.permute(0, 2, 1, 3).contiguous()
            feats.append(s.view(s.shape[0], s.shape[1], -1))

        concat = torch.cat(feats, dim=-1)  # [B, S_img, L*H*n_text]
        a_tex = self.proj(concat)  # [B, S_img, out_dim]
        return a_tex
