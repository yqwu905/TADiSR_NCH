"""
TADiSR super-resolution losses.

Stage 1 baseline: L2 + LPIPS (no text-aware focal / segmentation yet).
The full TADiSR loss (modified focal + seg) is added in stage 4.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRLoss(nn.Module):
    """
    Stage-1 SR loss: weighted L2 + optional LPIPS.

    Parameters
    ----------
    w_l2 : float
        Weight for the pixel-wise L2 (MSE) term.
    w_lpips : float
        Weight for the LPIPS perceptual term. Set to 0 to skip (avoids
        importing the lpips package, useful for CPU smoke tests).
    lpips_net : str
        LPIPS backbone, forwarded to lpips.LPIPS(net=...).
    """

    def __init__(
        self,
        w_l2: float = 1.0,
        w_lpips: float = 5.0,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.w_l2 = float(w_l2)
        self.w_lpips = float(w_lpips)
        self._lpips = None
        self._lpips_net = lpips_net
        if self.w_lpips > 0:
            self._build_lpips()

    def _build_lpips(self):
        try:
            import lpips
        except ImportError as e:
            print(
                f"[SRLoss] lpips not available ({e}); "
                f"disabling LPIPS term. Install with: pip install lpips"
            )
            self.w_lpips = 0.0
            return
        self._lpips = lpips.LPIPS(net=self._lpips_net)
        self._lpips.eval()
        for p in self._lpips.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: [B, C, H, W] images in [-1, 1] range (LPIPS expects
        this; if your images are [0,1] rescale before calling).

        Returns dict {"loss": ..., "l2": ..., "lpips": ...}.
        """
        l2 = F.mse_loss(pred, target)
        total = self.w_l2 * l2

        lpips_val = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.w_lpips > 0 and self._lpips is not None:
            lpips_val = self._lpips(pred, target).mean()
            total = total + self.w_lpips * lpips_val

        return {
            "loss": total,
            "l2": l2.detach(),
            "lpips": lpips_val.detach(),
        }
