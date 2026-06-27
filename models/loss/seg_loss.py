"""
TADiSR segmentation-oriented loss (Stage 4).

Implements the segmentation loss from TADiSR (NeurIPS 2025,
arXiv:2506.04641), Eq. (9):

    ell_seg = ||s_hat - s||^2_2 + lambda3 * FocalLoss(s_hat, s)
              + lambda4 * DiceLoss(s_hat, s)

where
  - s_hat, s    predicted / ground-truth text seg mask  [B, 1, H, W] in [0, 1]
  - lambda3 = 10.0, lambda4 = 1.0

The predicted mask comes from the JSD segmentation branch (already
sigmoid-activated), so the losses operate on probabilities.  For numerical
stability the probabilities are clamped before taking logs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    """Segmentation-oriented loss: MSE + Focal + Dice.

    Parameters
    ----------
    w_l2 : float
        Weight for the pixel-wise MSE term (``||s_hat - s||^2_2``).
    w_focal : float
        Weight for the Focal loss term (``lambda3``).
    w_dice : float
        Weight for the Dice loss term (``lambda4``).
    focal_gamma : float
        Focal focusing exponent (default 2.0, after Lin2017focal).
    focal_alpha : float
        Balancing weight for the positive (text) class (default 0.25,
        after RetinaNet).  ``alpha_t = alpha`` where ``target=1`` and
        ``1 - alpha`` otherwise.
    smooth : float
        Smoothing constant for the Dice denominator to avoid division by
        zero.
    eps : float
        Numerical clamp for predicted probabilities before ``log``.
    """

    def __init__(
        self,
        w_l2: float = 1.0,
        w_focal: float = 10.0,
        w_dice: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        smooth: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.w_l2 = float(w_l2)
        self.w_focal = float(w_focal)
        self.w_dice = float(w_dice)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.smooth = float(smooth)
        self.eps = float(eps)

    def _focal_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Binary focal loss on probabilities.

        ``FL = -alpha_t * (1 - p_t)^gamma * log(p_t)``
        where ``p_t = p`` if ``target=1`` else ``1 - p``.
        """
        p = pred.clamp(self.eps, 1.0 - self.eps)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        log_pt = torch.log(p * target + (1.0 - p) * (1.0 - target) + self.eps)
        alpha_t = self.focal_alpha * target + (1.0 - self.focal_alpha) * (
            1.0 - target
        )
        focal = -alpha_t * (1.0 - p_t) ** self.focal_gamma * log_pt
        return focal.mean()

    def _dice_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Soft Dice loss ``1 - 2|p (.) g| / (|p| + |g|)`` averaged over
        the batch."""
        dims = tuple(range(1, pred.ndim))
        inter = (pred * target).sum(dim=dims)
        denom = pred.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute the segmentation-oriented loss.

        Parameters
        ----------
        pred : [B, 1, H, W]
            Predicted text mask in ``[0, 1]``.
        target : [B, 1, H, W]
            Ground-truth text mask in ``[0, 1]`` (or ``{0, 1}``).

        Returns
        -------
        dict with keys ``loss``, ``l2``, ``focal``, ``dice``.
        """
        if pred.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"pred and target spatial size mismatch: "
                f"{tuple(pred.shape[-2:])} vs {tuple(target.shape[-2:])}"
            )
        if pred.shape[1] != target.shape[1]:
            raise ValueError(
                f"pred and target channel mismatch: "
                f"{pred.shape[1]} vs {target.shape[1]}"
            )

        l2 = F.mse_loss(pred, target)
        total = self.w_l2 * l2

        focal = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.w_focal > 0:
            focal = self._focal_loss(pred, target)
            total = total + self.w_focal * focal

        dice = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.w_dice > 0:
            dice = self._dice_loss(pred, target)
            total = total + self.w_dice * dice

        return {
            "loss": total,
            "l2": l2.detach(),
            "focal": focal.detach(),
            "dice": dice.detach(),
        }
