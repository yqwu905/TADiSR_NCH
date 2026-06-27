"""
TADiSR image super-resolution loss (Stage 4).

Implements the SR-oriented loss from TADiSR (NeurIPS 2025,
arXiv:2506.04641), Eq. (7)-(8):

    ell_img = ||x_hat - x||^2_2 + lambda1 * LPIPS(x_hat, x)
              + lambda2 * ell_mf

    ell_mf  = ||[1 - s_hat (.) s - (1 - s_hat) (.) (1 - s)]^gamma
              (.) (grad x_hat - grad x)^2||_1

where
  - x_hat, x    predicted / ground-truth HR image   [B, C, H, W]
  - s_hat, s    predicted / ground-truth text seg    [B, 1, H, W] in [0, 1]
  - grad        Sobel edge operator (Gx, Gy)
  - (.)         pixel-wise (Hadamard) multiplication
  - gamma       focal focusing parameter (default 2.0, after Lin2017focal)
  - lambda1 = 5.0, lambda2 = 10.0

The modified focal term weights the squared Sobel-edge difference by the
per-pixel misclassification probability of the segmentation, raised to a
focal power.  Boundary pixels where the segmentation prediction disagrees
with the ground truth receive a higher weight, steering the SR model
toward faithful text-edge reconstruction and coupling the two tasks.

LPIPS is optional: when the ``lpips`` package is unavailable (e.g. CPU
smoke tests) the term is disabled gracefully, so the module still runs.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sobel_weight(
    channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build a grouped-conv weight ``[2C, 1, 3, 3]`` applying the Sobel
    Gx and Gy kernels independently to each of the ``channels`` input
    channels (``groups=C``).  Output channel ordering is
    ``[Gx_c0, Gy_c0, Gx_c1, Gy_c1, ...]``."""
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    )
    gy = gx.t().contiguous()
    base = torch.stack([gx, gy])[:, None]  # [2, 1, 3, 3]
    return base.repeat(channels, 1, 1, 1)   # [2C, 1, 3, 3]


class TADiSRLoss(nn.Module):
    """SR-oriented loss: MSE + LPIPS + modified focal edge loss.

    Parameters
    ----------
    w_l2 : float
        Weight for the pixel-wise MSE term (``||x_hat - x||^2_2``).
    w_lpips : float
        Weight for the LPIPS perceptual term (``lambda1``).  Set to 0 to
        skip and avoid importing the ``lpips`` package.
    w_mf : float
        Weight for the modified focal edge term (``lambda2``).
    focal_gamma : float
        Focal focusing exponent ``gamma`` applied to the per-pixel
        misclassification probability.
    lpips_net : str
        LPIPS backbone, forwarded to ``lpips.LPIPS(net=...)``.
    reduction : str
        Reduction for the modified focal term, ``"mean"`` or ``"sum"``.
        Default ``"mean"`` keeps the scale comparable to MSE.
    eps : float
        Numerical clamp for segmentation probabilities.
    """

    def __init__(
        self,
        w_l2: float = 1.0,
        w_lpips: float = 5.0,
        w_mf: float = 10.0,
        focal_gamma: float = 2.0,
        lpips_net: str = "vgg",
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.w_l2 = float(w_l2)
        self.w_lpips = float(w_lpips)
        self.w_mf = float(w_mf)
        self.focal_gamma = float(focal_gamma)
        self.reduction = reduction
        self.eps = float(eps)

        self._lpips = None
        self._lpips_net = lpips_net
        if self.w_lpips > 0:
            self._build_lpips()

        self._sobel_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _build_lpips(self) -> None:
        try:
            import lpips
        except ImportError as e:
            print(
                f"[TADiSRLoss] lpips not available ({e}); disabling LPIPS "
                f"term. Install with: pip install lpips"
            )
            self.w_lpips = 0.0
            return
        self._lpips = lpips.LPIPS(net=self._lpips_net)
        self._lpips.eval()
        for p in self._lpips.parameters():
            p.requires_grad_(False)

    def _sobel(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Sobel gradients (Gx, Gy) per channel.

        Returns two tensors of shape ``[B, C, H, W]`` (spatial size
        preserved via reflect padding)."""
        b, c, h, w = img.shape
        key = (c, img.device, img.dtype)
        weight = self._sobel_cache.get(key)
        if weight is None:
            weight = _build_sobel_weight(c, img.device, img.dtype)
            self._sobel_cache[key] = weight
        elif weight.device != img.device or weight.dtype != img.dtype:
            weight = weight.to(device=img.device, dtype=img.dtype)
            self._sobel_cache[key] = weight

        out = F.conv2d(img, weight, padding=1, groups=c)  # [B, 2C, H, W]
        gx = out[:, 0::2]
        gy = out[:, 1::2]
        return gx, gy

    def _modified_focal(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_seg: torch.Tensor,
        target_seg: torch.Tensor,
    ) -> torch.Tensor:
        gx_p, gy_p = self._sobel(pred)
        gx_t, gy_t = self._sobel(target)
        edge_diff_sq = (gx_p - gx_t) ** 2 + (gy_p - gy_t) ** 2  # [B, C, H, W]

        s_hat = pred_seg.clamp(self.eps, 1.0 - self.eps)
        s = target_seg.clamp(self.eps, 1.0 - self.eps)
        correct = s_hat * s + (1.0 - s_hat) * (1.0 - s)  # P(correct), [B,1,H,W]
        weight = (1.0 - correct) ** self.focal_gamma       # [B,1,H,W]

        if edge_diff_sq.shape[1] != weight.shape[1]:
            weight = weight.expand_as(edge_diff_sq)

        weighted = weight * edge_diff_sq
        if self.reduction == "sum":
            return weighted.sum()
        return weighted.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_seg: Optional[torch.Tensor] = None,
        target_seg: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the SR-oriented loss.

        Parameters
        ----------
        pred : [B, C, H, W]
            Predicted HR image.  Expected in the same range as ``target``;
            for LPIPS both should be in ``[-1, 1]``.
        target : [B, C, H, W]
            Ground-truth HR image.
        pred_seg : [B, 1, H, W] or None
            Predicted text segmentation mask in ``[0, 1]``.  Required for
            the modified focal term; if ``None`` the term is skipped.
        target_seg : [B, 1, H, W] or None
            Ground-truth text segmentation mask in ``[0, 1]``.

        Returns
        -------
        dict with keys ``loss``, ``l2``, ``lpips``, ``mf``.
        """
        if pred.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"pred and target spatial size mismatch: "
                f"{tuple(pred.shape[-2:])} vs {tuple(target.shape[-2:])}"
            )

        l2 = F.mse_loss(pred, target)
        total = self.w_l2 * l2

        lpips_val = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.w_lpips > 0 and self._lpips is not None:
            if pred.shape[1] != 3:
                print(
                    f"[TADiSRLoss] LPIPS expects 3-channel input, got "
                    f"{pred.shape[1]}; skipping LPIPS term"
                )
            else:
                lpips_val = self._lpips(pred, target).mean()
                total = total + self.w_lpips * lpips_val

        mf_val = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.w_mf > 0 and pred_seg is not None and target_seg is not None:
            if pred_seg.shape[-2:] != pred.shape[-2:]:
                pred_seg = F.interpolate(
                    pred_seg, size=pred.shape[-2:], mode="bilinear",
                    align_corners=False,
                )
            if target_seg.shape[-2:] != pred.shape[-2:]:
                target_seg = F.interpolate(
                    target_seg, size=pred.shape[-2:], mode="bilinear",
                    align_corners=False,
                )
            mf_val = self._modified_focal(pred, target, pred_seg, target_seg)
            total = total + self.w_mf * mf_val

        return {
            "loss": total,
            "l2": l2.detach(),
            "lpips": lpips_val.detach(),
            "mf": mf_val.detach(),
        }
