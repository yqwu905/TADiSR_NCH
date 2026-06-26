from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionMSELoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = F.mse_loss(prediction, target)
        mae = (prediction - target).abs().mean()
        return {
            "loss": loss,
            "mae": mae.detach(),
        }

