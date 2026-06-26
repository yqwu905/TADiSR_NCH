from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint


class TinyRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        output_dim: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.net(x)
        return {
            "prediction": prediction,
            "mean_prediction": prediction.mean(dim=-1),
        }


class WrappedRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        output_dim: int = 2,
    ):
        super().__init__()
        self.model = TinyRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)


class CheckpointedRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        output_dim: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.use_gc = False
        self.gc_method_calls: list[tuple[str, dict]] = []

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_gc:
            prediction = torch.utils.checkpoint.checkpoint(
                self.net, x, use_reentrant=False
            )
        else:
            prediction = self.net(x)
        return {
            "prediction": prediction,
            "mean_prediction": prediction.mean(dim=-1),
        }

    def gradient_checkpointing_enable(self, **kwargs):
        self.use_gc = True
        self.gc_method_calls.append(("gradient_checkpointing_enable", dict(kwargs)))

    def enable_gradient_checkpointing(self, **kwargs):
        self.use_gc = True
        self.gc_method_calls.append(("enable_gradient_checkpointing", dict(kwargs)))

