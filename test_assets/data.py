from __future__ import annotations

import torch
from torch.utils.data import Dataset


class LinearRegressionDataset(Dataset):
    """Deterministic toy dataset for framework training smoke tests."""

    def __init__(
        self,
        length: int = 32,
        input_dim: int = 4,
        output_dim: int = 2,
        seed: int = 123,
        noise_std: float = 0.0,
    ):
        if length <= 0:
            raise ValueError("length must be positive")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        generator = torch.Generator().manual_seed(int(seed))
        self.inputs = torch.randn(length, input_dim, generator=generator)
        weight = torch.arange(
            1,
            input_dim * output_dim + 1,
            dtype=torch.float32,
        ).reshape(input_dim, output_dim)
        weight = weight / float(input_dim * output_dim)
        bias = torch.linspace(-0.25, 0.25, output_dim)
        targets = self.inputs @ weight + bias

        if noise_std:
            targets = targets + torch.randn(
                targets.shape,
                generator=generator,
                dtype=targets.dtype,
            ) * float(noise_std)

        self.targets = targets

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int):
        return {
            "input": self.inputs[index],
            "target": self.targets[index],
            "sample_id": torch.tensor(index, dtype=torch.long),
        }

