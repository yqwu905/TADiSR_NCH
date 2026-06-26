"""
TADiSR training dataset loader.

Returns (lr, hr, mask, prompt) tuples. The text condition is provided as a
prompt string; the EmbeddingDB component (called via the `call` op in the
training phase) looks up the precomputed pangu embedding from the SQLite
cache. This keeps the dataset side lightweight and avoids bundling the pangu
inference stack into the DataLoader workers.

When real FTSR data is unavailable, set `data_root=None` (or point it at a
non-existent path) to use a synthetic random-data mode that produces valid
shapes for pipeline smoke testing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Fixed prompt used in TADiSR. This must exist as a key in the EmbeddingDB
# SQLite cache (after task-token prefixing if task_tokens is configured).
# The "text" keyword's token position in the pangu 256-sequence is determined
# offline via scripts/analyze_pangu_text_token.py and hardcoded into the
# training config's taca.text_token_indices.
DEFAULT_PROMPT = "A high-quality photo with clear text"


class TADiSRDataset(Dataset):
    """
    Dataset for FTSR triplets: (x_L, x_H, s) plus a fixed text prompt.

    Each sample provides:
      - lr:     [3, H_lr, W_lr]   low-resolution image, float32, [0,1]
      - hr:     [3, H_hr, W_hr]   high-resolution image, float32, [0,1]
      - mask:   [1, H_hr, W_hr]   text segmentation mask, float32, [0,1]
      - prompt: str               fixed text prompt for EmbeddingDB lookup

    Directory layout expected (when data_root points to real FTSR data):
        {data_root}/HR/{name}.png
        {data_root}/{lr_path}/{name}.png
        {data_root}/Mask/{name}.png
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        lr_path: str = "LR",
        prompt: str = DEFAULT_PROMPT,
        hr_size: int = 512,
        lr_size: Optional[int] = None,
        length: int = 64,
        seed: int = 42,
    ):
        self.prompt = prompt
        self.hr_size = hr_size
        self.lr_size = lr_size if lr_size is not None else hr_size // 4
        self._mode = "real"

        if data_root is not None:
            root = Path(data_root)
            self.hr_dir = root / "HR"
            self.lr_dir = root / lr_path
            self.mask_dir = root / "Mask"
            try:
                self.samples = sorted(
                    f.name for f in self.hr_dir.glob("*.png")
                )
            except (FileNotFoundError, OSError):
                self.samples = []
            if not self.samples:
                print(
                    f"[TADiSRDataset] no PNG found under {self.hr_dir}; "
                    f"falling back to synthetic mode"
                )
                self._mode = "synthetic"
        else:
            self._mode = "synthetic"

        if self._mode == "synthetic":
            self.length = int(length)
            self._gen = torch.Generator().manual_seed(int(seed))
            print(
                f"[TADiSRDataset] synthetic mode: {self.length} samples, "
                f"hr={self.hr_size}, lr={self.lr_size}"
            )
        else:
            self.length = len(self.samples)
            print(f"[TADiSRDataset] real mode: {self.length} samples from {data_root}")

    def __len__(self) -> int:
        return self.length

    def _synthetic_sample(self) -> dict:
        hr = torch.rand(3, self.hr_size, self.hr_size, generator=self._gen)
        # LR is generated at lr_size then upsampled to hr_size so the VAE
        # encoder produces an HR-resolution latent (the DiT operates at HR
        # latent resolution; the degradation is encoded in the pixel content).
        lr_small = torch.rand(3, self.lr_size, self.lr_size, generator=self._gen)
        lr = torch.nn.functional.interpolate(
            lr_small.unsqueeze(0), size=(self.hr_size, self.hr_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        mask = (torch.rand(1, self.hr_size, self.hr_size, generator=self._gen) > 0.7).float()
        return {"lr": lr, "hr": hr, "mask": mask, "prompt": self.prompt}

    def _real_sample(self, idx: int) -> dict:
        filename = self.samples[idx]
        hr_path = self.hr_dir / filename
        lr_path = self.lr_dir / filename
        mask_path = self.mask_dir / filename

        hr_img = cv2.imread(str(hr_path))
        lr_img = cv2.imread(str(lr_path))
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if hr_img is None or lr_img is None or mask_img is None:
            for offset in range(1, self.length):
                alt_idx = (idx + offset) % self.length
                return self._real_sample(alt_idx)
            return self._empty_sample(filename)

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1)).float() / 255.0
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).float() / 255.0

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "mask": mask_tensor,
            "prompt": self.prompt,
        }

    def _empty_sample(self, filename: str) -> dict:
        return {
            "lr": torch.zeros(3, self.lr_size, self.lr_size),
            "hr": torch.zeros(3, self.hr_size, self.hr_size),
            "mask": torch.zeros(1, self.hr_size, self.hr_size),
            "prompt": self.prompt,
        }

    def __getitem__(self, idx: int) -> dict:
        if self._mode == "synthetic":
            return self._synthetic_sample()
        return self._real_sample(idx)


def tadisr_collate_fn(batch):
    """
    Custom collate: prompt (str) and filename (str) stay as lists,
    image/mask tensors are stacked.
    """
    elem = batch[0]
    result = {}
    for key in elem:
        val = elem[key]
        if isinstance(val, str):
            result[key] = [d[key] for d in batch]
        elif isinstance(val, torch.Tensor):
            result[key] = torch.stack([d[key] for d in batch])
        else:
            result[key] = [d[key] for d in batch]
    return result


class TADiSRCollateFn:
    """
    Instantiable collate function for the framework's build_dataloader.

    Configured in YAML via:
        data.train.dataloader.collate_fn:
          target: data.tadisr_dataset.TADiSRCollateFn
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, batch):
        return tadisr_collate_fn(batch)


if __name__ == "__main__":
    ds = TADiSRDataset(data_root=None, hr_size=512, length=4)
    print(f"Dataset size: {len(ds)}")
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v!r}")
    batch = tadisr_collate_fn([ds[0], ds[1]])
    print(f"batch prompt: {batch['prompt']}")
    print(f"batch lr: {batch['lr'].shape}")
