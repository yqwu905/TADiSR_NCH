"""Lightweight image dataset for TADiSR inference.

The training dataset expects FTSR HR/LR/mask triplets. Inference only needs
one degraded RGB image plus the fixed text prompt used by the offline
embedding cache, while still exposing a batch shape compatible with the
TADiSR phase inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.tadisr_dataset import DEFAULT_PROMPT, _parse_target_size


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

_INTERPOLATION = {
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def _is_image_file(path: Path, extensions: Iterable[str]) -> bool:
    return path.is_file() and path.suffix.lower() in set(extensions)


def _discover_images(input_path: str, recursive: bool, extensions: Sequence[str]) -> list[Path]:
    root = Path(input_path).expanduser()
    if root.is_file():
        if root.suffix.lower() not in set(extensions):
            raise ValueError(f"input file is not a supported image: {root}")
        return [root]

    if not root.is_dir():
        raise FileNotFoundError(f"input path does not exist: {root}")

    pattern = "**/*" if recursive else "*"
    images = sorted(p for p in root.glob(pattern) if _is_image_file(p, extensions))
    if not images:
        raise FileNotFoundError(f"no images found under {root}")
    return images


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported channel count for {path}: {image.shape}")

    return image


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


class InferenceImageDataset(Dataset):
    """Read a single image or a directory of images for inference.

    Each sample contains:
      - ``lr``: image tensor ``[3, H, W]`` in ``[0, 1]``
      - ``hr``: alias of ``lr`` for configs that still contain GT-only ops
      - ``mask``: zero mask ``[1, H, W]`` for compatibility
      - ``prompt``: fixed prompt for ``EmbeddingDB.get_embedding``
      - path/name metadata used by the inference script when saving outputs
    """

    def __init__(
        self,
        input_path: str,
        size: Optional[int | str | Sequence[int]] = None,
        prompt: str = DEFAULT_PROMPT,
        recursive: bool = True,
        interpolation: str = "bicubic",
        square_policy: str = "resize",
        return_meta: bool = True,
        extensions: Optional[Sequence[str]] = None,
    ):
        self.prompt = prompt
        self.return_meta = bool(return_meta)
        self.square_policy = str(square_policy)
        self.extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (extensions or sorted(IMAGE_EXTENSIONS))
        )
        self.paths = _discover_images(input_path, bool(recursive), self.extensions)

        self.size = _parse_target_size(size) if size is not None else None
        interp_key = str(interpolation).lower()
        if interp_key not in _INTERPOLATION:
            raise ValueError(
                f"unknown interpolation {interpolation!r}; "
                f"expected one of {sorted(_INTERPOLATION)}"
            )
        self.interpolation = _INTERPOLATION[interp_key]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        image = _load_rgb(path)
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])

        if self.size is not None:
            target_h, target_w = self.size
            image = cv2.resize(
                image,
                (int(target_w), int(target_h)),
                interpolation=self.interpolation,
            )

        image = _to_float01(image)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.zeros(1, tensor.shape[-2], tensor.shape[-1], dtype=tensor.dtype)

        sample = {
            "lr": tensor,
            "hr": tensor.clone(),
            "mask": mask,
            "prompt": self.prompt,
        }
        if self.return_meta:
            sample.update(
                {
                    "path": str(path),
                    "filename": path.name,
                    "stem": path.stem,
                    "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
                    "input_size": torch.tensor(
                        [tensor.shape[-2], tensor.shape[-1]], dtype=torch.long
                    ),
                }
            )
        return sample


def inference_collate_fn(batch: list[dict]) -> dict:
    elem = batch[0]
    out = {}
    for key, value in elem.items():
        if torch.is_tensor(value):
            out[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(value, str):
            out[key] = [sample[key] for sample in batch]
        else:
            out[key] = [sample[key] for sample in batch]
    return out


class InferenceCollateFn:
    def __init__(self, **kwargs):
        pass

    def __call__(self, batch: list[dict]) -> dict:
        return inference_collate_fn(batch)
