from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _safe_ctx_get(ctx, key, default=None):
    try:
        return ctx.get(key, required=False)
    except TypeError:
        try:
            return ctx.get(key)
        except Exception:
            return default
    except Exception:
        return default


def to_log_images(
    value,
    *,
    value_range="auto",
    max_images=4,
):
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        value = [v for v in value if torch.is_tensor(v)]
        if not value:
            return None
        value = torch.stack(value, dim=0)

    if not torch.is_tensor(value):
        return None

    x = value.detach()

    if x.ndim == 3:
        x = x.unsqueeze(0)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    if x.ndim != 4:
        return None

    if x.shape[-1] in {1, 3, 4} and (
        x.shape[1] not in {1, 3, 4} or x.shape[2] not in {1, 3, 4}
    ):
        x = x.permute(0, 3, 1, 2)

    x = x[: int(max_images)]

    if x.shape[1] > 3:
        x = x[:, :3]

    x = x.float().cpu()

    value_range = str(value_range or "auto").lower()

    if value_range in {"-1_1", "minus1_1", "neg1_1"}:
        x = (x + 1.0) / 2.0
    elif value_range in {"0_1", "01"}:
        pass
    elif value_range in {"0_255", "255"}:
        x = x / 255.0
    elif value_range == "auto":
        xmin = float(x.min())
        xmax = float(x.max())

        if xmin < -0.05:
            x = (x + 1.0) / 2.0
        elif xmax > 2.0:
            x = x / 255.0
    else:
        raise ValueError(f"unsupported image value_range: {value_range}")

    return x.clamp(0.0, 1.0)


class MetricLogger:
    def log_metrics(self, metrics: Dict[str, float], step: int):
        raise NotImplementedError

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        raise NotImplementedError

    def flush(self):
        pass

    def close(self):
        self.flush()


class TensorBoardLogger(MetricLogger):
    def __init__(self, cfg: Dict[str, Any], output_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise ImportError(
                "TensorBoard logging requires tensorboard. Install with: pip install tensorboard"
            ) from e

        log_dir = cfg.get("log_dir")
        if log_dir is None:
            log_dir = os.path.join(output_dir, "tensorboard")

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        self.writer.add_images(tag, images, global_step=step, dataformats="NCHW")

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()


class AimLogger(MetricLogger):
    def __init__(self, cfg: Dict[str, Any], output_dir: str):
        try:
            import aim
        except Exception as e:
            raise ImportError("Aim logging requires aim. Install with: pip install aim") from e

        params = dict(cfg.get("params", {}) or {})
        params.setdefault("repo", cfg.get("repo") or os.path.join(output_dir, "aim"))
        for key in ("experiment", "run_hash", "system_tracking_interval"):
            if cfg.get(key) is not None:
                params[key] = cfg[key]

        self.aim = aim
        self.run = aim.Run(**params)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            self.run.track(value, name=key, step=step)

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        for idx, image in enumerate(_images_to_hwc_uint8(images)):
            name = tag if len(images) == 1 else f"{tag}/{idx}"
            self.run.track(self.aim.Image(image), name=name, step=step)

    def close(self):
        close = getattr(self.run, "close", None)
        if close is not None:
            close()


class WandbLogger(MetricLogger):
    def __init__(self, cfg: Dict[str, Any], output_dir: str):
        try:
            import wandb
        except Exception as e:
            raise ImportError("W&B logging requires wandb. Install with: pip install wandb") from e

        params = dict(cfg.get("params", {}) or {})
        params.setdefault("dir", cfg.get("dir") or output_dir)
        for key in ("project", "entity", "name", "group", "job_type", "tags", "notes", "mode"):
            if cfg.get(key) is not None:
                params[key] = cfg[key]

        self.wandb = wandb
        self.run = wandb.init(**params)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        self.wandb.log(metrics, step=step)

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        payload = {
            tag: [self.wandb.Image(image) for image in _images_to_hwc_uint8(images)]
        }
        self.wandb.log(payload, step=step)

    def close(self):
        finish = getattr(self.run, "finish", None)
        if finish is not None:
            finish()


class LoggerCollection:
    def __init__(
        self,
        loggers: Iterable[MetricLogger],
        *,
        image_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.loggers = list(loggers)
        image_cfg = image_cfg or {}
        self.image_enabled = bool(image_cfg.get("enabled", False))
        self.image_every_n_steps = int(image_cfg.get("every_n_steps", 100) or 100)
        self.image_max_images = int(image_cfg.get("max_images", 4) or 4)
        self.image_items = list(image_cfg.get("items", []) or [])

    @property
    def enabled(self) -> bool:
        return bool(self.loggers)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        if not self.loggers:
            return

        for backend in self.loggers:
            backend.log_metrics(metrics, step)

    def log_images_from_context(self, ctx, step: int):
        if not self.loggers or not self.image_enabled:
            return

        if step % self.image_every_n_steps != 0:
            return

        for item in self.image_items:
            item = _plain(item)
            tag = item["tag"]
            key = item["key"]
            value_range = item.get("value_range", "auto")
            max_images = int(item.get("max_images", self.image_max_images) or self.image_max_images)

            value = _safe_ctx_get(ctx, key, default=None)
            images = to_log_images(
                value,
                value_range=value_range,
                max_images=max_images,
            )

            if images is None:
                continue

            for backend in self.loggers:
                backend.log_images(tag, images, step)

    def flush(self):
        for backend in self.loggers:
            backend.flush()

    def close(self):
        for backend in self.loggers:
            backend.close()


def build_loggers(logging_cfg, output_dir: str, is_main_process: bool) -> LoggerCollection:
    logging_cfg = _plain(logging_cfg) or {}

    if not is_main_process:
        return LoggerCollection([])

    backend_cfgs = _normalize_backend_configs(logging_cfg)
    image_cfg = _resolve_image_cfg(logging_cfg, backend_cfgs)
    loggers: List[MetricLogger] = []

    for cfg in backend_cfgs:
        enabled = bool(cfg.get("enabled", cfg.get("enable", True)))
        if not enabled:
            continue

        backend_type = str(cfg.get("type", "")).lower()

        if backend_type in {"tensorboard", "tb"}:
            loggers.append(TensorBoardLogger(cfg, output_dir))
        elif backend_type == "aim":
            loggers.append(AimLogger(cfg, output_dir))
        elif backend_type in {"wandb", "weights_and_biases"}:
            loggers.append(WandbLogger(cfg, output_dir))
        else:
            raise ValueError(f"unsupported logging backend: {backend_type!r}")

    if loggers:
        logger.info("[logging] enabled backends: %s", ", ".join(type(x).__name__ for x in loggers))

    return LoggerCollection(loggers, image_cfg=image_cfg)


def _normalize_backend_configs(logging_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_backends = logging_cfg.get("backends", None)

    if raw_backends is None:
        backends = []
    elif isinstance(raw_backends, dict):
        backends = []
        for name, cfg in raw_backends.items():
            cfg = dict(cfg or {})
            cfg.setdefault("type", name)
            backends.append(cfg)
    else:
        backends = []
        for cfg in raw_backends:
            if isinstance(cfg, str):
                backends.append({"type": cfg})
            else:
                backends.append(dict(cfg or {}))

    legacy_tb_cfg = dict(logging_cfg.get("tensorboard", {}) or {})
    if legacy_tb_cfg and not any(
        str(cfg.get("type", "")).lower() in {"tensorboard", "tb"} for cfg in backends
    ):
        legacy_tb_cfg.setdefault("type", "tensorboard")
        legacy_tb_cfg.setdefault("enabled", legacy_tb_cfg.get("enable", False))
        backends.append(legacy_tb_cfg)

    return backends


def _resolve_image_cfg(
    logging_cfg: Dict[str, Any],
    backend_cfgs: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    image_cfg = dict(logging_cfg.get("images", {}) or {})

    if image_cfg:
        return image_cfg

    backend_cfgs = list(backend_cfgs)
    for cfg in backend_cfgs:
        if str(cfg.get("type", "")).lower() in {"tensorboard", "tb"}:
            image_cfg = dict(cfg.get("images", {}) or {})
            if image_cfg:
                return image_cfg

    for cfg in backend_cfgs:
        image_cfg = dict(cfg.get("images", {}) or {})
        if image_cfg:
            return image_cfg

    return {}


def _images_to_hwc_uint8(images: torch.Tensor):
    x = images.detach().float().cpu().clamp(0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(0, 2, 3, 1).numpy()
    if x.shape[-1] == 1:
        x = x[..., 0]
    return list(x)
