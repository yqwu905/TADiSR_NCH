from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from framework.instantiate import instantiate
from framework.resolver import resolve_input


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


class LossWrapper(nn.Module):
    def __init__(
        self,
        name: str,
        loss_fn,
        inputs: Mapping[str, Any],
        weight: float = 1.0,
        enable: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.name = name
        self.inputs = dict(inputs or {})
        self.weight = float(weight)
        self.enable_cfg = dict(enable or {})

        if isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            self.loss_fn = loss_fn

    def enabled(self, ctx) -> bool:
        global_step = int(ctx.get("global_step", 0))

        after_step = self.enable_cfg.get("after_step")
        if after_step is not None and global_step < int(after_step):
            return False

        before_step = self.enable_cfg.get("before_step")
        if before_step is not None and global_step >= int(before_step):
            return False

        every_n_steps = self.enable_cfg.get("every_n_steps")
        if every_n_steps is not None and int(every_n_steps) > 1:
            if global_step % int(every_n_steps) != 0:
                return False

        return True

    def forward(self, ctx):
        kwargs = {
            arg_name: resolve_input(input_spec, ctx)
            for arg_name, input_spec in self.inputs.items()
        }

        result = self.loss_fn(**kwargs)

        if torch.is_tensor(result):
            loss = result
            metrics = {}
        elif isinstance(result, dict):
            if "loss" not in result:
                raise KeyError(f"loss {self.name} returned dict but missing key 'loss'")
            loss = result["loss"]
            metrics = {k: v for k, v in result.items() if k != "loss"}
        else:
            raise TypeError(
                f"loss {self.name} should return Tensor or dict, got {type(result)}"
            )

        loss = loss * self.weight

        clean_metrics = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                clean_metrics[f"{self.name}/{k}"] = v.detach()
            else:
                clean_metrics[f"{self.name}/{k}"] = v

        clean_metrics[f"{self.name}/loss"] = loss.detach()

        return loss, clean_metrics


def build_losses(losses_cfg):
    losses_cfg = _plain(losses_cfg) or {}
    losses = {}

    for name, cfg in losses_cfg.items():
        loss_fn = instantiate(cfg)
        losses[name] = LossWrapper(
            name=name,
            loss_fn=loss_fn,
            inputs=cfg.get("inputs", {}),
            weight=cfg.get("weight", 1.0),
            enable=cfg.get("enable", {}),
        )

    return losses