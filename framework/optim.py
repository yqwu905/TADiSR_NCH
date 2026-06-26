from __future__ import annotations

from typing import Any, Dict, Mapping
import torch
from omegaconf import DictConfig, OmegaConf
from .instantiate import instantiate


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def collect_params(param_cfg: Mapping[str, Any], components):
    param_cfg = _plain(param_cfg) or {}
    include = param_cfg.get("include", [])
    params = []
    for item in include:
        if isinstance(item, str) and item.endswith(".trainable"):
            comp_name = item.rsplit(".", 1)[0]
            params.extend(components.trainable_parameters([comp_name]))
        elif isinstance(item, str):
            params.extend(components.trainable_parameters([item]))
        else:
            raise ValueError(f"unsupported params include item: {item}")
    if not params:
        raise ValueError(f"optimizer got no parameters from config: {param_cfg}")
    return params


def build_optimizers(cfg, components):
    cfg = _plain(cfg) or {}
    optimizers = {}
    for name, ocfg in cfg.items():
        ocfg = dict(ocfg)
        opt_type = ocfg.get("type", "adamw").lower()
        params = collect_params(ocfg.get("params", {}), components)
        lr = ocfg.get("lr", 1e-4)
        weight_decay = ocfg.get("weight_decay", 0.0)
        kwargs = dict(ocfg.get("kwargs", {}) or {})

        if opt_type == "adamw":
            optimizers[name] = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif opt_type == "adam":
            optimizers[name] = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif "target" in ocfg:
            optimizers[name] = instantiate(ocfg, params=params)
        else:
            raise ValueError(f"unknown optimizer type: {opt_type}")
    return optimizers


def build_schedulers(cfg, optimizers):
    cfg = _plain(cfg) or {}
    schedulers = {}
    for name, scfg in cfg.items():
        scfg = dict(scfg)
        optimizer = optimizers[scfg["optimizer"]]
        stype = scfg.get("type", "none").lower()
        if stype == "none":
            continue
        if stype == "cosine":
            total_steps = int(scfg["total_steps"])
            warmup_steps = int(scfg.get("warmup_steps", 0))

            def lr_lambda(step, warmup_steps=warmup_steps, total_steps=total_steps):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                import math
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif "target" in scfg:
            schedulers[name] = instantiate(scfg, optimizer=optimizer)
        else:
            raise ValueError(f"unknown scheduler type: {stype}")
    return schedulers
