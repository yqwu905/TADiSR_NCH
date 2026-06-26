from __future__ import annotations

import os
from typing import Any, Mapping
import torch
from omegaconf import DictConfig, OmegaConf

from framework.registry import register_op
from framework.resolver import resolve_kwargs, resolve_input
from framework.instantiate import locate


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _write_outputs(ctx, result, outputs):
    outputs = _plain(outputs) or {}
    if not outputs:
        return

    if len(outputs) == 1 and "_" in outputs:
        ctx.set(outputs["_"], result)
        return

    if isinstance(result, dict):
        for local_key, ctx_key in outputs.items():
            ctx.set(ctx_key, result[local_key])
        return

    if isinstance(result, (tuple, list)):
        for local_key, ctx_key in outputs.items():
            idx = int(local_key)
            ctx.set(ctx_key, result[idx])
        return

    raise TypeError(f"cannot map outputs from result type {type(result)} with outputs={outputs}")


@register_op("call")
class CallOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        component_name = self.cfg.get("component")
        method_name = self.cfg.get("method", "forward")
        no_grad = bool(self.cfg.get("no_grad", False))
        detach_inputs = set(self.cfg.get("detach_inputs", []) or [])

        if component_name is not None:
            obj = components[component_name]
            fn = obj if method_name == "forward" else getattr(obj, method_name)
        else:
            fn = locate(self.cfg["function"])

        kwargs = resolve_kwargs(self.cfg.get("inputs", {}) or {}, ctx)
        for name in detach_inputs:
            if name in kwargs and torch.is_tensor(kwargs[name]):
                kwargs[name] = kwargs[name].detach()

        if no_grad:
            with torch.no_grad():
                result = fn(**kwargs)
        else:
            result = fn(**kwargs)

        _write_outputs(ctx, result, self.cfg.get("outputs", {}))


@register_op("make_tensor")
class MakeTensorOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        mode = self.cfg["mode"]
        ref = ctx.get(self.cfg["ref"])
        if mode == "ones_like":
            value = torch.ones_like(ref)
        elif mode == "zeros_like":
            value = torch.zeros_like(ref)
        elif mode == "full_like":
            value = torch.full_like(ref, self.cfg["value"])
        elif mode == "randn_like":
            value = torch.randn_like(ref)
        else:
            raise ValueError(f"unknown make_tensor mode: {mode}")
        ctx.set(self.cfg["output"], value)


@register_op("set_value")
class SetValueOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        value = resolve_input(self.cfg["value"], ctx)
        ctx.set(self.cfg["output"], value)


@register_op("detach")
class DetachOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        value = ctx.get(self.cfg["input"])
        ctx.set(self.cfg["output"], value.detach())

@register_op("save_image")
class SaveImageOp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.inputs = dict(cfg.get("inputs", {}) or {})
        self.params = dict(cfg.get("params", {}) or {})

    def __call__(self, ctx, components):
        image = resolve_input(self.inputs["image"], ctx)

        output_dir = self.params.get("output_dir", "outputs/inference")
        filename_key = self.inputs.get("filename")
        path_key = self.inputs.get("path")

        os.makedirs(output_dir, exist_ok=True)

        if torch.is_tensor(image):
            image = image.detach().float().cpu()
        else:
            raise TypeError(f"save_image expects tensor image, got {type(image)}")

        # image: [C,H,W] -> [1,C,H,W]
        if image.ndim == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]

        filenames = None

        if filename_key is not None:
            filenames = resolve_input(filename_key, ctx)
            filenames = _ensure_list(filenames)
        elif path_key is not None:
            paths = resolve_input(path_key, ctx)
            paths = _ensure_list(paths)
            filenames = [_basename_without_ext(p) + ".png" for p in paths]
        else:
            global_step = int(ctx.get("global_step", 0, required=False) or 0)
            filenames = [f"{global_step:08d}_{i:04d}.png" for i in range(batch_size)]

        for i in range(batch_size):
            filename = filenames[i] if i < len(filenames) else f"{i:04d}.png"
            save_path = os.path.join(output_dir, filename)

            img = image[i].clamp(0, 1)
            save_image(img, save_path)
