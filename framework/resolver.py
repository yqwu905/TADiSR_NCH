from __future__ import annotations

from typing import Any, Mapping
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_plain(spec: Any) -> Any:
    if isinstance(spec, (DictConfig, ListConfig)):
        return OmegaConf.to_container(spec, resolve=True)
    return spec


def resolve_input(spec: Any, ctx):
    spec = _to_plain(spec)

    if isinstance(spec, str):
        return ctx.get(spec)

    if spec is None or isinstance(spec, (int, float, bool)):
        return spec

    if isinstance(spec, list):
        return [resolve_input(x, ctx) for x in spec]

    if isinstance(spec, tuple):
        return tuple(resolve_input(x, ctx) for x in spec)

    if isinstance(spec, Mapping):
        value_type = spec.get("type")
        if value_type is None:
            raise ValueError(f"dict input spec must contain 'type': {spec}")

        if value_type == "const":
            return spec.get("value")

        if value_type == "ctx":
            return ctx.get(spec["key"])

        if value_type == "ones_like":
            ref = ctx.get(spec["ref"])
            return torch.ones_like(ref)

        if value_type == "zeros_like":
            ref = ctx.get(spec["ref"])
            return torch.zeros_like(ref)

        if value_type == "full_like":
            ref = ctx.get(spec["ref"])
            return torch.full_like(ref, spec["value"])

        if value_type == "randn_like":
            ref = ctx.get(spec["ref"])
            return torch.randn_like(ref)

        if value_type == "detach":
            ref = ctx.get(spec["ref"])
            return ref.detach()

        if value_type == "cast":
            ref = resolve_input(spec["value"], ctx)
            dtype = spec.get("dtype")
            if dtype is not None:
                dtype_obj = getattr(torch, dtype) if isinstance(dtype, str) else dtype
                ref = ref.to(dtype=dtype_obj)
            device_ref = spec.get("device_ref")
            if device_ref is not None:
                ref = ref.to(device=ctx.get(device_ref).device)
            return ref

        if value_type == "getattr":
            obj = resolve_input(spec["object"], ctx)
            return getattr(obj, spec["name"])

        raise ValueError(f"Unknown input spec type: {value_type}")

    raise TypeError(f"Unsupported input spec: {spec!r}")


def resolve_kwargs(inputs_cfg: Mapping[str, Any], ctx) -> dict:
    inputs_cfg = _to_plain(inputs_cfg) or {}
    return {name: resolve_input(spec, ctx) for name, spec in inputs_cfg.items()}
