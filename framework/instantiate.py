import importlib
from typing import Any, Mapping
from omegaconf import DictConfig, OmegaConf


def to_container(obj: Any) -> Any:
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def locate(target: str):
    if not isinstance(target, str) or "." not in target:
        raise ValueError(f"target must be a full import path, got: {target!r}")
    module_name, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def instantiate(cfg: Mapping[str, Any], **extra_kwargs):
    cfg = to_container(cfg)
    if cfg is None:
        return None
    if "target" not in cfg:
        raise KeyError(f"missing 'target' in config: {cfg}")
    cls_or_fn = locate(cfg["target"])
    params = dict(cfg.get("params", {}) or {})
    params.update(extra_kwargs)
    return cls_or_fn(**params)
