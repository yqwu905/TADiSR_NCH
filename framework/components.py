from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import fnmatch
import logging
from typing import Any, Dict, Iterable, Mapping, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .instantiate import instantiate
from .distributed import (
    DistState,
    fully_shard_module,
    register_fsdp2_forward_methods,
    unwrap_model,
    wrap_ddp,
)


logger = logging.getLogger(__name__)


def resolve_submodule(module, path: str):
    obj = module
    for part in str(path).split("."):
        obj = getattr(obj, part)
    return obj


@dataclass
class ComponentEntry:
    name: str
    module: Any
    cfg: Mapping[str, Any]
    trainable_flags: Dict[str, bool]
    parallel_strategy: str = "none"
    gradient_checkpointing: bool = False


class ComponentManager:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg or {})
        self.entries: Dict[str, ComponentEntry] = {}

    def build_all(self, max_workers: Optional[int] = None):
        items = list(self.cfg.items())
        if not items:
            return self

        def _build_one(name, c):
            module = instantiate(c)

            ckpt = c.get("checkpoint")
            if ckpt:
                self.load_checkpoint(
                    module,
                    ckpt,
                    name,
                    strict=c.get("strict", True),
                    save_submodule=c.get("save_submodule"),
                )

            module = self.apply_train_policy(module, c.get("train", {"strategy": "full"}))
            trainable_flags = self._snapshot_trainable_flags(module)

            return ComponentEntry(
                name=name,
                module=module,
                cfg=c,
                trainable_flags=trainable_flags,
            )

        if not max_workers or max_workers <= 1 or len(items) <= 1:
            for name, c in items:
                self.entries[name] = _build_one(name, c)
            return self

        worker_count = max(1, min(int(max_workers), len(items)))
        results: Dict[str, ComponentEntry] = {}
        first_exc: Optional[BaseException] = None

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="comp-build",
        ) as ex:
            future_to_name = {
                ex.submit(_build_one, name, c): name for name, c in items
            }
            for fut in as_completed(future_to_name):
                try:
                    entry = fut.result()
                    results[entry.name] = entry
                except BaseException as exc:
                    if first_exc is None:
                        first_exc = exc

        if first_exc is not None:
            raise first_exc

        for name, _ in items:
            if name not in results:
                raise RuntimeError(f"component {name!r} was not built")
            self.entries[name] = results[name]

        return self

    def _snapshot_trainable_flags(self, module) -> Dict[str, bool]:
        if not hasattr(module, "named_parameters"):
            return {}
        return {name: p.requires_grad for name, p in module.named_parameters()}

    def load_checkpoint(
        self,
        module,
        path: str,
        name: str,
        strict: bool = True,
        save_submodule: Optional[str] = None,
    ):
        state = torch.load(path, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        target = module
        if save_submodule:
            try:
                target = resolve_submodule(module, save_submodule)
            except AttributeError as e:
                raise AttributeError(
                    f"component {name} save_submodule {save_submodule!r} could not be "
                    f"resolved on {type(module).__name__}: {e}"
                ) from e

        if not hasattr(target, "load_state_dict"):
            raise TypeError(f"component {target} has no load_state_dict")

        result = target.load_state_dict(state, strict=strict)

        missing = getattr(result, "missing_keys", None)
        unexpected = getattr(result, "unexpected_keys", None)

        if missing is None or unexpected is None:
            try:
                missing, unexpected = result
            except Exception:
                missing, unexpected = [], []

        logger.info("%s load from %s, result:", name, path)
        log_missing = logger.warning if missing else logger.info
        log_unexpected = logger.warning if unexpected else logger.info
        log_missing("Missing: %s", missing)
        log_unexpected("Unexpected: %s", unexpected)

    def apply_train_policy(self, module, train_cfg):
        if train_cfg is None:
            train_cfg = {"strategy": "full"}

        if isinstance(train_cfg, str):
            train_cfg = {"strategy": train_cfg}

        train_cfg = dict(train_cfg)
        strategy = train_cfg.get("strategy", "full")

        if not hasattr(module, "parameters"):
            return module

        if strategy == "frozen":
            module.requires_grad_(False)
            return module

        if strategy == "full":
            module.requires_grad_(True)
            return module

        if strategy == 'keep':
            return module

        raise ValueError(f"unknown train strategy: {strategy}")

    _GC_METHOD_CANDIDATES = (
        "gradient_checkpointing_enable",
        "enable_gradient_checkpointing",
    )

    def apply_gradient_checkpointing(self):
        for name, entry in self.entries.items():
            gc_cfg = entry.cfg.get("gradient_checkpointing")
            if not gc_cfg:
                continue

            module = entry.module
            if not isinstance(module, nn.Module):
                logger.warning(
                    "component %s has gradient_checkpointing configured but is not an nn.Module; skipping",
                    name,
                )
                continue

            enabled, method_name, method_kwargs = self._resolve_gc_cfg(gc_cfg)
            if not enabled:
                continue

            self._enable_gc_on_module(name, module, method_name, method_kwargs)
            entry.gradient_checkpointing = True

        return self

    @staticmethod
    def _resolve_gc_cfg(gc_cfg):
        if isinstance(gc_cfg, bool):
            return gc_cfg, None, {}

        gc_cfg = dict(gc_cfg)
        enabled = bool(gc_cfg.get("enabled", True))
        method_name = gc_cfg.get("method")
        method_kwargs = dict(gc_cfg.get("method_kwargs", {}) or {})
        return enabled, method_name, method_kwargs

    @staticmethod
    def _enable_gc_on_module(name, module, method_name, method_kwargs):
        if method_name is not None:
            fn = getattr(module, method_name, None)
            if not callable(fn):
                raise ValueError(
                    f"component {name} gradient_checkpointing.method "
                    f"{method_name!r} not found on {type(module).__name__}"
                )
            fn(**method_kwargs)
            logger.info(
                "[component %s] gradient checkpointing enabled via %s",
                name,
                method_name,
            )
            return

        for candidate in ComponentManager._GC_METHOD_CANDIDATES:
            fn = getattr(module, candidate, None)
            if callable(fn):
                fn(**method_kwargs)
                logger.info(
                    "[component %s] gradient checkpointing enabled via %s",
                    name,
                    candidate,
                )
                return

        raise ValueError(
            f"component {name} has gradient_checkpointing enabled but implements none of "
            f"{list(ComponentManager._GC_METHOD_CANDIDATES)}; implement one or set "
            f"gradient_checkpointing.method explicitly"
        )

    def to(self, device):
        for entry in self.entries.values():
            if hasattr(entry.module, "to"):
                entry.module.to(device)
        return self

    def apply_parallel(
        self,
        dist_state: DistState,
        distributed_cfg: Optional[Mapping[str, Any]] = None,
    ):
        distributed_cfg = dict(distributed_cfg or {})

        if not dist_state.enabled:
            self.to(dist_state.device)
            return self

        if dist_state.strategy == "ddp":
            self.to(dist_state.device)
            self._apply_ddp_parallel(dist_state, dict(distributed_cfg.get("ddp", {}) or {}))
            return self

        if dist_state.strategy != "fsdp2":
            raise ValueError(f"unsupported distributed strategy: {dist_state.strategy}")

        fsdp_global_cfg = dict(distributed_cfg.get("fsdp", {}) or {})
        ddp_global_cfg = dict(distributed_cfg.get("ddp", {}) or {})

        for name, entry in self.entries.items():
            module = entry.module

            if not isinstance(module, nn.Module):
                entry.parallel_strategy = "none"
                continue

            strategy = self.resolve_parallel_strategy(entry, fsdp_global_cfg)
            entry.parallel_strategy = strategy

            if strategy == "fsdp2":
                fsdp_cfg = dict(fsdp_global_cfg)
                fsdp_cfg.update(dict(entry.cfg.get("fsdp", {}) or {}))
                self._apply_fsdp2_to_entry(entry, dist_state, fsdp_cfg)
            elif strategy == "ddp":
                module.to(dist_state.device)
                component_ddp_cfg = dict(ddp_global_cfg)
                component_ddp_cfg.update(dict(entry.cfg.get("ddp", {}) or {}))
                entry.module = wrap_ddp(
                    module,
                    dist_state=dist_state,
                    ddp_cfg=component_ddp_cfg,
                )
            elif strategy == "replicated":
                self._validate_replicated_entry(entry)
                module.to(dist_state.device)
                self._broadcast_module_state(module)
            elif strategy == "none":
                module.to(dist_state.device)
            else:
                raise ValueError(f"unsupported parallel strategy for {name}: {strategy}")

        return self

    def _apply_ddp_parallel(self, dist_state: DistState, ddp_cfg: Mapping[str, Any]):
        for name, entry in self.entries.items():
            module = entry.module

            if not isinstance(module, nn.Module):
                entry.parallel_strategy = "none"
                continue

            has_trainable = self._has_trainable_params(module)
            if not has_trainable:
                entry.parallel_strategy = "replicated"
                continue

            component_ddp_cfg = dict(ddp_cfg)
            component_ddp_cfg.update(dict(entry.cfg.get("ddp", {}) or {}))

            entry.module = wrap_ddp(
                module,
                dist_state=dist_state,
                ddp_cfg=component_ddp_cfg,
            )
            entry.parallel_strategy = "ddp"

        return self

    def wrap_ddp(self, dist_state: DistState, ddp_cfg=None):
        distributed_cfg = {"ddp": dict(ddp_cfg or {})}
        return self.apply_parallel(dist_state, distributed_cfg=distributed_cfg)

    def resolve_parallel_strategy(
        self,
        entry: ComponentEntry,
        fsdp_global_cfg: Optional[Mapping[str, Any]] = None,
    ) -> str:
        fsdp_global_cfg = dict(fsdp_global_cfg or {})
        parallel_cfg = dict(entry.cfg.get("parallel", {}) or {})
        explicit = parallel_cfg.get("strategy")

        if explicit is not None:
            return self._normalize_parallel_strategy(explicit)

        fsdp_cfg = dict(entry.cfg.get("fsdp", {}) or {})
        if bool(fsdp_cfg.get("enabled", False)):
            return "fsdp2"

        if self._has_trainable_params(entry.module):
            default_strategy = fsdp_global_cfg.get("default_non_fsdp_trainable", "ddp")
            return self._normalize_parallel_strategy(default_strategy)

        return "replicated"

    @staticmethod
    def _normalize_parallel_strategy(value) -> str:
        strategy = str(value).lower().replace("-", "_")
        aliases = {
            "fsdp": "fsdp2",
            "fsdp2": "fsdp2",
            "fully_sharded": "fsdp2",
            "ddp": "ddp",
            "distributed_data_parallel": "ddp",
            "replica": "replicated",
            "replicated": "replicated",
            "none": "replicated",
        }
        if strategy not in aliases:
            raise ValueError(f"unsupported component parallel strategy: {value!r}")
        return aliases[strategy]

    @staticmethod
    def _has_trainable_params(module) -> bool:
        if not hasattr(module, "parameters"):
            return False
        return any(p.requires_grad for p in module.parameters())

    def _validate_replicated_entry(self, entry: ComponentEntry):
        if self._has_trainable_params(entry.module):
            allow = bool(
                dict(entry.cfg.get("parallel", {}) or {}).get(
                    "allow_trainable_replicated",
                    False,
                )
            )
            if not allow:
                raise ValueError(
                    f"component {entry.name} uses replicated parallel strategy "
                    "but still has trainable parameters; use ddp/fsdp2 or set "
                    "parallel.allow_trainable_replicated=true explicitly"
                )

    def _apply_fsdp2_to_entry(
        self,
        entry: ComponentEntry,
        dist_state: DistState,
        fsdp_cfg: Mapping[str, Any],
    ):
        module = entry.module
        if not isinstance(module, nn.Module):
            return

        wrap_patterns = list(fsdp_cfg.get("wrap_modules", []) or [])
        if wrap_patterns:
            wrapped_modules = self._resolve_fsdp_wrap_modules(
                module,
                wrap_patterns,
                component_name=entry.name,
            )
            wrap_submodules = [submodule for _, submodule in wrapped_modules]
        else:
            wrap_submodules = self._resolve_fsdp_wrap_modules_from_method(module)
            if not wrap_submodules:
                logger.warning(
                    "component %s enabled fsdp2 but neither fsdp.wrap_modules nor "
                    "get_fsdp_wrap_module_list() provided any wrap targets; "
                    "only the top-level module will be sharded.",
                    entry.name,
                )

        for submodule in wrap_submodules:
            fully_shard_module(submodule, dist_state, fsdp_cfg)

        sharded_module = fully_shard_module(module, dist_state, fsdp_cfg)
        entry.module = module if sharded_module is None else sharded_module
        register_fsdp2_forward_methods(
            entry.module,
            fsdp_cfg.get("forward_methods", []) or [],
        )

    @staticmethod
    def _resolve_fsdp_wrap_modules(
        module: nn.Module,
        patterns: Iterable[str],
        *,
        component_name: str,
    ):
        patterns = list(patterns or [])
        if not patterns:
            return []

        named_modules = [(name, submodule) for name, submodule in module.named_modules() if name]
        selected = []
        missing = []

        for pattern in patterns:
            matches = [
                item
                for item in named_modules
                if fnmatch.fnmatchcase(item[0], str(pattern))
            ]
            if not matches:
                missing.append(str(pattern))
            selected.extend(matches)

        if missing:
            available = ", ".join(name for name, _ in named_modules[:20])
            raise ValueError(
                f"component {component_name} fsdp.wrap_modules matched no modules: "
                f"{missing}. Available module paths include: {available}"
            )

        deduped = {}
        for name, submodule in selected:
            deduped[name] = submodule

        return sorted(deduped.items(), key=lambda item: item[0].count("."), reverse=True)

    @staticmethod
    def _resolve_fsdp_wrap_modules_from_method(module: nn.Module):
        getter = getattr(module, "get_fsdp_wrap_module_list", None)
        if not callable(getter):
            return []

        returned = getter()
        if not returned:
            return []

        name_by_id = {id(sub): name for name, sub in module.named_modules() if name}

        seen = set()
        keyed = []
        for sub in returned:
            if not isinstance(sub, nn.Module) or sub is module:
                continue
            sid = id(sub)
            if sid in seen:
                continue
            seen.add(sid)
            name = name_by_id.get(sid, "")
            depth = name.count(".") if name else -1
            keyed.append((depth, sub))

        keyed.sort(key=lambda item: item[0], reverse=True)
        return [sub for _, sub in keyed]

    @staticmethod
    def _broadcast_module_state(module: nn.Module):
        if not (dist.is_available() and dist.is_initialized()):
            return

        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.detach(), src=0)

    def set_phase_state(self, trainable=None, frozen=None, modes=None):
        trainable = set(trainable or [])
        frozen = set(frozen or [])
        modes = dict(modes or {})

        for name, entry in self.entries.items():
            module = entry.module

            if hasattr(module, "requires_grad_"):
                if name in trainable:
                    self._restore_component_trainable_flags(name)
                elif name in frozen:
                    module.requires_grad_(False)

            mode = modes.get(name)

            if mode is None:
                if name in trainable and hasattr(module, "train"):
                    module.train()
                elif name in frozen and hasattr(module, "eval"):
                    module.eval()
            else:
                if mode == "train" and hasattr(module, "train"):
                    module.train()
                elif mode == "eval" and hasattr(module, "eval"):
                    module.eval()
                else:
                    raise ValueError(f"invalid mode for {name}: {mode}")

    def _restore_component_trainable_flags(self, name: str):
        entry = self.entries[name]
        module = unwrap_model(entry.module)

        if not hasattr(module, "named_parameters"):
            return

        for pname, p in module.named_parameters():
            p.requires_grad_(entry.trainable_flags.get(pname, False))

    def trainable_parameters(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)
        params = []

        for name in names:
            module = unwrap_model(self.entries[name].module)

            if hasattr(module, "parameters"):
                params.extend([p for p in module.parameters() if p.requires_grad])

        return params

    def all_parameters(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)
        params = []

        for name in names:
            module = unwrap_model(self.entries[name].module)

            if hasattr(module, "parameters"):
                params.extend(list(module.parameters()))

        return params

    def named_trainable_parameters(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)

        for cname in names:
            module = unwrap_model(self.entries[cname].module)

            if hasattr(module, "named_parameters"):
                for pname, p in module.named_parameters():
                    if p.requires_grad:
                        yield f"{cname}.{pname}", p

    def unwrap(self, name: str):
        return unwrap_model(self.entries[name].module)

    def unwrapped_items(self):
        return ((name, unwrap_model(entry.module)) for name, entry in self.entries.items())

    def get(self, name: str):
        return self.entries[name].module

    def __getitem__(self, name: str):
        return self.get(name)

    def items(self):
        return ((name, entry.module) for name, entry in self.entries.items())

    def set_initial_modes(self):
        for name, entry in self.entries.items():
            mode = entry.cfg.get("mode")
            module = entry.module

            if mode == "train" and hasattr(module, "train"):
                module.train()
            elif mode == "eval" and hasattr(module, "eval"):
                module.eval()

    def parameter_summary(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)
        summaries = {}

        for name in names:
            module = unwrap_model(self.entries[name].module)

            if not hasattr(module, "parameters"):
                summaries[name] = {
                    "total": 0,
                    "trainable": 0,
                    "frozen": 0,
                    "trainable_ratio": 0.0,
                }
                continue

            total = 0
            trainable = 0

            for p in module.parameters():
                n = p.numel()
                total += n
                if p.requires_grad:
                    trainable += n

            frozen = total - trainable
            ratio = trainable / total if total > 0 else 0.0

            summaries[name] = {
                "total": total,
                "trainable": trainable,
                "frozen": frozen,
                "trainable_ratio": ratio,
            }

        return summaries

    @staticmethod
    def _format_param_count(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.3f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.3f}M"
        if n >= 1_000:
            return f"{n / 1_000:.3f}K"
        return str(n)

    def print_parameter_summary(self, component_names: Optional[Iterable[str]] = None):
        summaries = self.parameter_summary(component_names)

        logger.info("")
        logger.info("[component parameters]")

        total_all = 0
        trainable_all = 0

        for name, s in summaries.items():
            total = s["total"]
            trainable = s["trainable"]
            frozen = s["frozen"]
            ratio = s["trainable_ratio"]

            total_all += total
            trainable_all += trainable

            logger.info(
                f"  {name}: "
                f"trainable={self._format_param_count(trainable)} "
                f"({trainable:,}), "
                f"total={self._format_param_count(total)} "
                f"({total:,}), "
                f"frozen={self._format_param_count(frozen)} "
                f"({frozen:,}), "
                f"ratio={ratio * 100:.4f}%",
            )

        frozen_all = total_all - trainable_all
        ratio_all = trainable_all / total_all if total_all > 0 else 0.0

        logger.info(
            f"  TOTAL: "
            f"trainable={self._format_param_count(trainable_all)} "
            f"({trainable_all:,}), "
            f"total={self._format_param_count(total_all)} "
            f"({total_all:,}), "
            f"frozen={self._format_param_count(frozen_all)} "
            f"({frozen_all:,}), "
            f"ratio={ratio_all * 100:.4f}%\n",
        )
