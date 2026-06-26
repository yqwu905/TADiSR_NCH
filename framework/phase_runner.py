from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Mapping
import torch
from omegaconf import DictConfig, OmegaConf

from .registry import get_op
from .ops import *


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _metric_to_float(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().float().item()
        return value.detach().float().mean().item()
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _iter_ops(ops_cfg):
    ops_cfg = _plain(ops_cfg) or []

    if isinstance(ops_cfg, Mapping):
        if "type" in ops_cfg:
            raise ValueError(
                "phase ops should be a list or a mapping of op_name -> op config; "
                "got a single op mapping with a top-level 'type'"
            )

        for op_name, op_cfg in ops_cfg.items():
            op_cfg = dict(op_cfg or {})
            op_cfg.setdefault("name", op_name)
            op_cfg.setdefault("type", "call")
            yield op_cfg
        return

    for op_cfg in ops_cfg:
        op_cfg = dict(op_cfg or {})
        op_cfg.setdefault("type", "call")
        op_cfg.setdefault("name", op_cfg["type"])
        yield op_cfg


def _autocast_context(device, mixed_precision: str):
    mixed_precision = str(mixed_precision or "no").lower()
    if mixed_precision in {"no", "none", "fp32", "float32"}:
        return nullcontext()
    if mixed_precision in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif mixed_precision in {"fp16", "float16"}:
        dtype = torch.float16
    else:
        raise ValueError(f"unsupported mixed_precision: {mixed_precision}")
    return torch.autocast(device_type=device.type, dtype=dtype)


class PhaseRunner:
    def __init__(
        self,
        components,
        optimizers,
        schedulers,
        losses,
        device,
        mixed_precision="no",
        timer=None,
    ):
        self.components = components
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.losses = losses
        self.device = device
        self.mixed_precision = mixed_precision
        self.timer = timer

    def _time(self, name: str):
        if self.timer is None:
            return nullcontext()
        return self.timer.time(name)

    def run(self, ctx, phase_cfg, *, do_zero_grad=True, do_step=True, loss_scale=1.0):
        phase_cfg = _plain(phase_cfg)
        name = phase_cfg["name"]

        metrics: Dict[str, Any] = {}

        with self._time(f"phase/{name}/total"):
            with self._time(f"phase/{name}/set_phase_state"):
                self.components.set_phase_state(
                    trainable=phase_cfg.get("trainable", []),
                    frozen=phase_cfg.get("frozen", []),
                    modes=phase_cfg.get("modes", {}),
                )

            every_n_steps = phase_cfg.get("every_n_steps")
            if every_n_steps is not None:
                step = int(ctx.get("global_step", 0, required=False) or 0)
                if step % int(every_n_steps) != 0:
                    return {}

            if do_zero_grad:
                with self._time(f"phase/{name}/zero_grad"):
                    for opt_name in phase_cfg.get("zero_grad", []) or []:
                        with self._time(f"optimizer/{opt_name}/zero_grad"):
                            self.optimizers[opt_name].zero_grad(set_to_none=True)

            total_loss = None

            with self._time(f"phase/{name}/forward_and_loss"):
                with _autocast_context(self.device, self.mixed_precision):
                    for op_cfg in _iter_ops(phase_cfg.get("ops", [])):
                        op_type = op_cfg["type"]
                        op_name = op_cfg.get("name", op_type)
                        component_name = op_cfg.get("component")

                        op = get_op(op_type)(op_cfg)

                        with self._time(f"phase/{name}/op/{op_name}"):
                            if component_name is not None:
                                with self._time(f"component/{component_name}/op/{op_name}"):
                                    op(ctx, self.components)
                            else:
                                with self._time(f"op_type/{op_type}/{op_name}"):
                                    op(ctx, self.components)

                    for loss_name in phase_cfg.get("losses", []) or []:
                        loss = self.losses[loss_name]

                        with self._time(f"phase/{name}/loss/{loss_name}/enabled_check"):
                            enabled = loss.enabled(ctx)

                        if not enabled:
                            continue

                        with self._time(f"phase/{name}/loss/{loss_name}"):
                            value, loss_metrics = loss(ctx)

                        total_loss = value if total_loss is None else total_loss + value
                        metrics.update(loss_metrics)

            if total_loss is None:
                return metrics

            metrics[f"{name}/total_loss"] = total_loss.detach()

            if phase_cfg.get("backward", True):
                with self._time(f"phase/{name}/backward"):
                    (total_loss * float(loss_scale)).backward()

            if do_step:
                clip_cfg = phase_cfg.get("clip_grad")
                if clip_cfg:
                    with self._time(f"phase/{name}/clip_grad"):
                        params = self.components.trainable_parameters(phase_cfg.get("trainable", []))
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            params,
                            float(clip_cfg["max_norm"]),
                        )
                    metrics[f"{name}/grad_norm"] = grad_norm

                for opt_name in phase_cfg.get("step", []) or []:
                    with self._time(f"optimizer/{opt_name}/step"):
                        self.optimizers[opt_name].step()

                for sched_name in phase_cfg.get("schedulers", []) or []:
                    with self._time(f"scheduler/{sched_name}/step"):
                        self.schedulers[sched_name].step()

        return {k: _metric_to_float(v) for k, v in metrics.items()}
