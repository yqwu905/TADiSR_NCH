from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.distributed as dist


class StepTimer:
    def __init__(
        self,
        enabled: bool = False,
        device=None,
        synchronize: bool = True,
        reduce: str = "max",  # none / mean / max
        is_main_process: bool = True,
    ):
        self.enabled = enabled
        self.device = device
        self.synchronize = synchronize
        self.reduce = reduce
        self.is_main_process = is_main_process
        self.records: Dict[str, float] = {}

    def reset(self):
        self.records = {}

    def _sync(self):
        if not self.enabled or not self.synchronize:
            return

        if self.device is None:
            return

        device_type = getattr(self.device, "type", str(self.device))

        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elif device_type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize(self.device)

    @contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return

        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            cost = time.perf_counter() - t0
            self.records[name] = self.records.get(name, 0.0) + cost

    def add(self, name: str, cost: float):
        if not self.enabled:
            return
        self.records[name] = self.records.get(name, 0.0) + float(cost)

    def _reduce_value(self, value: float) -> float:
        if self.reduce == "none":
            return value

        if not (dist.is_available() and dist.is_initialized()):
            return value

        tensor = torch.tensor(float(value), device=self.device)

        if self.reduce == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / dist.get_world_size()
        elif self.reduce == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"unknown timer reduce mode: {self.reduce}")

        return float(tensor.detach().cpu().item())

    def get_reduced_records(self) -> Dict[str, float]:
        if not self.enabled:
            return {}

        return {
            name: self._reduce_value(value)
            for name, value in self.records.items()
        }

    def format(self, multiline: bool = True) -> str:
        records = self.get_reduced_records()
        if not records:
            return ""

        # 优先用 step/all_phases 作为百分比分母，否则退化用所有记录最大值
        ref_total = records.get("step/all_phases")
        if ref_total is None:
            ref_total = records.get("phase/generator/total")
        if ref_total is None:
            ref_total = max(records.values()) if records else 0.0

        groups = {
            "step": [],
            "phase": [],
            "component": [],
            "optimizer": [],
            "scheduler": [],
            "op_type": [],
            "other": [],
        }

        for name, cost in sorted(records.items(), key=lambda x: x[0]):
            if name.startswith("step/"):
                groups["step"].append((name, cost))
            elif name.startswith("phase/"):
                groups["phase"].append((name, cost))
            elif name.startswith("component/"):
                groups["component"].append((name, cost))
            elif name.startswith("optimizer/"):
                groups["optimizer"].append((name, cost))
            elif name.startswith("scheduler/"):
                groups["scheduler"].append((name, cost))
            elif name.startswith("op_type/"):
                groups["op_type"].append((name, cost))
            else:
                groups["other"].append((name, cost))

        def fmt_item(name: str, cost: float) -> str:
            if ref_total and ref_total > 0:
                return f"  {name}: {cost:.4f}s ({cost / ref_total * 100:.1f}%)"
            return f"  {name}: {cost:.4f}s"

        lines = []
        lines.append(f"reference_total: {ref_total:.4f}s")

        for group_name in ["step", "component", "phase", "optimizer", "scheduler", "op_type", "other"]:
            items = groups[group_name]
            if not items:
                continue

            lines.append(f"{group_name}:")
            for name, cost in items:
                lines.append(fmt_item(name, cost))

        if multiline:
            return "\n".join(lines)

        return " ".join(lines)