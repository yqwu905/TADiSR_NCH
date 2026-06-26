from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


@dataclass
class DistState:
    enabled: bool
    strategy: str
    backend: Optional[str]
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main_process: bool
    device_mesh: Any = None


def _has_npu() -> bool:
    return hasattr(torch, "npu")


def _set_device(device_type: str, local_rank: int):
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    elif device_type == "npu":
        torch.npu.set_device(local_rank)


def _infer_device_type(device_cfg: str) -> str:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _has_npu():
            return "npu"
        return "cpu"

    if device_cfg.startswith("cuda"):
        return "cuda"
    if device_cfg.startswith("npu"):
        return "npu"
    return "cpu"


def _default_backend(device_type: str) -> str:
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    return "gloo"


def init_distributed(runtime_cfg: Optional[Mapping[str, Any]] = None) -> DistState:
    runtime_cfg = _plain(runtime_cfg) or {}
    dist_cfg = dict(runtime_cfg.get("distributed", {}) or {})

    strategy = str(dist_cfg.get("strategy", "none")).lower()
    device_cfg = str(runtime_cfg.get("device", "auto"))

    if strategy in {"none", "null", "false"}:
        device_type = _infer_device_type(device_cfg)
        if device_type == "cuda":
            device = torch.device("cuda:0")
        elif device_type == "npu":
            device = torch.device("npu:0")
        else:
            device = torch.device("cpu")

        return DistState(
            enabled=False,
            strategy="none",
            backend=None,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
            is_main_process=True,
        )

    if strategy not in {"ddp", "fsdp2"}:
        raise ValueError(f"unsupported distributed strategy: {strategy}")

    required_envs = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    missing = [k for k in required_envs if k not in os.environ]
    if missing:
        raise RuntimeError(
            f"{strategy} requires torchrun, missing envs: {missing}. "
            f"Launch with: torchrun --nproc_per_node=... -m framework.train --config xxx.yaml"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device_type = _infer_device_type(device_cfg)
    if strategy == "fsdp2" and device_type == "npu":
        raise ValueError("FSDP2 is not supported for NPU in this framework yet")

    if device_type == "cpu":
        device = torch.device("cpu")
    else:
        _set_device(device_type, local_rank)
        device = torch.device(f"{device_type}:{local_rank}")

    backend = str(dist_cfg.get("backend", _default_backend(device_type)))
    init_method = dist_cfg.get("init_method", "env://")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

    device_mesh = None
    if strategy == "fsdp2":
        device_mesh = _init_device_mesh(device_type, world_size)

    return DistState(
        enabled=True,
        strategy=strategy,
        backend=backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main_process=(rank == 0),
        device_mesh=device_mesh,
    )


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(module):
    if isinstance(module, DDP):
        return module.module
    return module


def is_ddp_wrapped(module) -> bool:
    return isinstance(module, DDP)


def wrap_ddp(module: Any, dist_state: DistState, ddp_cfg: Optional[Mapping[str, Any]] = None):
    if not dist_state.enabled:
        return module

    if not isinstance(module, nn.Module):
        return module

    ddp_cfg = dict(ddp_cfg or {})

    device_ids = None
    output_device = None

    if dist_state.device.type in {"cuda", "npu"}:
        device_ids = [dist_state.local_rank]
        output_device = dist_state.local_rank

    return DDP(
        module,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=bool(ddp_cfg.get("find_unused_parameters", False)),
        broadcast_buffers=bool(ddp_cfg.get("broadcast_buffers", True)),
        static_graph=bool(ddp_cfg.get("static_graph", False)),
    )


def _init_device_mesh(device_type: str, world_size: int):
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except Exception as e:
        raise ImportError("FSDP2 requires torch.distributed.device_mesh.init_device_mesh") from e

    return init_device_mesh(
        device_type,
        (world_size,),
        mesh_dim_names=("dp",),
    )


def _parse_dtype(value):
    if value in (None, "none", "null", "false", False):
        return None
    if isinstance(value, torch.dtype):
        return value

    key = str(value).lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if key not in aliases:
        raise ValueError(f"unsupported FSDP dtype: {value!r}")
    return aliases[key]


def build_fsdp2_kwargs(
    dist_state: DistState,
    fsdp_cfg: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    fsdp_cfg = dict(fsdp_cfg or {})
    kwargs: dict[str, Any] = {"mesh": dist_state.device_mesh}

    if "reshard_after_forward" in fsdp_cfg:
        kwargs["reshard_after_forward"] = fsdp_cfg["reshard_after_forward"]

    mp_cfg = dict(fsdp_cfg.get("mixed_precision", {}) or {})
    if bool(mp_cfg.get("enabled", False)):
        try:
            from torch.distributed.fsdp import MixedPrecisionPolicy
        except Exception as e:
            raise ImportError("FSDP2 mixed precision requires torch.distributed.fsdp") from e

        kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=_parse_dtype(mp_cfg.get("param_dtype")),
            reduce_dtype=_parse_dtype(mp_cfg.get("reduce_dtype")),
            output_dtype=_parse_dtype(mp_cfg.get("output_dtype")),
            cast_forward_inputs=bool(mp_cfg.get("cast_forward_inputs", True)),
        )

    offload_cfg = dict(fsdp_cfg.get("cpu_offload", {}) or {})
    if bool(offload_cfg.get("enabled", False)):
        try:
            from torch.distributed.fsdp import CPUOffloadPolicy
        except Exception as e:
            raise ImportError("FSDP2 CPU offload requires torch.distributed.fsdp") from e

        kwargs["offload_policy"] = CPUOffloadPolicy(
            pin_memory=bool(offload_cfg.get("pin_memory", True)),
        )

    return kwargs


def fully_shard_module(
    module: nn.Module,
    dist_state: DistState,
    fsdp_cfg: Optional[Mapping[str, Any]] = None,
):
    try:
        from torch.distributed.fsdp import fully_shard
    except Exception as e:
        raise ImportError("FSDP2 requires torch.distributed.fsdp.fully_shard") from e

    return fully_shard(module, **build_fsdp2_kwargs(dist_state, fsdp_cfg))


def is_fsdp2_module(module) -> bool:
    try:
        from torch.distributed.fsdp import FSDPModule
    except Exception:
        return False
    return isinstance(module, FSDPModule)


def register_fsdp2_forward_methods(module, method_names):
    method_names = list(method_names or [])
    if not method_names:
        return

    try:
        from torch.distributed.fsdp import register_fsdp_forward_method
    except Exception as e:
        raise ImportError("FSDP2 custom forward methods require register_fsdp_forward_method") from e

    for method_name in method_names:
        register_fsdp_forward_method(module, str(method_name))


def reduce_scalar(value, op: str = "mean"):
    if not (dist.is_available() and dist.is_initialized()):
        return value

    if not torch.is_tensor(value):
        value = torch.tensor(float(value), device="cuda" if torch.cuda.is_available() else "cpu")

    value = value.detach().float()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    if op == "mean":
        value = value / dist.get_world_size()

    return value
