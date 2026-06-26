from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, OmegaConf
import torch.distributed as torch_dist

from .context import TrainContext
from .instantiate import instantiate
from .components import ComponentManager, resolve_submodule
from .optim import build_optimizers, build_schedulers
from .losses import build_losses
from .phase_runner import PhaseRunner
from .distributed import init_distributed, barrier, is_fsdp2_module, unwrap_model
from .utils import StepTimer
from .loggers import build_loggers


logger = logging.getLogger(__name__)


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


def _resolve_build_workers(value) -> Optional[int]:
    if value is None or value is False:
        return None

    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "0", "false", "none", "null"):
            return None
        value = int(s)

    value = int(value)
    return value if value > 1 else None


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj

def metric_to_scalar_tensor(value, device) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return value.detach().float().mean().to(device)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.tensor(float(value), device=device)

    return None


def reduce_scalar_tensor(value: torch.Tensor, average: bool = True) -> torch.Tensor:
    if torch_dist.is_available() and torch_dist.is_initialized():
        value = value.clone()
        torch_dist.all_reduce(value, op=torch_dist.ReduceOp.SUM)
        if average:
            value = value / torch_dist.get_world_size()
        return value
    return value


def build_dataloader(data_cfg, dist_state=None):
    data_cfg = _plain(data_cfg)
    dataset = instantiate(data_cfg["dataset"])
    dl_cfg = dict(data_cfg.get("dataloader", {}) or {})

    num_workers = int(dl_cfg.get("num_workers", 0))

    if num_workers > 0:
        dl_cfg.setdefault("pin_memory", False)
        dl_cfg.setdefault("persistent_workers", False)
        dl_cfg.setdefault("prefetch_factor", 1)
        dl_cfg.setdefault("timeout", 60)
        dl_cfg.setdefault("multiprocessing_context", "spawn")

    sampler = None

    if dist_state is not None and dist_state.enabled:
        shuffle = bool(dl_cfg.pop("shuffle", True))
        drop_last = bool(dl_cfg.get("drop_last", False))

        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        dl_cfg["shuffle"] = False
        dl_cfg["sampler"] = sampler

    dataloader = DataLoader(dataset, **dl_cfg)
    return dataloader


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        runtime_cfg = _plain(cfg.get("runtime", {})) or {}

        self.dist_state = init_distributed(runtime_cfg)
        self.device = self.dist_state.device
        self.mixed_precision = runtime_cfg.get("mixed_precision", "no")
        self.global_step = 0
        profiling_cfg = _plain(cfg.get("profiling", {})) or {}
        self.profiling_enabled = bool(profiling_cfg.get("enabled", False))
        self.profiling_log_every = int(profiling_cfg.get("log_every", 1))
        checkpoint_cfg = _plain(cfg.get("checkpoint", {})) or {}
        self.save_every_steps = int(checkpoint_cfg.get("save_every_steps", 0) or 0)
        self.keep_last = checkpoint_cfg.get("keep_last", None)
        self.keep_last = int(self.keep_last) if self.keep_last not in (None, 0, "0") else None
        self.save_optimizer_state = bool(checkpoint_cfg.get("save_optimizer", True))
        self.save_scheduler_state = bool(checkpoint_cfg.get("save_scheduler", True))
        distributed_cfg = dict(runtime_cfg.get("distributed", {}) or {})
        fsdp_cfg = dict(distributed_cfg.get("fsdp", {}) or {})
        fsdp_checkpoint_cfg = dict(fsdp_cfg.get("checkpoint", {}) or {})
        self.fsdp_checkpoint_format = str(
            fsdp_checkpoint_cfg.get("format", "full_rank0")
        ).lower()

        self.timer = StepTimer(
            enabled=self.profiling_enabled,
            device=self.device,
            synchronize=bool(profiling_cfg.get("synchronize", True)),
            reduce=str(profiling_cfg.get("reduce", "max")),
            is_main_process=self.dist_state.is_main_process,
        )

        experiment_cfg = _plain(cfg.get("experiment", {})) or {}
        self.output_dir = experiment_cfg.get("output_dir", "outputs")

        logging_cfg = _plain(cfg.get("logging", {})) or {}
        self.log_reduce = bool(logging_cfg.get("reduce_metrics", True))
        self.loggers = build_loggers(
            logging_cfg=logging_cfg,
            output_dir=self.output_dir,
            is_main_process=self.dist_state.is_main_process,
        )

        if self.dist_state.is_main_process:
            logger.info(
                f"[runtime] distributed={self.dist_state.enabled}, "
                f"strategy={self.dist_state.strategy}, "
                f"backend={self.dist_state.backend}, "
                f"rank={self.dist_state.rank}, "
                f"local_rank={self.dist_state.local_rank}, "
                f"world_size={self.dist_state.world_size}, "
                f"device={self.device}",
            )

        self.train_loader = build_dataloader(cfg.data.train, dist_state=self.dist_state)

        build_cfg = _plain(runtime_cfg.get("build", {})) or {}
        max_workers = _resolve_build_workers(build_cfg.get("parallel_build_components"))
        self.components = ComponentManager(
            cfg.get("components", cfg.get("models", {}))
        ).build_all(max_workers=max_workers)
        self.components.apply_gradient_checkpointing()
        self.components.apply_parallel(
            self.dist_state,
            distributed_cfg=distributed_cfg,
        )
        self.components.set_initial_modes()

        if self.dist_state.is_main_process:
            self.components.print_parameter_summary()

        self.optimizers = build_optimizers(cfg.get("optimizers", {}), self.components)
        self.schedulers = build_schedulers(cfg.get("schedulers", {}), self.optimizers)

        self.losses = build_losses(cfg.get("losses", {}))
        for loss in self.losses.values():
            loss.to(self.device)

        self.phase_runner = PhaseRunner(
            self.components,
            self.optimizers,
            self.schedulers,
            self.losses,
            device=self.device,
            mixed_precision=self.mixed_precision,
            timer=self.timer,
        )

    def train(self):
        train_cfg = self.cfg.get("train", {})
        max_steps = int(train_cfg.get("max_steps", 1000))
        max_epochs = int(train_cfg.get("max_epochs", 10**9))
        log_every = int(train_cfg.get("log_every", 10))
        output_dir = self.output_dir

        if self.dist_state.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        if self.dist_state.enabled:
            barrier()

        for epoch in range(max_epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            data_t0 = time.perf_counter()
            for batch in self.train_loader:
                self.timer.reset()
                data_cost = time.perf_counter() - data_t0
                self.timer.add("step/data_wait", data_cost)

                if self.global_step >= max_steps:
                    self.save_checkpoint(output_dir, tag="last")
                    self.loggers.close()
                    return

                with self.timer.time("step/move_to_device"):
                    batch = move_to_device(batch, self.device)

                ctx = TrainContext()
                ctx.set("global_step", self.global_step)
                ctx.set("epoch", epoch)
                ctx.set("rank", self.dist_state.rank)
                ctx.set("local_rank", self.dist_state.local_rank)
                ctx.set("world_size", self.dist_state.world_size)
                ctx.set("batch", batch)

                all_metrics: Dict[str, Any] = {}

                with self.timer.time("step/all_phases"):
                    for phase_cfg in self.cfg.train_program.phases:
                        metrics = self.phase_runner.run(ctx, phase_cfg)
                        all_metrics.update(metrics)

                if self.dist_state.is_main_process:
                    self.loggers.log_images_from_context(ctx, self.global_step)
                scalar_metrics = self.prepare_scalar_metrics(all_metrics)
                if self.global_step % log_every == 0:
                    self.log_metrics(scalar_metrics)

                    if self.dist_state.is_main_process:
                        msg = f"step={self.global_step} " + " ".join(
                            f"{k}={v:.6g}" for k, v in scalar_metrics.items()
                        )
                        logger.info(msg)
                if (
                    self.profiling_enabled
                    and self.global_step % self.profiling_log_every == 0
                    and self.dist_state.is_main_process
                ):
                    logger.info(
                        "\n[profile] step=%s\n%s\n",
                        self.global_step,
                        self.timer.format(multiline=True),
                    )

                self.global_step += 1
                if (
                    self.save_every_steps > 0
                    and self.global_step % self.save_every_steps == 0
                ):
                    self.save_checkpoint(self.output_dir, tag=f"step-{self.global_step}")
                    self.cleanup_old_checkpoints(self.output_dir)
                data_t0 = time.perf_counter()

        self.save_checkpoint(output_dir, tag="last")
        self.loggers.close()

    def prepare_scalar_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        scalar_metrics = {}

        for key, value in metrics.items():
            scalar_tensor = metric_to_scalar_tensor(value, self.device)
            if scalar_tensor is None:
                continue

            if self.log_reduce:
                scalar_tensor = reduce_scalar_tensor(scalar_tensor, average=True)

            scalar_metrics[key] = float(scalar_tensor.detach().cpu().item())

        return scalar_metrics

    def log_metrics(self, metrics: Dict[str, float]):
        if not self.dist_state.is_main_process:
            return

        self.loggers.log_metrics(metrics, self.global_step)
        self.loggers.flush()

    def save_checkpoint(self, output_dir: str, tag: str = "last"):
        if self.dist_state.strategy == "fsdp2":
            self.save_fsdp2_checkpoint(output_dir, tag=tag)
            return

        if not self.dist_state.is_main_process:
            return

        ckpt_dir = os.path.join(output_dir, f"checkpoint-{tag}")
        os.makedirs(ckpt_dir, exist_ok=True)

        OmegaConf.save(self.cfg, os.path.join(ckpt_dir, "config.yaml"))

        torch.save(
            {
                "global_step": self.global_step,
                "world_size": self.dist_state.world_size,
            },
            os.path.join(ckpt_dir, "trainer_state.pt"),
        )

        models_dir = os.path.join(ckpt_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        for name, module in self.components.unwrapped_items():
            if not hasattr(module, "state_dict"):
                continue

            entry = self.components.entries[name]
            save_policy = entry.cfg.get("save", "full")
            if save_policy == "none":
                continue

            save_submodule = entry.cfg.get("save_submodule")
            if save_submodule:
                module = resolve_submodule(module, save_submodule)

            if save_policy == "lora_only" and hasattr(module, "save_pretrained"):
                module.save_pretrained(os.path.join(models_dir, f"{name}_lora"))
            else:
                torch.save(
                    module.state_dict(),
                    os.path.join(models_dir, f"{name}.pt"),
                )

        if self.save_optimizer_state:
            optim_dir = os.path.join(ckpt_dir, "optimizers")
            os.makedirs(optim_dir, exist_ok=True)

            for name, optimizer in self.optimizers.items():
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(optim_dir, f"{name}.pt"),
                )

        if self.save_scheduler_state:
            sched_dir = os.path.join(ckpt_dir, "schedulers")
            os.makedirs(sched_dir, exist_ok=True)

            for name, scheduler in self.schedulers.items():
                if hasattr(scheduler, "state_dict"):
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(sched_dir, f"{name}.pt"),
                    )

        logger.info("[checkpoint] saved to %s", ckpt_dir)

    def save_fsdp2_checkpoint(self, output_dir: str, tag: str = "last"):
        if self.fsdp_checkpoint_format == "dcp_sharded":
            self.save_fsdp2_dcp_checkpoint(output_dir, tag=tag)
            return

        if self.fsdp_checkpoint_format != "full_rank0":
            raise ValueError(
                f"unsupported FSDP2 checkpoint format: {self.fsdp_checkpoint_format}"
            )

        ckpt_dir = os.path.join(output_dir, f"checkpoint-{tag}")
        if self.dist_state.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            OmegaConf.save(self.cfg, os.path.join(ckpt_dir, "config.yaml"))
            torch.save(
                {
                    "global_step": self.global_step,
                    "world_size": self.dist_state.world_size,
                },
                os.path.join(ckpt_dir, "trainer_state.pt"),
            )
            os.makedirs(os.path.join(ckpt_dir, "models"), exist_ok=True)
            if self.save_optimizer_state:
                os.makedirs(os.path.join(ckpt_dir, "optimizers"), exist_ok=True)
            if self.save_scheduler_state:
                os.makedirs(os.path.join(ckpt_dir, "schedulers"), exist_ok=True)

        if self.dist_state.enabled:
            barrier()

        model_options, get_model_state_dict, get_optimizer_state_dict = (
            self._fsdp_full_state_dict_helpers()
        )
        models_dir = os.path.join(ckpt_dir, "models")

        for name, entry in self.components.entries.items():
            module = unwrap_model(entry.module)
            if not hasattr(module, "state_dict"):
                continue

            save_policy = entry.cfg.get("save", "full")
            if save_policy == "none":
                continue

            save_submodule = entry.cfg.get("save_submodule")
            if save_submodule:
                module = resolve_submodule(module, save_submodule)

            if is_fsdp2_module(module):
                if save_policy == "lora_only":
                    raise NotImplementedError(
                        "FSDP2 checkpointing does not support save: lora_only yet"
                    )
                state_dict = get_model_state_dict(module, options=model_options)
                if self.dist_state.is_main_process:
                    torch.save(state_dict, os.path.join(models_dir, f"{name}.pt"))
                continue

            if save_submodule and is_fsdp2_module(unwrap_model(entry.module)):
                raise ValueError(
                    f"component {name} save_submodule {save_submodule!r} is not an "
                    f"FSDPModule; add it to fsdp.wrap_modules so it gets sharded, "
                    f"otherwise the saved checkpoint would contain DTensor shards "
                    f"instead of a full state dict."
                )

            if not self.dist_state.is_main_process:
                continue

            if save_policy == "lora_only" and hasattr(module, "save_pretrained"):
                module.save_pretrained(os.path.join(models_dir, f"{name}_lora"))
            else:
                torch.save(
                    self._cpu_state_dict(module.state_dict()),
                    os.path.join(models_dir, f"{name}.pt"),
                )

        if self.save_optimizer_state:
            bundle = self._component_state_bundle()
            optim_dir = os.path.join(ckpt_dir, "optimizers")

            for name, optimizer in self.optimizers.items():
                state_dict = get_optimizer_state_dict(
                    bundle,
                    optimizer,
                    options=model_options,
                )
                if self.dist_state.is_main_process:
                    torch.save(state_dict, os.path.join(optim_dir, f"{name}.pt"))

        if self.save_scheduler_state and self.dist_state.is_main_process:
            sched_dir = os.path.join(ckpt_dir, "schedulers")
            for name, scheduler in self.schedulers.items():
                if hasattr(scheduler, "state_dict"):
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(sched_dir, f"{name}.pt"),
                    )

        if self.dist_state.enabled:
            barrier()

        if self.dist_state.is_main_process:
            logger.info("[checkpoint] saved to %s", ckpt_dir)

    def save_fsdp2_dcp_checkpoint(self, output_dir: str, tag: str = "last"):
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{tag}")
        dcp_dir = os.path.join(ckpt_dir, "dcp")
        os.makedirs(dcp_dir, exist_ok=True)

        if self.dist_state.is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            OmegaConf.save(self.cfg, os.path.join(ckpt_dir, "config.yaml"))

        state: Dict[str, Any] = {
            "trainer_state": {
                "global_step": self.global_step,
                "world_size": self.dist_state.world_size,
            },
            "models": {},
        }

        for name, entry in self.components.entries.items():
            module = unwrap_model(entry.module)
            if not hasattr(module, "state_dict"):
                continue
            if entry.cfg.get("save", "full") == "none":
                continue

            save_submodule = entry.cfg.get("save_submodule")
            if save_submodule:
                module = resolve_submodule(module, save_submodule)

            if entry.cfg.get("save", "full") == "lora_only" and is_fsdp2_module(module):
                raise NotImplementedError(
                    "FSDP2 dcp_sharded checkpointing does not support save: lora_only yet"
                )
            state["models"][name] = module.state_dict()

        if self.save_optimizer_state:
            state["optimizers"] = {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            }

        if self.save_scheduler_state:
            state["schedulers"] = {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
                if hasattr(scheduler, "state_dict")
            }

        try:
            import torch.distributed.checkpoint as dcp
        except Exception as e:
            raise ImportError("FSDP2 dcp_sharded checkpointing requires torch.distributed.checkpoint") from e

        dcp.save(state, checkpoint_id=dcp_dir)

        if self.dist_state.enabled:
            barrier()

        if self.dist_state.is_main_process:
            logger.info("[checkpoint] saved to %s", ckpt_dir)

    @staticmethod
    def _fsdp_full_state_dict_helpers():
        try:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                get_optimizer_state_dict,
            )
        except Exception as e:
            raise ImportError(
                "FSDP2 full checkpointing requires torch.distributed.checkpoint.state_dict"
            ) from e

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return options, get_model_state_dict, get_optimizer_state_dict

    def _component_state_bundle(self) -> nn.Module:
        bundle = nn.Module()
        for name, entry in self.components.entries.items():
            module = unwrap_model(entry.module)
            if isinstance(module, nn.Module):
                bundle.add_module(name, module)
        return bundle

    @staticmethod
    def _cpu_state_dict(state_dict):
        cpu_state = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                cpu_state[key] = value.detach().cpu()
            else:
                cpu_state[key] = value
        return cpu_state

    def cleanup_old_checkpoints(self, output_dir: str):
        if not self.dist_state.is_main_process:
            return
    
        if self.keep_last is None or self.keep_last <= 0:
            return
    
        if not os.path.isdir(output_dir):
            return
    
        ckpts = []
    
        for name in os.listdir(output_dir):
            if not name.startswith("checkpoint-step-"):
                continue
            
            step_str = name.replace("checkpoint-step-", "")
            try:
                step = int(step_str)
            except ValueError:
                continue
            
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                ckpts.append((step, path))
    
        ckpts.sort(key=lambda x: x[0])
    
        while len(ckpts) > self.keep_last:
            step, path = ckpts.pop(0)
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            logger.info("[checkpoint] removed old checkpoint: %s", path)
