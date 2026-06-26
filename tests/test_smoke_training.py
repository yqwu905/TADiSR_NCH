from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from framework.config import load_config
from framework.context import TrainContext
from framework.distributed import DistState
from framework.engine import Trainer, build_dataloader, _resolve_build_workers
from framework.loggers import LoggerCollection, _normalize_backend_configs, to_log_images
from framework.losses import build_losses
from framework.ops.common import CallOp
from framework.registry import get_op
from framework.components import ComponentEntry, ComponentManager
from framework.phase_runner import PhaseRunner
from framework.optim import build_optimizers
from test_assets.models import CheckpointedRegressor, TinyRegressor, WrappedRegressor


class SmokeTrainingTest(unittest.TestCase):
    def test_dataset_batch_matches_context_contract(self):
        cfg = load_config("configs/test/tiny_regression.yaml")
        loader = build_dataloader(cfg.data.train)
        batch = next(iter(loader))

        self.assertEqual(batch["input"].shape, (4, 4))
        self.assertEqual(batch["target"].shape, (4, 2))
        self.assertEqual(batch["sample_id"].shape, (4,))

    def test_phase_runner_updates_model_and_reports_loss_metrics(self):
        torch.manual_seed(11)
        cfg = load_config("configs/test/tiny_regression.yaml")
        batch = next(iter(build_dataloader(cfg.data.train)))
        ctx = TrainContext(global_step=0, batch=batch)

        components = ComponentManager(cfg.components).build_all()
        optimizers = build_optimizers(cfg.optimizers, components)
        losses = build_losses(cfg.losses)
        runner = PhaseRunner(
            components,
            optimizers,
            schedulers={},
            losses=losses,
            device=torch.device("cpu"),
        )

        before = {
            name: param.detach().clone()
            for name, param in components.unwrap("regressor").named_parameters()
        }
        metrics = runner.run(ctx, cfg.train_program.phases[0])
        after = dict(components.unwrap("regressor").named_parameters())

        self.assertIn("mse/loss", metrics)
        self.assertIn("mse/mae", metrics)
        self.assertIn("regression/total_loss", metrics)
        self.assertTrue(ctx.has("pred.value"))
        self.assertTrue(
            any(not torch.equal(before[name], after[name]) for name in before),
            "expected at least one model parameter to change",
        )

    def test_trainer_runs_and_writes_checkpoint(self):
        cfg = load_config("configs/test/tiny_regression.yaml")
        output_dir = tempfile.mkdtemp(prefix="fictional-fortnight-test-")
        try:
            cfg = OmegaConf.merge(cfg, {"experiment": {"output_dir": output_dir}})
            trainer = Trainer(cfg)
            trainer.train()

            self.assertEqual(trainer.global_step, 3)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "checkpoint-last")))
            self.assertTrue(
                os.path.exists(
                    os.path.join(output_dir, "checkpoint-last", "models", "regressor.pt")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(output_dir, "checkpoint-last", "optimizers", "main.pt")
                )
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_logger_backend_configs_support_legacy_and_multi_backend_forms(self):
        legacy = _normalize_backend_configs(
            {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "/tmp/tb",
                }
            }
        )
        self.assertEqual(legacy[0]["type"], "tensorboard")
        self.assertTrue(legacy[0]["enabled"])

        multi = _normalize_backend_configs(
            {
                "backends": {
                    "tensorboard": {"enabled": True},
                    "aim": {"enabled": False},
                    "wandb": {"enabled": True, "project": "smoke"},
                }
            }
        )
        self.assertEqual([cfg["type"] for cfg in multi], ["tensorboard", "aim", "wandb"])
        self.assertEqual(multi[2]["project"], "smoke")

        shorthand = _normalize_backend_configs({"backends": ["tensorboard", "wandb"]})
        self.assertEqual([cfg["type"] for cfg in shorthand], ["tensorboard", "wandb"])

    def test_logger_collection_records_scalars_and_context_images(self):
        class RecordingLogger:
            def __init__(self):
                self.metrics = []
                self.images = []
                self.flushed = False

            def log_metrics(self, metrics, step):
                self.metrics.append((step, dict(metrics)))

            def log_images(self, tag, images, step):
                self.images.append((step, tag, tuple(images.shape)))

            def flush(self):
                self.flushed = True

            def close(self):
                self.flush()

        backend = RecordingLogger()
        collection = LoggerCollection(
            [backend],
            image_cfg={
                "enabled": True,
                "every_n_steps": 2,
                "max_images": 2,
                "items": [
                    {
                        "tag": "image/input",
                        "key": "batch.image",
                        "value_range": "-1_1",
                    }
                ],
            },
        )
        ctx = TrainContext()
        ctx.set("batch.image", torch.zeros(3, 4, 5))

        collection.log_metrics({"loss": 1.25}, step=1)
        collection.log_images_from_context(ctx, step=1)
        collection.log_images_from_context(ctx, step=2)
        collection.flush()

        self.assertEqual(backend.metrics, [(1, {"loss": 1.25})])
        self.assertEqual(backend.images, [(2, "image/input", (1, 3, 4, 5))])
        self.assertTrue(backend.flushed)

    def test_to_log_images_handles_nhwc_and_value_ranges(self):
        value = torch.full((2, 4, 5, 3), 255.0)
        images = to_log_images(value, value_range="0_255", max_images=1)
        self.assertEqual(tuple(images.shape), (1, 3, 4, 5))
        self.assertTrue(torch.allclose(images, torch.ones_like(images)))

    def test_nch_ldm_v3_two_step_op_calls_dit_component(self):
        class RecordingDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(
                self,
                hidden_states,
                timestep,
                encoder_hidden_states,
                enable_skip_level,
            ):
                self.calls.append(
                    {
                        "hidden_shape": tuple(hidden_states.shape),
                        "timestep": timestep.detach().clone(),
                        "encoder_dtype": encoder_hidden_states.dtype,
                        "encoder_device": encoder_hidden_states.device,
                        "enable_skip_level": enable_skip_level,
                    }
                )
                view_shape = [timestep.shape[0]] + [1] * (hidden_states.ndim - 1)
                value = timestep.view(*view_shape)
                output_shape = (
                    hidden_states.shape[0],
                    2,
                    *hidden_states.shape[2:],
                )
                return (
                    torch.ones(
                        output_shape,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    * value,
                )

        hidden = torch.ones(2, 2, 3, 4)
        encoder = torch.ones(2, 5, 6, dtype=torch.float64)
        ctx = TrainContext()
        ctx.set("latent.lq", hidden)
        ctx.set("text.embedding", encoder)

        model = RecordingDenoiser()
        op = get_op("nch_ldm_v3_two_step")(
            {
                "component": "dit",
                "input_type": "lq_one_lq",
                "timesteps": [1.0, 0.4],
                "enable_skip_level": "60",
                "inputs": {
                    "hidden_states": "latent.lq",
                    "encoder_hidden_states": "text.embedding",
                },
                "outputs": {"out": "pred.value"},
            }
        )
        op(ctx, {"dit": model})

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["hidden_shape"], (2, 8, 3, 4))
        self.assertEqual(model.calls[0]["enable_skip_level"], "60")
        self.assertEqual(model.calls[0]["encoder_dtype"], hidden.dtype)
        self.assertEqual(model.calls[0]["encoder_device"], hidden.device)
        self.assertTrue(
            torch.allclose(model.calls[0]["timestep"], torch.full((2,), 1.0))
        )
        self.assertTrue(
            torch.allclose(model.calls[1]["timestep"], torch.full((2,), 0.4))
        )
        self.assertTrue(
            torch.allclose(ctx.get("pred.value"), torch.full_like(hidden, 0.24))
        )

    def test_component_parallel_strategy_defaults_for_fsdp2(self):
        cfg = OmegaConf.create(
            {
                "trainable": {
                    "target": "test_assets.models.TinyRegressor",
                    "train": {"strategy": "full"},
                },
                "frozen": {
                    "target": "test_assets.models.TinyRegressor",
                    "train": {"strategy": "frozen"},
                },
                "explicit_fsdp": {
                    "target": "test_assets.models.TinyRegressor",
                    "train": {"strategy": "full"},
                    "parallel": {"strategy": "fsdp2"},
                },
                "legacy_fsdp_enabled": {
                    "target": "test_assets.models.TinyRegressor",
                    "train": {"strategy": "full"},
                    "fsdp": {"enabled": True},
                },
            }
        )
        components = ComponentManager(cfg).build_all()
        fsdp_cfg = {"default_non_fsdp_trainable": "ddp"}

        self.assertEqual(
            components.resolve_parallel_strategy(components.entries["trainable"], fsdp_cfg),
            "ddp",
        )
        self.assertEqual(
            components.resolve_parallel_strategy(components.entries["frozen"], fsdp_cfg),
            "replicated",
        )
        self.assertEqual(
            components.resolve_parallel_strategy(components.entries["explicit_fsdp"], fsdp_cfg),
            "fsdp2",
        )
        self.assertEqual(
            components.resolve_parallel_strategy(
                components.entries["legacy_fsdp_enabled"],
                fsdp_cfg,
            ),
            "fsdp2",
        )

    def test_trainable_replicated_component_requires_explicit_opt_in(self):
        cfg = OmegaConf.create(
            {
                "bad": {
                    "target": "test_assets.models.TinyRegressor",
                    "train": {"strategy": "full"},
                    "parallel": {"strategy": "replicated"},
                }
            }
        )
        components = ComponentManager(cfg).build_all()
        dist_state = DistState(
            enabled=True,
            strategy="fsdp2",
            backend="gloo",
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            is_main_process=True,
        )

        with self.assertRaisesRegex(ValueError, "trainable parameters"):
            components.apply_parallel(dist_state, distributed_cfg={"fsdp": {}})

    def test_fsdp_wrap_modules_reports_missing_paths(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "target": "test_assets.models.TinyRegressor",
                    "parallel": {"strategy": "fsdp2"},
                    "fsdp": {"wrap_modules": ["missing.*"]},
                }
            }
        )
        components = ComponentManager(cfg).build_all()
        dist_state = DistState(
            enabled=True,
            strategy="fsdp2",
            backend="gloo",
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            is_main_process=True,
        )

        with self.assertRaisesRegex(ValueError, "matched no modules"):
            components.apply_parallel(dist_state, distributed_cfg={"fsdp": {}})

    def _make_fsdp2_dist_state(self):
        return DistState(
            enabled=True,
            strategy="fsdp2",
            backend="gloo",
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            is_main_process=True,
        )

    def _register_fsdp_component(self, components, name, module, cfg=None):
        components.entries[name] = ComponentEntry(
            name=name,
            module=module,
            cfg=cfg or {"parallel": {"strategy": "fsdp2"}},
            trainable_flags={},
        )

    def test_fsdp_wrap_modules_falls_back_to_module_method(self):
        class WrapModuleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_a = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
                self.block_b = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
                self.head = nn.Linear(4, 2)

            def forward(self, x):
                return self.head(self.block_b(self.block_a(x)))

            def get_fsdp_wrap_module_list(self):
                return [self.block_a, self.block_b]

        components = ComponentManager({})
        module = WrapModuleModel()
        self._register_fsdp_component(components, "model", module)
        dist_state = self._make_fsdp2_dist_state()

        shard_calls = []

        def fake_fully_shard(mod, _dist_state, _fsdp_cfg=None):
            shard_calls.append(mod)
            return mod

        with patch("framework.components.fully_shard_module", side_effect=fake_fully_shard):
            components.apply_parallel(dist_state, distributed_cfg={"fsdp": {}})

        self.assertEqual(len(shard_calls), 3)
        self.assertIs(shard_calls[0], module.block_a)
        self.assertIs(shard_calls[1], module.block_b)
        self.assertIs(shard_calls[2], module)

    def test_fsdp_wrap_modules_config_takes_precedence_over_method(self):
        class WrapModuleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_a = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
                self.block_b = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
                self.head = nn.Linear(4, 2)

            def forward(self, x):
                return self.head(self.block_b(self.block_a(x)))

            def get_fsdp_wrap_module_list(self):
                raise AssertionError("get_fsdp_wrap_module_list should not be called")

        components = ComponentManager({})
        module = WrapModuleModel()
        self._register_fsdp_component(
            components,
            "model",
            module,
            cfg={
                "parallel": {"strategy": "fsdp2"},
                "fsdp": {"wrap_modules": ["block_b"]},
            },
        )
        dist_state = self._make_fsdp2_dist_state()

        shard_calls = []

        def fake_fully_shard(mod, _dist_state, _fsdp_cfg=None):
            shard_calls.append(mod)
            return mod

        with patch("framework.components.fully_shard_module", side_effect=fake_fully_shard):
            components.apply_parallel(dist_state, distributed_cfg={"fsdp": {}})

        self.assertEqual(len(shard_calls), 2)
        self.assertIs(shard_calls[0], module.block_b)
        self.assertIs(shard_calls[1], module)

    def test_fsdp_warns_when_no_wrap_modules_and_no_method(self):
        class PlainModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_a = nn.Sequential(nn.Linear(4, 8), nn.Tanh())

            def forward(self, x):
                return self.block_a(x)

        components = ComponentManager({})
        module = PlainModel()
        self._register_fsdp_component(components, "model", module)
        dist_state = self._make_fsdp2_dist_state()

        shard_calls = []

        def fake_fully_shard(mod, _dist_state, _fsdp_cfg=None):
            shard_calls.append(mod)
            return mod

        with patch("framework.components.fully_shard_module", side_effect=fake_fully_shard):
            with self.assertLogs("framework.components", level="WARNING") as cm:
                components.apply_parallel(dist_state, distributed_cfg={"fsdp": {}})

        self.assertEqual(len(shard_calls), 1)
        self.assertIs(shard_calls[0], module)
        self.assertTrue(any("get_fsdp_wrap_module_list" in msg for msg in cm.output))

    def test_build_all_parallel_runs_concurrently_and_preserves_order(self):
        from framework.instantiate import instantiate as real_instantiate

        cfg = OmegaConf.create(
            {
                f"comp{i}": {"target": "test_assets.models.TinyRegressor"}
                for i in range(4)
            }
        )
        sleep_seconds = 0.3

        def slow_instantiate(c, **kw):
            time.sleep(sleep_seconds)
            return real_instantiate(c, **kw)

        with patch("framework.components.instantiate", side_effect=slow_instantiate):
            t0 = time.perf_counter()
            components = ComponentManager(cfg).build_all(max_workers=4)
            elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, sleep_seconds * 2.5)
        self.assertEqual(
            list(components.entries.keys()),
            [f"comp{i}" for i in range(4)],
        )

    def test_build_all_serial_when_max_workers_le_one(self):
        from framework.instantiate import instantiate as real_instantiate

        cfg = OmegaConf.create(
            {
                f"comp{i}": {"target": "test_assets.models.TinyRegressor"}
                for i in range(4)
            }
        )
        sleep_seconds = 0.2

        def slow_instantiate(c, **kw):
            time.sleep(sleep_seconds)
            return real_instantiate(c, **kw)

        with patch("framework.components.instantiate", side_effect=slow_instantiate):
            t0 = time.perf_counter()
            components = ComponentManager(cfg).build_all(max_workers=1)
            elapsed = time.perf_counter() - t0

        self.assertGreaterEqual(elapsed, sleep_seconds * 3.5)
        self.assertEqual(
            list(components.entries.keys()),
            [f"comp{i}" for i in range(4)],
        )

    def test_build_all_parallel_propagates_first_exception(self):
        cfg = OmegaConf.create(
            {
                "good": {"target": "test_assets.models.TinyRegressor"},
                "bad": {"target": "test_assets.models.NoSuchClass"},
            }
        )

        with self.assertRaises((AttributeError, ModuleNotFoundError)):
            ComponentManager(cfg).build_all(max_workers=2)

    def test_resolve_build_workers_accepts_common_falsey_values(self):
        self.assertIsNone(_resolve_build_workers(None))
        self.assertIsNone(_resolve_build_workers(False))
        self.assertIsNone(_resolve_build_workers(0))
        self.assertIsNone(_resolve_build_workers(1))
        self.assertIsNone(_resolve_build_workers(""))
        self.assertIsNone(_resolve_build_workers("0"))
        self.assertIsNone(_resolve_build_workers("false"))
        self.assertIsNone(_resolve_build_workers("none"))
        self.assertEqual(_resolve_build_workers(4), 4)
        self.assertEqual(_resolve_build_workers("4"), 4)

    def test_call_op_default_forward_invokes_module_call_hooks(self):
        class HookedModule(nn.Module):
            def forward(self, x):
                return x + 1

        module = HookedModule()
        called = {"pre": False}
        module.register_forward_pre_hook(lambda *_: called.__setitem__("pre", True))

        class Components:
            def __getitem__(self, name):
                assert name == "model"
                return module

        ctx = TrainContext()
        ctx.set("x", torch.tensor(1.0))
        op = CallOp(
            {
                "component": "model",
                "inputs": {"x": "x"},
                "outputs": {"_": "y"},
            }
        )

        op(ctx, Components())

        self.assertTrue(called["pre"])
        self.assertEqual(float(ctx.get("y")), 2.0)

    def test_gradient_checkpointing_bool_shorthand_enables_via_auto_detect(self):
        components = ComponentManager({})
        module = CheckpointedRegressor()
        components.entries["m"] = ComponentEntry(
            name="m",
            module=module,
            cfg={"gradient_checkpointing": True},
            trainable_flags={},
        )

        components.apply_gradient_checkpointing()

        self.assertTrue(components.entries["m"].gradient_checkpointing)
        self.assertTrue(module.use_gc)
        self.assertEqual(module.gc_method_calls[0][0], "gradient_checkpointing_enable")

    def test_gradient_checkpointing_dict_form_passes_method_and_kwargs(self):
        components = ComponentManager({})
        module = CheckpointedRegressor()
        components.entries["m"] = ComponentEntry(
            name="m",
            module=module,
            cfg={
                "gradient_checkpointing": {
                    "enabled": True,
                    "method": "enable_gradient_checkpointing",
                    "method_kwargs": {"use_reentrant": False},
                }
            },
            trainable_flags={},
        )

        components.apply_gradient_checkpointing()

        self.assertTrue(components.entries["m"].gradient_checkpointing)
        self.assertTrue(module.use_gc)
        self.assertEqual(module.gc_method_calls[0][0], "enable_gradient_checkpointing")
        self.assertEqual(module.gc_method_calls[0][1], {"use_reentrant": False})

    def test_gradient_checkpointing_falsy_values_are_noop(self):
        components = ComponentManager({})
        off_module = CheckpointedRegressor()
        components.entries["off"] = ComponentEntry(
            name="off",
            module=off_module,
            cfg={"gradient_checkpointing": False},
            trainable_flags={},
        )
        missing_module = CheckpointedRegressor()
        components.entries["missing"] = ComponentEntry(
            name="missing",
            module=missing_module,
            cfg={},
            trainable_flags={},
        )
        disabled_module = CheckpointedRegressor()
        components.entries["disabled"] = ComponentEntry(
            name="disabled",
            module=disabled_module,
            cfg={"gradient_checkpointing": {"enabled": False}},
            trainable_flags={},
        )

        components.apply_gradient_checkpointing()

        self.assertFalse(components.entries["off"].gradient_checkpointing)
        self.assertFalse(components.entries["missing"].gradient_checkpointing)
        self.assertFalse(components.entries["disabled"].gradient_checkpointing)
        self.assertFalse(off_module.use_gc)
        self.assertFalse(missing_module.use_gc)
        self.assertFalse(disabled_module.use_gc)

    def test_gradient_checkpointing_missing_method_raises(self):
        components = ComponentManager({})
        components.entries["m"] = ComponentEntry(
            name="m",
            module=TinyRegressor(),
            cfg={"gradient_checkpointing": True},
            trainable_flags={},
        )

        with self.assertRaisesRegex(ValueError, "gradient_checkpointing_enable"):
            components.apply_gradient_checkpointing()

    def test_gradient_checkpointing_explicit_method_missing_raises(self):
        components = ComponentManager({})
        components.entries["m"] = ComponentEntry(
            name="m",
            module=CheckpointedRegressor(),
            cfg={
                "gradient_checkpointing": {
                    "enabled": True,
                    "method": "nonexistent_method",
                }
            },
            trainable_flags={},
        )

        with self.assertRaisesRegex(ValueError, "nonexistent_method"):
            components.apply_gradient_checkpointing()

    def test_gradient_checkpointing_non_module_warns_and_skips(self):
        class NonModule:
            pass

        components = ComponentManager({})
        components.entries["m"] = ComponentEntry(
            name="m",
            module=NonModule(),
            cfg={"gradient_checkpointing": True},
            trainable_flags={},
        )

        with self.assertLogs("framework.components", level="WARNING") as cm:
            components.apply_gradient_checkpointing()

        self.assertFalse(components.entries["m"].gradient_checkpointing)
        self.assertTrue(any("not an nn.Module" in msg for msg in cm.output))

    def test_gradient_checkpointing_integration_via_phase_runner(self):
        torch.manual_seed(11)
        cfg = load_config("configs/test/tiny_gradient_checkpointing.yaml")
        batch = next(iter(build_dataloader(cfg.data.train)))
        ctx = TrainContext(global_step=0, batch=batch)

        components = ComponentManager(cfg.components).build_all()
        components.apply_gradient_checkpointing()

        regressor = components.unwrap("regressor")
        self.assertTrue(regressor.use_gc)
        self.assertTrue(components.entries["regressor"].gradient_checkpointing)

        optimizers = build_optimizers(cfg.optimizers, components)
        losses = build_losses(cfg.losses)
        runner = PhaseRunner(
            components,
            optimizers,
            schedulers={},
            losses=losses,
            device=torch.device("cpu"),
        )

        before = {
            name: param.detach().clone()
            for name, param in regressor.named_parameters()
        }
        metrics = runner.run(ctx, cfg.train_program.phases[0])
        after = dict(regressor.named_parameters())

        self.assertIn("mse/loss", metrics)
        self.assertIn("regression/total_loss", metrics)
        self.assertTrue(
            any(p.grad is not None for p in regressor.parameters()),
            "expected gradients to flow through gradient checkpointing",
        )
        self.assertTrue(
            any(not torch.equal(before[name], after[name]) for name in before),
            "expected at least one model parameter to change",
        )

    @unittest.skipUnless(
        os.environ.get("RUN_FSDP2_TORCHRUN_SMOKE") == "1",
        "set RUN_FSDP2_TORCHRUN_SMOKE=1 to run the two-process FSDP2 smoke test",
    )
    def test_torchrun_mixed_fsdp2_smoke(self):
        torchrun = shutil.which("torchrun")
        if torchrun is None:
            self.skipTest("torchrun is not available")

        output_dir = tempfile.mkdtemp(prefix="fictional-fortnight-fsdp2-test-")
        try:
            cmd = [
                torchrun,
                "--standalone",
                "--nproc_per_node=2",
                "-m",
                "framework.train",
                "--config",
                "configs/test/tiny_mixed_fsdp2.yaml",
                f"experiment.output_dir={output_dir}",
            ]
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "checkpoint-last")))
            self.assertTrue(
                os.path.exists(
                    os.path.join(output_dir, "checkpoint-last", "models", "fsdp_model.pt")
                )
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_save_submodule_strips_wrapper_prefix_and_round_trips(self):
        cfg = load_config("configs/test/tiny_save_submodule.yaml")
        output_dir = tempfile.mkdtemp(prefix="fictional-fortnight-save-submodule-")
        try:
            cfg = OmegaConf.merge(cfg, {"experiment": {"output_dir": output_dir}})
            trainer = Trainer(cfg)
            trainer.train()

            ckpt_path = os.path.join(
                output_dir, "checkpoint-last", "models", "regressor.pt"
            )
            self.assertTrue(os.path.exists(ckpt_path))

            state = torch.load(ckpt_path, map_location="cpu")

            expected_keys = set(TinyRegressor().state_dict().keys())
            self.assertEqual(set(state.keys()), expected_keys)
            self.assertFalse(
                any(k.startswith("model.") for k in state.keys()),
                f"save_submodule should strip wrapper prefix, got keys: {list(state.keys())}",
            )

            wrapper = WrappedRegressor(input_dim=4, hidden_dim=8, output_dim=2)
            result = wrapper.model.load_state_dict(state, strict=True)
            self.assertEqual(getattr(result, "missing_keys", []), [])
            self.assertEqual(getattr(result, "unexpected_keys", []), [])

            for name, param in wrapper.model.named_parameters():
                self.assertTrue(
                    torch.equal(param, state[name]),
                    f"round-trip mismatch for {name}",
                )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_save_submodule_loads_into_inner_module(self):
        import tempfile as _tempfile

        ckpt_path = _tempfile.mktemp(suffix=".pt")
        try:
            torch.manual_seed(31)
            src = WrappedRegressor(input_dim=4, hidden_dim=8, output_dim=2)
            torch.save(src.model.state_dict(), ckpt_path)

            cfg = OmegaConf.create(
                {
                    "wrapper": {
                        "target": "test_assets.models.WrappedRegressor",
                        "params": {
                            "input_dim": 4,
                            "hidden_dim": 8,
                            "output_dim": 2,
                        },
                        "checkpoint": ckpt_path,
                        "save_submodule": "model",
                        "strict": True,
                    }
                }
            )
            components = ComponentManager(cfg).build_all()
            loaded = components.unwrap("wrapper")

            for name, param in src.model.named_parameters():
                self.assertTrue(
                    torch.equal(param, dict(loaded.model.named_parameters())[name]),
                    f"load_checkpoint save_submodule mismatch for {name}",
                )
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    @unittest.skipUnless(
        os.environ.get("RUN_FSDP2_TORCHRUN_SMOKE") == "1",
        "set RUN_FSDP2_TORCHRUN_SMOKE=1 to run the two-process FSDP2 smoke test",
    )
    def test_torchrun_save_submodule_fsdp2_smoke(self):
        torchrun = shutil.which("torchrun")
        if torchrun is None:
            self.skipTest("torchrun is not available")

        output_dir = tempfile.mkdtemp(prefix="fictional-fortnight-save-submodule-fsdp2-")
        try:
            cmd = [
                torchrun,
                "--standalone",
                "--nproc_per_node=2",
                "-m",
                "framework.train",
                "--config",
                "configs/test/tiny_save_submodule_fsdp2.yaml",
                f"experiment.output_dir={output_dir}",
            ]
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)

            ckpt_path = os.path.join(
                output_dir, "checkpoint-last", "models", "fsdp_model.pt"
            )
            self.assertTrue(os.path.exists(ckpt_path))

            state = torch.load(ckpt_path, map_location="cpu")
            expected_keys = set(TinyRegressor().state_dict().keys())
            self.assertEqual(set(state.keys()), expected_keys)
            self.assertFalse(any(k.startswith("model.") for k in state.keys()))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
