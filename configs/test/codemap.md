# configs/test/

## Responsibility

Minimal smoke-test configurations that exercise core framework features (multi-phase training, gradient checkpointing, FSDP2 mixed parallelism, mixed precision, submodule save, regression) using `test_assets/` stand-in modules instead of the removed business modules (`models.*`/`network.*`/`data.*`). Each config is kept as small as possible (1-3 components, 1-2 phases, 2-3 `max_steps`) to validate framework mechanics without requiring full model implementations.

## Design

Six YAML configs, each targeting a specific framework feature:

### tiny_regression.yaml — Regression Baseline
- **Components**: `regressor` (`test_assets.models.TinyRegressor`)
- **Data**: `test_assets.data.LinearRegressionDataset` (length=16, input_dim=4, output_dim=2, seed=7)
- **Loss**: `test_assets.losses.RegressionMSELoss` (weight=1.0)
- **Optimizer**: single `main` (Adam, lr=0.01) over `regressor.trainable`
- **Runtime**: CPU, no distributed, no mixed precision
- **Train**: `max_steps=3`, `max_epochs=2`, `log_every=1`
- **Phases**: single `regression` phase — forward op writes `batch.input -> pred.value` via component call, loss reads `pred.value` + `batch.target`, backward + optimizer step
- **Logging**: `tensorboard.enabled=false`, `reduce_metrics=false`

### tiny_gradient_checkpointing.yaml — Gradient Checkpointing Bool Shorthand
- Identical structure to `tiny_regression.yaml` except:
  - Uses `test_assets.models.CheckpointedRegressor` (has `gradient_checkpointing_enable` method)
  - Sets `gradient_checkpointing: true` (bool shorthand) on the component
  - `seed=7`, `output_dir=/tmp/fictional-fortnight-gc-smoke`
- Exercises `ComponentManager.apply_gradient_checkpointing()` auto-detection of the `gradient_checkpointing_enable` method

### tiny_dmd_two_phase.yaml — Two-Phase Training (GAN/DMD Pattern)
- **Components**: `generator` + `discriminator` (both `test_assets.models.TinyRegressor`)
- **Data**: `LinearRegressionDataset` (seed=13)
- **Losses**: 4 losses — `generator_reconstruction` (weight=0.25), `generator_dmd` (weight=1.0), `discriminator_real` (weight=1.0), `discriminator_fake` (weight=1.0) — all `RegressionMSELoss`
- **Optimizers**: `generator` (Adam) over `generator.trainable`, `discriminator` (Adam) over `discriminator.trainable`
- **Runtime**: CPU, no distributed
- **Phases**:
  - Phase 1 `generator`: trains generator, freezes discriminator. Ops: `generate_fake` (generator forward), `score_fake_for_generator` (discriminator forward on generator output), `generator_real_label` (ones_like target). Losses: reconstruction + DMD. Backward + step on generator optimizer.
  - Phase 2 `discriminator`: trains discriminator, freezes generator. Ops: `generate_fake_for_discriminator` (generator forward with `no_grad: true`), `score_real` (discriminator on real), `score_fake` (discriminator on fake with `detach_inputs`), `discriminator_real_label` (ones_like), `discriminator_fake_label` (zeros_like). Losses: real + fake. Backward + step on discriminator optimizer.
- Exercises: multi-phase alternation, `no_grad` for inference-only forward, `detach_inputs` on op inputs, `make_tensor` ops for label generation, separate optimizer per phase group, context key isolation across phases (`dmd.*` namespace)

### tiny_mixed_fsdp2.yaml — FSDP2 with Mixed Parallel Strategies
- **Components** (3, demonstrating 3 parallel strategies):
  - `fsdp_model` (`TinyRegressor`): `parallel.strategy=fsdp2`, `fsdp.wrap_modules=[net.0]`, `train.strategy=full`
  - `adapter` (`TinyRegressor`): `parallel.strategy=ddp`, `train.strategy=full`
  - `frozen_teacher` (`TinyRegressor`): `parallel.strategy=replicated`, `train.strategy=frozen`, `mode=eval`
- **Runtime**: `device=auto`, `distributed.strategy=fsdp2`, `fsdp.default_non_fsdp_trainable=ddp`, `fsdp.checkpoint.format=full_rank0`
- **Data**: `LinearRegressionDataset` (seed=17)
- **Phases**: single `mixed` phase — ops chain `fsdp_model` (batch.input -> hidden.value) -> `adapter` (hidden.value -> pred.value)
- **Loss**: single `mse` over `pred.value` + `batch.target`, backward + step on `main` optimizer
- **Logging**: `reduce_metrics=true`
- Exercises: FSDP2 sharding with per-component wrap_modules, mixed parallel strategies (FSDP2 + DDP + replicated) in same training program, frozen component under FSDP2, chained FSDP2->DDP forward, full_rank0 checkpoint format

### tiny_save_submodule.yaml — Submodule Save (Unwrapped)
- **Components**: `regressor` (`test_assets.models.WrappedRegressor` — wraps a `TinyRegressor` as `.model` submodule), with `save_submodule: model`
- Same structure as `tiny_regression.yaml` otherwise
- Exercises: `save_submodule` strips the wrapper prefix from checkpoint `state_dict` keys (e.g. `model.net.0.weight` -> `net.0.weight`), enabling round-trip load into the inner module directly. Test verifies saved keys match `TinyRegressor()` state_dict keys exactly (no `model.` prefix).

### tiny_save_submodule_fsdp2.yaml — Submodule Save under FSDP2
- **Components**: `fsdp_model` (`WrappedRegressor`), `parallel.strategy=fsdp2`, `save_submodule=model`, `fsdp.wrap_modules=[model]`
- **Runtime**: FSDP2, `checkpoint.format=full_rank0`
- Exercises: `save_submodule` combined with FSDP2 wrapping — the wrapper prefix is stripped after FSDP2 checkpoint gathering (full_rank0 format), and the saved state_dict keys match the inner `TinyRegressor` structure without `model.` prefix

### Common Shape

Every config shares:
- **Data**: `LinearRegressionDataset` returning `{"input": (B, input_dim), "target": (B, output_dim), "sample_id": (B,)}` batches
- **Models**: `TinyRegressor` (3-layer MLP: Linear-Tanh-Linear) or its variants (`CheckpointedRegressor`, `WrappedRegressor`)
- **Loss**: `RegressionMSELoss` returning `{"loss": mse_scalar, "mae": mae_scalar}`
- **Dataloader**: `batch_size=4`, `num_workers=0`, `shuffle=false` (deterministic)
- **Checkpoint**: `save_every_steps=0` (only last), `save_optimizer=true`
- **Train**: `log_every=1` (full visibility in test output)

## Data & Control Flow

### Smoke Test Loading Path

```
tests/test_smoke_training.py
  -> framework.config.load_config("configs/test/<name>.yaml")
  -> OmegaConf + imports/includes/extends resolution
  -> framework.engine.Trainer(cfg).train()    # or manual ComponentManager + PhaseRunner
```

### Typical Phase Flow (single-phase configs)

```
[DataLoader] batch dict -> TrainContext.batch
  |
  v
[Phase] ops:
  1. component.forward(x=batch.input)
     -> writes {"prediction": ..., "mean_prediction": ...} to context
     -> CallOp writes cfg.outputs keys to ctx, e.g. "pred.value", "pred.mean"
  |
  v
[Phase] losses:
  1. RegressionMSELoss(inputs={"prediction": "pred.value", "target": "batch.target"})
     -> reads ctx["pred.value"] and ctx["batch.target"]
     -> returns {"loss": Tensor, "mae": Tensor}
  |
  v
[Phase] backward + optimizer step:
  1. loss.backward()
  2. optimizer.step()
```

### Two-Phase Flow (tiny_dmd_two_phase.yaml)

```
Step:
  Phase 1 (generator):
    zero_grad(generator)
    generate_fake(gen)            -> dmd.fake
    score_fake_for_generator(disc) -> dmd.fake_score_for_generator
    make_tensor(ones_like)        -> dmd.real_label_for_generator
    losses: gen_recon + gen_dmd
    backward + step(generator)

  Phase 2 (discriminator):
    zero_grad(discriminator)
    generate_fake_for_discriminator(gen, no_grad) -> dmd.fake_for_discriminator
    score_real(disc)              -> dmd.real_score
    score_fake(disc, detach_inputs) -> dmd.fake_score
    make_tensor(ones_like)        -> dmd.real_label
    make_tensor(zeros_like)       -> dmd.fake_label
    losses: disc_real + disc_fake
    backward + step(discriminator)
```

### FSDP2 Flow (tiny_mixed_fsdp2.yaml)

```
[Components built] -> ComponentManager.build_all()
  -> instantiate fsdp_model (TinyRegressor)
  -> instantiate adapter (TinyRegressor)
  -> instantiate frozen_teacher (TinyRegressor)

[apply_parallel] (inside Trainer)
  -> fully_shard(fsdp_model, submodules=[net.0])
  -> DDP(adapter)
  -> replicated(frozen_teacher)  # no gradient needed

[Phase runner]
  fsdp_model(x)     -> hidden.value      # FSDP2 allgather + compute + reduce
  adapter(hidden)   -> pred.value        # DDP forward
  loss = mse(pred, target)               # FSDP2 backward reduces grads
  optimizer.step()                        # FSDP2 sharded optimizer state
```

## Integration Points

### Consumed By

- **`tests/test_smoke_training.py`** — Main consumer. Specific config-to-test mappings:
  - `tiny_regression.yaml`:
    - `test_dataset_batch_matches_context_contract` — loads config, checks batch shapes
    - `test_phase_runner_updates_model_and_reports_loss_metrics` — manual ComponentManager + PhaseRunner, asserts loss metrics and parameter updates
    - `test_trainer_runs_and_writes_checkpoint` — full Trainer.run(), checks checkpoint files
  - `tiny_gradient_checkpointing.yaml`:
    - `test_gradient_checkpointing_integration_via_phase_runner` — manual build + `apply_gradient_checkpointing()`, verifies `regressor.use_gc=True`, gradients flow, parameters update
  - `tiny_mixed_fsdp2.yaml`:
    - `test_torchrun_mixed_fsdp2_smoke` — `RUN_FSDP2_TORCHRUN_SMOKE=1` gated, launches `torchrun --nproc_per_node=2`, validates return code and checkpoint existence
  - `tiny_save_submodule.yaml`:
    - `test_save_submodule_strips_wrapper_prefix_and_round_trips` — full Trainer.run(), loads saved checkpoint, verifies no `model.` prefix in keys, round-trip load into inner `TinyRegressor`
    - `test_save_submodule_loads_into_inner_module` — programmatic config via OmegaConf.create(), verifies `save_submodule` checkpoint loading into inner module
  - `tiny_save_submodule_fsdp2.yaml`:
    - `test_torchrun_save_submodule_fsdp2_smoke` — `RUN_FSDP2_TORCHRUN_SMOKE=1` gated, `torchrun` subprocess, asserts saved keys match `TinyRegressor` keys with no prefix

- **`test_assets/`** — Provides stand-in components:
  - `test_assets.models`: `TinyRegressor` (simple MLP), `WrappedRegressor` (wrapper + inner model for save_submodule), `CheckpointedRegressor` (with `gradient_checkpointing_enable`/`enable_gradient_checkpointing` methods)
  - `test_assets.data`: `LinearRegressionDataset` (deterministic, returns `input`/`target`/`sample_id` dicts)
  - `test_assets.losses`: `RegressionMSELoss` (returns `loss` + `mae` dict)

### Depends On

- **`framework/`** — All framework modules:
  - `framework/config.py`: `load_config()` — YAML loading with imports/includes/extends resolution
  - `framework/engine.py`: `Trainer`, `build_dataloader` — training loop entry point
  - `framework/components.py`: `ComponentManager`, `ComponentEntry` — lifecycle, DDP/FSDP2 wrapping, checkpointing
  - `framework/phase_runner.py`: `PhaseRunner` — per-phase op/loss/backward/step execution
  - `framework/losses.py`: `build_losses` — loss construction from config
  - `framework/optim.py`: `build_optimizers` — optimizer construction
  - `framework/ops/common.py`: `CallOp`, `make_tensor` op types
  - `framework/resolver.py`: `resolve_kwargs` — context input resolution for ops/losses
  - `framework/context.py`: `TrainContext` — dotted-path state container
  - `framework/distributed.py`: `DistState`, `fully_shard_module` — parallelism

### Run Commands

```bash
# Run all smoke tests (single-process)
python3 -m unittest discover -s tests -v

# Run a specific test
python3 -m unittest tests.test_smoke_training.SmokeTrainingTest.test_trainer_runs_and_writes_checkpoint

# Run FSDP2 multi-process tests (requires 2 GPUs/CPUs)
RUN_FSDP2_TORCHRUN_SMOKE=1 python3 -m unittest tests.test_smoke_training.SmokeTrainingTest.test_torchrun_mixed_fsdp2_smoke

# Run a single config directly via framework.train
uv run python -m framework.train --config configs/test/tiny_regression.yaml train.max_steps=2

# Syntax and import validation
PYTHONPYCACHEPREFIX=/tmp/fictional-fortnight-pycache python3 -m compileall -q configs/test
```
