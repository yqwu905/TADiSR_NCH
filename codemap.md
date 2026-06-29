# Repository Atlas: TADiSR_NCH

## Project Responsibility

A modular, configuration-driven PyTorch training framework (`futuretrainer`) used to reproduce TADiSR (Text-Aware Real-World Image Super-Resolution, NeurIPS 2025). The framework decouples model definition, data loading, loss computation, optimization, and distributed execution into independently configurable YAML-driven components. Business modules implementing the TADiSR pipeline — NCH MMDiT DiT transformer (`models/dit/`), VAE encoder/decoder + JSD joint segmentation decoder (`models/vae/`), offline text embedding + trainable tokens (`models/text_encoder/`), TADiSR dataset (`data/`), SR/segmentation losses (`models/loss/`), and data generation/analysis scripts (`scripts/`) — are all present in the repository. Training configs in `configs/tadisr_*.yaml` are ready for training given appropriate checkpoints, embedding cache, and dataset paths. The `configs/test/` smoke-test configs use `test_assets/` stand-in modules and also run.

## System Entry Points

| File | Role |
|---|---|
| `framework/train.py` | CLI entry. Parses `--config` + dotlist overrides, calls `load_config()`, creates `Trainer`, runs `train()`, cleans up distributed. |
| `framework/engine.py` | `Trainer` class — the main orchestrator. Distributed init, DataLoader, component building, phase loop, logging, checkpointing. |
| `framework/config.py` | `load_config()` — resolves file-level `imports`/`includes`/`include` and node-level `extends`/`_base_` into a single merged `DictConfig`. |
| `pyproject.toml` | Project manifest (`futuretrainer` 0.1.0). Python >=3.10. Core deps: torch 2.9.0, diffusers, omegaconf, einops, safetensors, tensorboard. Optional groups: `npu` (torch_npu), `logging` (aim, wandb). |
| `AGENTS.md` | Primary project documentation: structure, framework model, config spec, coding conventions, run/debug commands, known state. |
| `README.md` | Currently empty; `AGENTS.md` is the main entry. |

## Framework Model

Training is composed of four YAML-defined object types, orchestrated through a shared `TrainContext` data bus (dotted-path key-value store):

1. **`components`** — trainable/non-trainable modules (VAE, DiT, text encoder, embedding DB). Each built via `target` (full Python import path) + `params`, with optional `checkpoint`, `train.strategy` (`full`/`frozen`/`keep`), `mode`, `save`, `gradient_checkpointing`.
2. **`train_program.phases`** — one or more phases per step. Each phase defines trainable/frozen components, mode, op list, loss list, optimizer actions, mixed precision.
3. **`ops`** — data-flow units. An op reads inputs from `TrainContext` (via `resolve_kwargs`), calls a component/function, writes outputs back to context.
4. **`losses`** — reads from context, computes weighted loss, returns metrics for logging.

Control flow: `train.py` -> `load_config` -> `Trainer.__init__` (distributed + components + optimizers + losses + `PhaseRunner`) -> `train()` loop (per step: build `TrainContext`, iterate phases, each phase runs ops -> losses -> backward -> grad clip -> optimizer/scheduler step -> log -> checkpoint).

## Directory Map (Aggregated)

| Directory | Responsibility Summary | Detailed Map |
|---|---|---|
| `framework/` | Reusable, configuration-driven PyTorch training library. Orchestrates config resolution, component lifecycle, phase-based training loop, op/loss registration, context resolver, and distributed strategy (none/DDP/FSDP2). | [framework/codemap.md](framework/codemap.md) |
| `framework/ops/` | Built-in op registration point and data-flow primitives. Ops read from `TrainContext`, invoke components/functions, write results back. Registry pattern via `@register_op`. Includes `common.py` (call/make_tensor/set_value/detach/save_image) and `diffusion.py` (sample_timestep/flow_matching_prepare/dmd_proxy_target/nch_ldm_v3_two_step). | [framework/ops/codemap.md](framework/ops/codemap.md) |
| `configs/` | YAML configuration tree defining full training experiments. Top-level configs orchestrate components, phases, ops, losses, optimizers, data, runtime, logging. Composition via `imports`/`includes`/`extends`/`_base_`. | [configs/codemap.md](configs/codemap.md) |
| `configs/data_config/` | Reusable dataset/dataloader configuration fragments imported by top-level configs via `imports` into `data.train.dataset`. | [configs/data_config/codemap.md](configs/data_config/codemap.md) |
| `configs/model_config/` | Reusable model component fragments (DiT/VAE/embedding definitions) imported into `components.*`. Contains the `SR_nch_v3_ti2i_with_guidance_f16c64.yaml` DiT config and a `text_encoder/` subdirectory. | [configs/model_config/codemap.md](configs/model_config/codemap.md) |
| `configs/model_config/text_encoder/` | Text encoder sub-fragments producing `encoder_hidden_states` conditioning. `offline_embedding.yaml` (EmbeddingDB with SQLite cache) composes `nch_trainable_vector.yaml` (multitask trainable token vectors). | [configs/model_config/text_encoder/codemap.md](configs/model_config/text_encoder/codemap.md) |
| `configs/network_config/` | Reusable network/backbone fragments imported into `components.*`. `ldm_nch_v3.yaml` defines the `NCH_LDM_V3` LDM backbone delegating architecture to a `model_cfg_path`. | [configs/network_config/codemap.md](configs/network_config/codemap.md) |
| `configs/test/` | Minimal smoke-test configs exercising framework features (multi-phase, gradient checkpointing, FSDP2 mixed parallelism, mixed precision, submodule save, regression) using `test_assets/` stand-in modules. Consumed by `tests/test_smoke_training.py`. | [configs/test/codemap.md](configs/test/codemap.md) |

### Not mapped (excluded by policy)

- `tests/` — smoke tests (`test_smoke_training.py`); documented as a consumer in `configs/test/codemap.md`.
- `test_assets/` — stand-in components (`TinyRegressor`, `LinearRegressionDataset`, `RegressionMSELoss`) used by smoke tests; documented as a dependency in `configs/test/codemap.md`.
- `models/` — TADiSR business modules (DiT, VAE, JSD, text encoder, loss); structured by domain, not separately mapped.
- `data/` — TADiSR dataset implementation (`tadisr_dataset.py`); single-file module.
- `scripts/` — auxiliary scripts (`generate_tadisr_data.py`, `analyze_pangu_text_token.py`); standalone entry points.
- `.venv/`, `.ruff_cache/`, `.codegraph/`, `output/` — environment, caches, build artifacts (gitignored or local-only).

## Cross-Reference: How Modules Connect

```
CLI: python -m framework.train --config configs/<exp>.yaml [dotlist overrides]
  |
  v
framework/config.py            load_config() -> imports/includes/extends resolution
  |
  v
framework/engine.py (Trainer)  init_distributed -> build_dataloader -> ComponentManager.build_all
  |                            -> apply_gradient_checkpointing -> apply_parallel (DDP/FSDP2)
  |                            -> build_optimizers -> build_schedulers -> build_losses -> PhaseRunner
  |
  v
framework/phase_runner.py      per phase: set_phase_state -> ops (framework/ops/*) -> losses
  |                            -> backward -> clip_grad -> optimizer.step -> scheduler.step
  |
  v
framework/context.py           TrainContext: dotted-path data bus shared by ops & losses
framework/resolver.py          resolve_kwargs/resolve_input: ctx/const/ones_like/zeros_like/
                              randn_like/detach/cast/getattr input specs
framework/components.py        target+params instantiation, checkpoint, freeze/thaw, DDP/FSDP2 wrap
framework/distributed.py       DistState, wrap_ddp, fully_shard_module, device inference (cuda/npu/cpu)
framework/loggers.py           LoggerCollection: TensorBoard/Aim/Wandb backends + image logging
framework/optim.py             build_optimizers/build_schedulers, param selection by component.trainable
framework/losses.py            LossWrapper: maps context -> loss fn args, weighted loss + metrics
framework/registry.py          @register_op / OPS dict / get_op
framework/instantiate.py       instantiate(): universal target+params factory via locate()
framework/utils.py             StepTimer: context-manager profiler with distributed reduction
```

## Known State

- `framework/`, `models/`, `data/`, `scripts/` all pass syntax-level compile (`python3 -m compileall -q framework test_assets tests models data scripts`).
- TADiSR training pipeline is complete through 5 stages (see `TODO.md`): SR baseline, TACA text-aware attention, JSD joint segmentation decoder, full loss + joint training config, and Colab GPU verification (512/1024 resolution, 37-layer DiT).
- `configs/tadisr_*.yaml` configs are ready for training; they require DiT/VAE checkpoints, a SQLite embedding cache, and FTSR dataset paths.
- `framework/ops/common.py` `save_image` op is **broken**: references undefined `save_image`, `_ensure_list`, `_basename_without_ext`. See [framework/ops/codemap.md](framework/ops/codemap.md).
- `configs/f16c64_vae_dit_proj_out_align.yaml` and `configs/f16c64_vae_x_embbder_align.yaml` are early framework debug configs with potentially missing import targets; use `configs/tadisr_*.yaml` for actual training.
- `configs/test/` smoke configs run via `python3 -m unittest discover -s tests -v` (FSDP2 tests gated by `RUN_FSDP2_TORCHRUN_SMOKE=1`).

## Quick Commands

```bash
# Sync environment
uv sync

# Framework syntax check
PYTHONPYCACHEPREFIX=/tmp/fictional-fortnight-pycache python3 -m compileall -q framework test_assets tests models data scripts

# Smoke tests
python3 -m unittest discover -s tests -v

# Single-process debug run (smoke config)
uv run python -m framework.train --config configs/test/tiny_regression.yaml train.max_steps=2

# List registered ops
uv run python -c "import framework.ops; from framework.registry import OPS; print(sorted(OPS))"
```
