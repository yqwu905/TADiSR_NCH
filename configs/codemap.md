# configs/

## Responsibility

YAML configuration tree defining experiments for the modular PyTorch training framework. Each top-level YAML describes the full training run: components (model modules with target/params/checkpoint/mode), train_program.phases (per-step op sequences, loss evaluation, optimizer actions), ops (data-flow steps reading/writing TrainContext), losses (weighted objectives with input mappings), optimizers (type, lr, parameter selection by component name), data loaders, runtime (distributed strategy, mixed precision), logging backends, image logging, checkpoint schedule, and profiling.

The config layer (`framework/config.py`) is the sole entry point: it resolves file-level `imports`, `includes`/`include`, and node-level `extends`/`_base_` inheritance, producing a single merged dict consumed by `Trainer`/`engine.py`.

## Design

**Composition mechanisms:**

- **File-level `imports`**: Maps a dotted config path to an external YAML file. The imported file's content is merged at that path. Example: `imports: { components.dit: configs/network_config/ldm_nch_v3.yaml }` populates `config["components"]["dit"]` from the fragment. Relative paths resolve from the run directory (repo root by default).
- **File-level `includes` / `include`**: Alternative merge mechanism (less used; `imports` is the primary pattern).
- **Node-level `extends`**: A node specifies a base node to inherit from, then overrides selected sub-keys. Example: `my_node: { extends: base_node, params: { override_arg: value } }`.
- **Node-level `_base_`**: Similar inheritance anchor.

**Directory layout:**

```
configs/
  codemap.md                        # this file
  f16c64_vae_dit_proj_out_align.yaml   # experiment: flare removal with frozen VAE + DiT
  f16c64_vae_x_embbder_align.yaml      # experiment: x-embedder knowledge distillation
  data_config/                         # dataset fragment YAMLs (imported into data.train.dataset)
    codemap.md
    0612_deflare_1k_filtered.yaml      # (referenced by imports; may not exist in current tree)
    infer_simple_dataset.yaml
  model_config/                        # model/component fragment YAMLs (imported into components.*)
    codemap.md
    text_encoder/
    SR_nch_v3_ti2i_with_guidance_f16c64.yaml
    SR_nch_v3_ti2i_with_guidance_x_embedder_only.yaml
  network_config/                      # network backbone fragment YAMLs (imported into components.*)
    codemap.md
    ldm_nch_v3.yaml
  test/                                # minimal smoke-test configs
    codemap.md
    tiny_dmd_two_phase.yaml
    tiny_gradient_checkpointing.yaml
    tiny_mixed_fsdp2.yaml
    tiny_regression.yaml
    tiny_save_submodule.yaml
    tiny_save_submodule_fsdp2.yaml
```

**Two top-level experiment configs:**

1. **`f16c64_vae_dit_proj_out_align.yaml`** -- Flare removal / image restoration pipeline.
   - Imports: `data.train.dataset` from `data_config/`, `components.dit` from `network_config/`, `components.offline_embedding` from `model_config/text_encoder/`.
   - Components: frozen `vae_encoder` (models.vae.npu.f16c64.VaeEncoder), frozen `vae_decoder` (models.vae.npu.f16c64.VaeDecoder), `offline_embedding` (text encoder), `dit` (DiT backbone, trainable via `keep` strategy).
   - Single phase `generator`: vae_encode -> text_encode -> dit -> vae_decode -> loss.
   - Loss: `Flare7KPlusMaskLossV2` on RGB with L1/LPIPS/masked components.
   - Optimizer: AdamW on `dit` only, lr=1e-5.
   - Runtime: DDP, no mixed precision, batch_size=1, 100k steps.

2. **`f16c64_vae_x_embbder_align.yaml`** -- x-embedder alignment (knowledge distillation between old and new x-embedder).
   - Imports: `data.train.dataset` from `data_config/`, `components.dit_old_x_embedder` and `components.dit_new_x_embedder` from `model_config/`.
   - Components: frozen `f16c64_vae_encoder`, frozen `f16c32_vae_encoder`, frozen `dit_old_x_embedder` (teacher), trainable `dit_new_x_embedder` (student).
   - Single phase `generator`: two parallel VAE encode branches -> two parallel x-embedder forward passes -> MSE loss between outputs.
   - Optimizer: AdamW on `dit_new_x_embedder` only, lr=4e-5, grad clip max_norm=10.
   - Runtime: DDP, no mixed precision, batch_size=4, 100k steps.

**Known broken targets (modules removed from repo):**
- `models.vae.npu.f16c64.VaeEncoder`, `models.vae.npu.f16c64.VaeDecoder`
- `models.vae.rdp.f16c32.VaeEncoder`
- `models.loss.flare7pp_loss.Flare7KPlusMaskLossV2`
- Import fragments (e.g., `configs/data_config/0612_deflare_1k_filtered.yaml`, `configs/network_config/ldm_nch_v3.yaml`, `configs/model_config/text_encoder/offline_embedding.yaml`, `configs/model_config/SR_nch_v3_ti2i_with_guidance_*.yaml`) reference `models.*`, `network.*`, `data.*` packages that are not present in the current repository. The configs serve as structural references only; they cannot be run without restoring those business modules.

## Data & Control Flow

```
CLI: uv run python -m framework.train --config configs/foo.yaml [dotlist overrides]
  |
  v
framework/config.py: load_config("configs/foo.yaml")
  |-- Resolves file-level imports (recursive): merges imported YAML into dotted paths
  |-- Resolves includes/include
  |-- Resolves node-level extends/_base_
  |-- Applies command-line dotlist overrides
  |-- Returns merged dict
  |
  v
framework/train.py: Trainer.__init__(cfg)
  |-- framework/components.py: ComponentManager(cfg.components) -> build_all()
  |     instantiates each component via target + params, loads checkpoints, wraps DDP
  |-- framework/optim.py: build_optimizers(cfg.optimizers, components)
  |     creates optimizer instances, selects parameters by component name includes
  |-- DataLoader: from cfg.data.train.dataloader + resolved dataset (cfg.data.train.dataset)
  |     dataset instantiated via target + params (imported fragment)
  |
  v
framework/engine.py: train loop (max_steps)
  | per step:
  |   for each phase in cfg.train_program.phases:
  |     set trainable/frozen components, mode
  |     for each op: resolve_kwargs(inputs) from TrainContext, call component/function, ctx.set(output)
  |     for each loss: resolve_kwargs(inputs) from TrainContext, compute loss, accumulate
  |     backward, clip_grad, optimizer.step(), scheduler.step()
  |     log metrics, write TensorBoard images
  |
  v
Output: checkpoints, logs, TensorBoard events in cfg.experiment.output_dir
```

The `imports` mechanism is the primary way top-level configs pull in reusable fragments. For example, `f16c64_vae_dit_proj_out_align.yaml` has three imports that populate `data.train.dataset`, `components.dit`, and `components.offline_embedding` from separate YAML files in the subdirectories. This separates concerns: data configs live in `data_config/`, model components in `model_config/` and `network_config/`, while the top-level config orchestrates the full experiment.

## Integration Points

**Consumed by:**
- `framework/train.py` -- CLI entry point, accepts `--config <path>`.
- `framework/config.py` -- `load_config()` resolves imports/includes/extends.
- `framework/engine.py` -- `Trainer` consumes merged config dict for component building, optimizer construction, data loading, phase execution, logging, checkpointing.
- `framework/components.py` -- `ComponentManager` uses `components.*` entries (target, params, checkpoint, train, mode, save, gradient_checkpointing).
- `framework/optim.py` -- `build_optimizers()` uses `optimizers.*` to create optimizers with parameter selection.
- `framework/phase_runner.py` -- executes `train_program.phases` ops and losses.
- `framework/ops/` -- op implementations referenced by `ops.*.component` and `ops.*.method`.
- `framework/losses.py` -- loss wrapper uses `losses.*.inputs` to map TrainContext keys to loss function arguments.
- `framework/resolver.py` -- `resolve_kwargs()` interprets `inputs` specs (`ctx`, `const`, `ones_like`, etc.) for ops and losses.

**Subdirectory codemaps:**
- `configs/data_config/codemap.md` -- dataset fragment YAMLs.
- `configs/model_config/codemap.md` -- model component fragment YAMLs (text encoders, SR backbones, x-embedder configs).
- `configs/network_config/codemap.md` -- network backbone fragment YAMLs (e.g., DiT LDM variants).
- `configs/test/codemap.md` -- smoke-test minimal configs for CI/validation.

**External (missing) targets (not in current repo, need restoration):**
- `models.vae.npu.f16c64.*`, `models.vae.rdp.f16c32.*` -- VAE implementations.
- `models.loss.flare7pp_loss.*` -- loss implementations.
- `network.*` -- network module references in imported fragments.
- `data.*` -- dataset implementations referenced by data_config fragments.
