# framework/

## Responsibility

The `framework/` directory is a reusable, configuration-driven PyTorch training library that decouples model definition, data loading, loss computation, optimization, and distributed execution into independently configurable components. It provides the orchestration layer: a user writes YAML configs specifying what components to build, what ops to run, what losses to apply, and how to parallelize — the framework handles instantiation, checkpointing, logging, phase execution, and distributed setup.

## Design

The framework is assembled from several cooperating modules, each with a distinct design pattern:

### Config Resolution (`config.py`)
- **Recursive merge pattern**: `load_config()` calls `_load_config_recursive()` which supports file-level `imports`/`includes`/`include` keys. Each import is merged via `OmegaConf.merge`; if a target path is given (dict form), it merges into a nested subtree.
- **Node-level inheritance**: `_resolve_extends()` processes `extends`/`_base_` keys node-by-node, performing a base-first merge with cycle detection via a `resolving_stack`.
- Input: YAML file path. Output: fully resolved `DictConfig`.

### Instantiation (`instantiate.py`)
- **Factory pattern**: `instantiate(cfg)` reads `target` (full Python import path), calls `locate()` to import the class/function, then constructs it with `params` + optional extra kwargs. Single entry point for all object creation.
- `locate()` uses `importlib.import_module` + `getattr`. No configuration injection beyond `params`.

### Registry (`registry.py`)
- **Registry pattern**: `OPS` global dict maps op type strings to callable classes/functions. `@register_op(name)` is the decorator that populates it. `get_op(name)` does a lookup with error message listing all registered ops.
- Used by ops only (see `framework/ops/codemap.md`).

### Component Lifecycle (`components.py`)
- `ComponentManager`: Container and lifecycle manager. `build_all()` iterates config items, for each: `instantiate()`, load checkpoint, apply train policy, snapshot trainable flags. Supports parallel building via `ThreadPoolExecutor`.
- `ComponentEntry`: Dataclass wrapping a built module with its config, trainable flags, parallel strategy, and gradient checkpointing state.
- Train policies: `full` (all trainable), `frozen` (all frozen), `keep` (no change).
- Checkpoint loading: handles both raw `state_dict` and dict-with-`state_dict` key. Optional `save_submodule` for loading into submodules.
- Gradient checkpointing: probes `gradient_checkpointing_enable` / `enable_gradient_checkpointing` automatically, or uses explicit method name + kwargs from config.
- **Strategy pattern for parallelism**: `apply_parallel()` dispatches to DDP wrapping (`wrap_ddp`), FSDP2 (`fully_shard`), replicated (broadcast), or none based on resolved strategy per component. `resolve_parallel_strategy()` uses component-level `parallel.strategy`, per-component `fsdp.enabled`, or global defaults.
- Phase state management: `set_phase_state()` freezes/thaws modules, sets train/eval modes per phase. `_restore_component_trainable_flags()` re-applies original trainable flags when a component is in the `trainable` list.

### Context (`context.py`)
- `TrainContext`: Simple dotted-path key-value store. `set("a.b.c", v)` creates nested dicts. `get("a.b.c")` traverses them. `keys()` flattens all leaf paths. `update_dict(prefix, data)` bulk-sets under a prefix. Used as the sole data conduit between ops and losses.

### Resolver (`resolver.py`)
- **Strategy pattern for input resolution**: `resolve_input(spec, ctx)` handles multiple input specifications:
  - `str`: direct context lookup via `ctx.get()`.
  - `{type: "ctx", key: "..."}`: explicit context key.
  - `{type: "const", value: ...}`: literal value.
  - `{type: "ones_like", ref: "..."}`, `zeros_like`, `full_like`, `randn_like`: synthetic tensors.
  - `{type: "detach", ref: "..."}`: detached tensor.
  - `{type: "cast", value: ..., dtype: ..., device_ref: ...}`: type/device cast.
  - `{type: "getattr", object: ..., name: ...}`: attribute access on a resolved object.
- `resolve_kwargs(inputs_cfg, ctx)` applies `resolve_input` to each key in a config mapping.

### Phase Runner (`phase_runner.py`)
- **Template Method pattern**: `PhaseRunner.run()` defines the invariant skeleton of a training phase:
  1. `components.set_phase_state()` — freeze/thaw/mode per component.
  2. Optional `every_n_steps` gating.
  3. `zero_grad` for specified optimizers.
  4. **Forward**: iterate ops (via `get_op( op_type )(op_cfg)`) then run losses.
  5. `backward` on accumulated loss (with optional `loss_scale`).
  6. `clip_grad_norm_` for parameter groups.
  7. `optimizer.step()` and `scheduler.step()` for named optimizers/schedulers.
- Mixed precision via `torch.autocast` context manager, configurable per phase as `bf16`, `fp16`, or off.
- Op iteration: supports both list-of-dicts and dict-of-dicts config formats. Each op is instantiated fresh per step from its config.

### Loss Wrapper (`losses.py`)
- `LossWrapper(nn.Module)`: Adapter that maps context data to loss function arguments. `forward(ctx)` resolves inputs via `resolve_input`, calls the underlying loss function, multiplies by weight, and returns `(loss, metrics_dict)`. Metrics are prefixed with `{loss_name}/`.
- `enabled(ctx)`: Supports step-based gating (`after_step`, `before_step`, `every_n_steps`).
- `build_losses()`: Instantiates each loss via `instantiate(cfg)` and wraps in `LossWrapper`.

### Optimizer & Scheduler (`optim.py`)
- `collect_params(param_cfg, components)`: Resolves `include` entries — strings ending in `.trainable` filter to trainable params of a component; bare component names include all trainable params.
- `build_optimizers()`: Supports builtin types (`adamw`, `adam`) and custom via `target` + `params` (Factory pattern).
- `build_schedulers()`: Supports `cosine` (with optional warmup) and custom via `target` + `params`.

### Engine (`engine.py` — class `Trainer`)
- Main orchestrator. Takes fully-resolved config, calls `init_distributed()`, builds dataloader, builds components, applies gradient checkpointing, applies parallel strategy, builds optimizers/schedulers/losses, constructs `PhaseRunner`.
- `train()`: Epoch loop → batch loop → per step:
  1. Create `TrainContext`, set metadata (`global_step`, `epoch`, `rank`, `world_size`, `batch`).
  2. Iterate `train_program.phases`, call `phase_runner.run()` for each.
  3. Aggregate metrics, log scalars to all backends.
  4. Write TensorBoard images from context keys (via `LoggerCollection.log_images_from_context`).
  5. Save checkpoint periodically + at end.
- Checkpoint: Saves `config.yaml`, `trainer_state.pt`, per-component state dicts (with `save` policy: `full`, `lora_only`, `none`), optimizer/scheduler states. FSDP2 supports `full_rank0` and `dcp_sharded` formats.
- Parameter stats: `print_parameter_summary()` logs per-component total/trainable/frozen counts with human-readable formatting.

### Distributed (`distributed.py`)
- `DistState`: Dataclass carrying distributed state (strategy, rank, world_size, device, device_mesh).
- `init_distributed()`: Reads `runtime.distributed.strategy` (`none`, `ddp`, `fsdp2`), infers device type (`cuda`, `npu`, `cpu`), initializes process group, creates device mesh for FSDP2.
- `wrap_ddp()`: Wraps module in `DistributedDataParallel` with configurable `find_unused_parameters`, `broadcast_buffers`, `static_graph`.
- FSDP2 helpers: `build_fsdp2_kwargs()`, `fully_shard_module()`, `register_fsdp2_forward_methods()`.
- `reduce_scalar()`: Distributed all-reduce with mean/op selection.

### Loggers (`loggers.py`)
- **Strategy pattern + Composite pattern**: `MetricLogger` abstract base; concrete backends: `TensorBoardLogger`, `AimLogger`, `WandbLogger`. `LoggerCollection` composites them, delegating `log_metrics()` and `log_images()` to all.
- `build_loggers()`: Parses `logging.backends` config, constructs backends, wraps in `LoggerCollection`. Only main process gets loggers.
- Image logging: Configured via `logging.images.items` — each item specifies a `tag`, a context `key`, and optional `value_range`. Images are normalized (auto-detect range, clamp to [0,1]) and dispatched to all backends.
- `to_log_images()`: Normalizes tensor images from various range conventions (`-1_1`, `0_1`, `0_255`, `auto`).

### Timer (`utils.py` — class `StepTimer`)
- Context-manager based profiler. `time(name)` records wall-clock with optional device synchronization. Supports distributed reduction (`mean`, `max`, `none`). `format()` groups records by category (`step/`, `phase/`, `component/`, `optimizer/`, etc.) and shows percentages relative to a reference total.

### CLI Entry (`train.py`)
- `main()`: Parse `--config` + positional dotlist overrides, call `load_config()`, merge overrides via `OmegaConf.from_dotlist`, create `Trainer(cfg)`, call `train()`, `cleanup_distributed()` in `finally`.

## Data & Control Flow

```
train.py::main()
  |
  +--> config.py::load_config(config_path)
  |      |
  |      +--> _load_config_recursive()              # Resolve imports/includes/include
  |      |      |
  |      |      +--> OmegaConf.load(yaml)            # Parse current file
  |      |      +--> OmegaConf.merge(imported, current)
  |      |      +--> recurse for transitive imports
  |      |
  |      +--> _resolve_extends(cfg)                 # Node-level extends/_base_
  |             |
  |             +--> _resolve_node()                 # DFS traversal, base-first merge
  |
  +--> OmegaConf.merge(cfg, dotlist_overrides)       # CLI overrides override config
  |
  +--> Trainer.__init__(cfg)
  |      |
  |      +--> init_distributed(runtime_cfg)
  |      |      +--> Infer device, parse strategy (none|ddp|fsdp2)
  |      |      +--> dist.init_process_group()       # DDP/FSDP2
  |      |      +--> init_device_mesh()              # FSDP2 only
  |      |      +--> return DistState
  |      |
  |      +--> build_dataloader(cfg.data.train, dist_state)
  |      |      +--> instantiate(dataset_cfg)        # dataset via target+params
  |      |      +--> DistributedSampler if DDP/FSDP2
  |      |      +--> return DataLoader
  |      |
  |      +--> ComponentManager(cfg.components).build_all()
  |      |      +--> For each component:
  |      |      |      instantiate(cfg)              # target + params
  |      |      |      load_checkpoint()             # optionally load .pt
  |      |      |      apply_train_policy()          # full / frozen / keep
  |      |      |      snapshot trainable flags
  |      |      +--> (parallel build via ThreadPoolExecutor if configured)
  |      |
  |      +--> ComponentManager.apply_gradient_checkpointing()
  |      +--> ComponentManager.apply_parallel(dist_state)
  |      |      +--> per component: resolve strategy -> DDP / FSDP2 / replicated / none
  |      |      +--> module.to(device)
  |      +--> ComponentManager.set_initial_modes()
  |      +--> print_parameter_summary()              # main process only
  |      |
  |      +--> build_optimizers(cfg.optimizers, components)
  |      |      +--> collect_params() via include list
  |      |      +--> instantiate or builtin (AdamW/Adam)
  |      |
  |      +--> build_schedulers(cfg.schedulers, optimizers)
  |      |      +--> cosine or custom via target+params
  |      |
  |      +--> build_losses(cfg.losses)
  |      |      +--> instantiate each loss fn
  |      |      +--> wrap in LossWrapper
  |      |
  |      +--> PhaseRunner(components, optimizers, schedulers, losses, ...)
  |
  +--> Trainer.train()
         |
         +--> For epoch in range(max_epochs):
                |
                +--> For batch in train_loader:
                       |
                       |  [Step start]
                       |  move_to_device(batch, device)
                       |  TrainContext.__init__()
                       |  ctx.set("batch", batch)
                       |  ctx.set("global_step", step)
                       |  ctx.set("epoch", epoch)
                       |  ctx.set("rank"/"local_rank"/"world_size", ...)
                       |
                       |  For each phase_cfg in cfg.train_program.phases:
                       |    |
                       |    +--> PhaseRunner.run(ctx, phase_cfg)
                       |           |
                       |           +--> components.set_phase_state(trainable, frozen, modes)
                       |           |
                       |           +--> [optional] zero_grad for named optimizers
                       |           |
                       |           +--> with autocast(mixed_precision):
                       |           |      |
                       |           |      +--> For each op_cfg in phase.ops:
                       |           |      |      op = get_op(op_cfg.type)(op_cfg)  # fresh instance
                       |           |      |      op(ctx, components)               # reads ctx, writes ctx
                       |           |      |
                       |           |      +--> For each loss_name in phase.losses:
                       |           |             loss = self.losses[loss_name]
                       |           |             if loss.enabled(ctx):
                       |           |                 resolved_kwargs = resolve_input(loss.inputs, ctx)
                       |           |                 loss_val, metrics = loss.loss_fn(**resolved_kwargs)
                       |           |                 total_loss += loss_val * loss.weight
                       |           |
                       |           +--> [if backward] total_loss.backward()
                       |           |
                       |           +--> [if clip_grad] clip_grad_norm_(trainable_params)
                       |           |
                       |           +--> optimizer.step() for named optimizers
                       |           +--> scheduler.step() for named schedulers
                       |           |
                       |           +--> return metrics dict (floats)
                       |
                       |  [Post-phase]
                       |  LoggerCollection.log_images_from_context(ctx, step)
                       |  reduce_scalar_tensor(metrics) if log_reduce
                       |  log_metrics() to all backends
                       |  logger.info() step summary
                       |  StepTimer.format() if profiling enabled
                       |
                       |  [Checkpoint]
                       |  if step % save_every_steps == 0:
                       |     save_checkpoint(output_dir, tag=f"step-{step}")
                       |     cleanup_old_checkpoints()
                       |
                       |  global_step += 1
                       |  if step >= max_steps: break
```

### TrainContext as Data Bus

`TrainContext` is the central data bus. Ops write to it (via `ctx.set`), losses read from it (via `resolve_input`/`ctx.get`). Keys are dotted paths grouped by domain:

- `batch.*` — raw data from DataLoader (set by `Trainer.train()`)
- `pred.*`, `noise.*`, `latent.*`, `cond.*` — typical op outputs
- `global_step`, `epoch`, `rank`, `world_size` — metadata (set by `Trainer.train()`)

The contract between config authors and op/loss implementations is purely key-based: if op A writes `latent.z` and loss L reads `latent.z`, they communicate. There is no compile-time checking; missing keys raise at runtime via `ctx.get(..., required=True)`.

## Integration Points

### Registration Decorators
- `@register_op("op_name")` in `framework/registry.py`: Registers callables into `OPS` dict. Used by op implementations in `framework/ops/`. See `framework/ops/codemap.md`.
- Op loading in `phase_runner.py`: `from .ops import *` triggers all `@register_op` decorated classes in `framework/ops/` to populate `OPS`.

### target + params Instantiation
- `instantiate.py::instantiate(cfg)`: The universal factory. Every component, dataset, loss function, custom optimizer, and custom scheduler is created by this function. Contract: `cfg` must have a `target` key (full Python path) and optionally `params` (kwargs dict). No framework injection into constructors — all parameters must be explicit in config.

### Context Key Conventions
- `batch` — set by `Trainer.train()` from DataLoader output. Subkeys are dataloader-dependent (e.g. `batch.gt`, `batch.txt`).
- `global_step`, `epoch`, `rank`, `local_rank`, `world_size` — set by `Trainer.train()`.
- Ops and losses access intermediate results by dotted paths. Config authors must ensure namespace consistency across ops and losses.
- Image logging reads config-defined context keys via `logging.images.items[].key`.

### Checkpoint Load/Save
- **Load**: `ComponentManager.load_checkpoint()` reads a `.pt` file, accepts raw `state_dict` or dict containing `"state_dict"` key. Supports `save_submodule` for loading into nested modules. Uses `strict` parameter.
- **Save**: `Trainer.save_checkpoint()`. Saves `config.yaml`, `trainer_state.pt` (global_step, world_size), per-component state dicts (filtered by `save` policy: `full`, `lora_only`, `none`), and optimizer/scheduler states. FSDP2 variants: `save_fsdp2_checkpoint()` for `full_rank0` format (gathers full state dict on rank 0), and `save_fsdp2_dcp_checkpoint()` for sharded `dcp` format.

### TensorBoard / Logging Hooks
- `build_loggers()` in `loggers.py` reads `logging.backends` (list of dicts with `type`, `enabled`, and backend-specific params). Supports `tensorboard`/`tb`, `aim`, `wandb`.
- Image logging is configured via `logging.images.items[]` — each item has `tag`, `key` (context path), `value_range` (auto/0_1/0_255/-1_1), `max_images`. `LoggerCollection.log_images_from_context()` is called every step; internal step gating via `image_every_n_steps`.
- Only main process (`rank == 0`) creates loggers and writes image/checkpoint files.

### DDP / FSDP2 Wrapping
- `ComponentManager.apply_parallel()` iterates entries and applies the resolved strategy per component. Strategy resolution is config-driven:
  - `parallel.strategy` per component (explicit).
  - `fsdp.enabled` per component (for FSDP2 opt-in).
  - Global defaults: `default_non_fsdp_trainable` (defaults to `ddp`), non-trainable → `replicated`.
- FSDP2 wrapping details:
  - Wrap targets from either `fsdp.wrap_modules` (glob patterns over `named_modules()`) or `get_fsdp_wrap_module_list()` method on the module.
  - `fully_shard_module()` calls `torch.distributed.fsdp.fully_shard` with configurable mixed precision, CPU offload, reshard-after-forward.
  - Custom forward methods can be registered via `register_fsdp2_forward_methods()`.
- `unwrap_model()` strips DDP wrapper to access underlying module.

### External Consumer Contracts
- Business modules (`models.*`, `network.*`, `data.*`) are external to this framework. They are consumed via `target + params` in YAML configs and are not present in the repo.
- A component module must:
  - Be importable from its `target` path.
  - Accept only `params` kwargs in `__init__`.
  - Optionally implement `state_dict()` / `load_state_dict()` for checkpointing.
  - Optionally implement `gradient_checkpointing_enable()` or `enable_gradient_checkpointing()` for gradient checkpointing.
  - Optionally implement `get_fsdp_wrap_module_list()` for FSDP2 wrapping hints.
- A dataset module must be constructable from `params` and return batches whose structure matches what ops and losses expect in `ctx.batch.*`.
- A loss function must accept kwargs mapped by `LossWrapper.inputs` and return either a Tensor or `{"loss": Tensor, ...}`.

### Module Dependency Graph
```
train.py
  +-- config.py (load_config)
  +-- engine.py (Trainer)
  |     +-- context.py (TrainContext)
  |     +-- instantiate.py (instantiate, locate)
  |     +-- components.py (ComponentManager, ComponentEntry)
  |     |     +-- instantiate.py
  |     |     +-- distributed.py (wrap_ddp, fully_shard_module, ...)
  |     +-- optim.py (build_optimizers, build_schedulers, collect_params)
  |     |     +-- instantiate.py
  |     +-- losses.py (LossWrapper, build_losses)
  |     |     +-- instantiate.py
  |     |     +-- resolver.py (resolve_input)
  |     +-- phase_runner.py (PhaseRunner)
  |     |     +-- registry.py (get_op)
  |     |     +-- framework/ops/* (via `from .ops import *`)
  |     +-- distributed.py (init_distributed, ...)
  |     +-- utils.py (StepTimer)
  |     +-- loggers.py (LoggerCollection, build_loggers)
  +-- distributed.py (cleanup_distributed)
```
