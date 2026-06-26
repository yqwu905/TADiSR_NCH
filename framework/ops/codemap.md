# framework/ops/

## Responsibility

Built-in op registration point and data-flow primitives for the training pipeline. Ops are composable, config-driven units that read from `TrainContext`, invoke a component or function, and write results back to `TrainContext`. The `__init__.py` re-exports all ops via wildcard imports from `common.py` and `diffusion.py`.

## Design

**Registry pattern with decorator.** Ops are registered globally via `@register_op("name")` (defined in `framework.registry`). The `OPS` dict maps string names to classes. `phase_runner.py` instantiates ops lazily: `op = get_op(op_type)(op_cfg)`, then calls `op(ctx, components)`.

**Op protocol.** Every op implements:
```python
def __init__(self, cfg):      # receives the op sub-dict from YAML config
def __call__(self, ctx, components):  # reads context, accesses components, writes outputs
```

**Input resolution.** Ops use `resolve_kwargs(inputs_cfg, ctx)` from `framework.resolver` to transform config-level input specs into concrete values. `resolve_input` supports:
- `str` -- direct `ctx.get(key)` via dotted path (e.g. `"batch.gt"`)
- `{"type": "ctx", "key": "..."}` -- explicit context lookup
- `{"type": "const", "value": ...}` -- literal constant
- `{"type": "ones_like", "ref": "..."}` / `zeros_like` / `full_like` / `randn_like` -- shape-inheriting tensor creation
- `{"type": "detach", "ref": "..."}` -- detached tensor
- `{"type": "cast", "value": ..., "dtype": "...", "device_ref": "..."}` -- type/device cast
- `{"type": "getattr", "object": "...", "name": "..."}` -- attribute access

**Output writing.** Ops write to context via `ctx.set(key, value)`. The helper `_write_outputs(ctx, result, outputs_cfg)` provides flexible mapping:
- `outputs: {"_": "ctx.key"}` -- single flat output
- `outputs: {"local_key": "ctx.key"}` -- dict result field mapping
- `outputs: {"0": "ctx.key"}` -- tuple/list index mapping

### Concrete ops in `common.py`

| Registered name  | Class          | What it does |
|------------------|----------------|--------------|
| `"call"`         | `CallOp`       | Resolves a component (by name + optional method) or a dotted-path function via `locate()`, builds kwargs from `inputs` spec, optionally detaches named inputs, optionally wraps with `torch.no_grad()`, calls the function, writes outputs. |
| `"make_tensor"`  | `MakeTensorOp` | Creates a tensor matching a reference tensor's shape/dev/dtype using `ones_like`, `zeros_like`, `full_like`, or `randn_like`. Writes result to `output` key. |
| `"set_value"`    | `SetValueOp`   | Resolves a single arbitrary value via `resolve_input` (supporting all input spec types) and writes it to the configured `output` key. |
| `"detach"`       | `DetachOp`     | Reads a tensor from `input`, calls `.detach()`, writes to `output`. |
| `"save_image"`   | `SaveImageOp`  | Reads an image tensor from `inputs.image`, normalizes to `[0,1]`, saves per-sample PNGs to `output_dir`. Supports filename resolution from `ctx` (explicit filenames, path-derived basenames, or step-indexed defaults). **BROKEN** -- references undefined `save_image` (torchvision save function), `_ensure_list`, and `_basename_without_ext`. |

### Concrete ops in `diffusion.py`

| Registered name            | Class                  | What it does |
|----------------------------|------------------------|--------------|
| `"sample_timestep"`        | `SampleTimestepOp`     | Samples timesteps `t in [eps, 1-eps]` for a batch from `uniform` or `logit_normal` distribution. Writes to `noise.t` (default output key). References a `ref` tensor (or falls back to `batch.gt`) for batch size and device. |
| `"flow_matching_prepare"`  | `FlowMatchingPrepareOp`| Computes linear interpolation `xt = (1-t)*x0 + t*x1` and target velocity `v = x1 - x0`. Reads `x1`, `x0` (optional, defaults to `randn_like`), and `t` from context. Writes `xt` to `latent.noisy` and `v` to `target.v` (default keys). |
| `"dmd_proxy_target"`       | `DMDProxyTargetOp`     | Implements the DMD target trick: `target = (pred - grad * scale).detach()`. Reads `pred` and `grad` from context, writes result to `dmd.target` (default key). |
| `"nch_ldm_v3_two_step"`    | `NCHLDMV3TwoStepOp`    | Task-specific multi-step NCH LDM inference loop. Reads `hidden_states` and `encoder_hidden_states` from context. Runs multiple Euler steps through a DiT component, building input by concatenating `[x_start, mask, mask_image_latents]` along `concat_dim`. Supports `noise_*` vs `lq_*` and `*_zero_lq` vs `*_one_lq` input types (mask branch). Writes final latent to `pred.out` (default key) or via `_write_outputs`. |

## Data & Control Flow

1. `phase_runner.py` iterates over configured ops in a phase.
2. For each op entry: `op = get_op(op_type)(op_cfg)` instantiates the class with its config dict.
3. `op(ctx, components)` is called.
4. Inside `__call__`:
   - `resolve_kwargs(self.cfg["inputs"], ctx)` resolves each named input spec into a concrete value from `TrainContext`.
   - Data flows: context values -> op-local variables -> component/function call.
   - For `CallOp` specifically: `components[component_name]` retrieves the managed module; the call passes `**kwargs`.
   - Results are written via `ctx.set(key, value)` or `_write_outputs(ctx, result, outputs)`.
5. Context keys are dotted paths. Examples from specific ops:
   - `batch.gt` (input ref for `sample_timestep`, fallback batch data)
   - `noise.t` (output of `sample_timestep`)
   - `latent.noisy` (output xt from `flow_matching_prepare`)
   - `target.v` (target velocity from `flow_matching_prepare`)
   - `dmd.target` (output of `dmd_proxy_target`)
   - `pred.out` (output of `nch_ldm_v3_two_step`)
   - `pred.rgb` / `pred.v` (general prediction keys used by `call` op outputs)

## Integration Points

- **`framework.registry`**: `register_op(name)` populates `OPS` dict; consumers do `get_op(name)`.
- **`framework.resolver`**: `resolve_kwargs` and `resolve_input` are the glue between config input specs and `TrainContext` values.
- **`framework.context` (`TrainContext`)**: All ops read from and write to this shared state object using dotted keys.
- **`framework.instantiate`**: `locate()` is used by `CallOp` to resolve dotted-path function strings when no component is specified.
- **`framework.phase_runner`**: The sole consumer -- it iterates `phase.ops`, instantiates each op, and calls it with `(ctx, components)`.
- **Config files**: Each op in a phase is configured as `{name: <registered_name>, inputs: {...}, outputs: {...}, ...}`. The registered name determines which class is instantiated.
- **Known broken op**: `SaveImageOp` (`"save_image"`) in `common.py` line 156 calls `save_image(img, save_path)` which references an undefined `save_image` (intended as `torchvision.utils.save_image`). It also calls `_ensure_list` (line 142, 145) and `_basename_without_ext` (line 146), neither of which are defined or imported in that file. Any config using `"save_image"` will raise `NameError` at runtime.
