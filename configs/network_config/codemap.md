# configs/network_config/

## Responsibility

Reusable network/backbone configuration fragments imported by top-level experiment configs via the `imports` mechanism. Each file defines a single backbone network (e.g., Latent Diffusion Model denoising backbone) as a `target + params` block, which gets merged into the top-level `components.*` hierarchy by `framework/config.py` and then instantiated by `framework/components.py`.

## Design

File: **`ldm_nch_v3.yaml`** — LDM (Latent Diffusion Model) backbone in NCHW channel format, version 3.

```yaml
target: network.nch_ldm_v3.NCH_LDM_V3
params:
  input_type: noise_one_lq
  timesteps: [1.0]
  enable_skip_level: '60'
  model_cfg_path: configs/model_config/SR_nch_v3_ti2i_with_guidance_f16c64.yaml
  base_model_ckpt_path: new_x_embedder_merge.pth
```

- **`target`**: Full Python import path to the backbone class. The module `network.nch_ldm_v3` is external (removed from the repo). The class name `NCH_LDM_V3` signals NCHW-tensor LDM v3.
- **`params.input_type: noise_one_lq`**: The backbone accepts a noisy latent concatenated with a low-quality condition (one_lq = one low-quality conditioning image). This implies the network is used in an image restoration / super-resolution diffusion pipeline, not unconditional generation.
- **`params.timesteps: [1.0]`**: A fixed single timestep (no diffusion noise schedule applied externally). This suggests a deterministic / one-step refinement mode, or timestep conditioning is injected elsewhere.
- **`params.enable_skip_level: '60'`**: Enables a specific skip-connection level (60) in the UNet-like backbone. Controls which resolution level receives a direct skip from the encoder to the decoder.
- **`params.model_cfg_path`**: Delegates detailed architecture parameters (depth, width, attention heads, cross-attention, guidance embedding) to an external YAML. The referenced file `configs/model_config/SR_nch_v3_ti2i_with_guidance_f16c64.yaml` is the source of truth for `dim`, `num_res_blocks`, `attn_resolutions`, `ch_mult`, `num_heads`, etc.
- **`params.base_model_ckpt_path`**: Path to a pretrained backbone state dict (`new_x_embedder_merge.pth`). This is loaded at component-build time by `ComponentManager` via the component-level `checkpoint` mechanism (the path appears in `params` but not as a top-level `checkpoint` key — the component card lacks explicit `checkpoint`, `strict`, `train`, `save`, `gradient_checkpointing` settings, so defaults apply: strict load enabled, full train strategy, full save, no gradient checkpointing).

**Config composition role**: This file is a **fragment** — it is never a standalone top-level config. It is designed to be merged into a parent config via `imports` under a named key (e.g., `components.dit`). The merge copies the entire file content under that key, producing:

```yaml
components:
  dit:
    target: network.nch_ldm_v3.NCH_LDM_V3
    params: { ... }
```

No `extends` or `_base_` are used; this is a flat target+params definition.

## Data & Control Flow

1. **Import merge** — `framework/config.py` reads the top-level experiment config (e.g., `configs/f16c64_vae_dit_proj_out_align.yaml`). Its `imports.components.dit` directive triggers a recursive file load of `ldm_nch_v3.yaml`. The content is deep-merged into the top-level config under `components.dit`.

2. **Component instantiation** — `framework/components.py` iterates over `components`. For `dit`, it:
   - Imports `network.nch_ldm_v3.NCH_LDM_V3`.
   - Calls the constructor with `**params` (input_type, timesteps, enable_skip_level, model_cfg_path, base_model_ckpt_path). The constructor internally reads `model_cfg_path` to build the full architecture.
   - Attempts to load `base_model_ckpt_path` — since there is no top-level `checkpoint` key, this load is handled inside the constructor (if implemented), not by the framework's explicit checkpoint loader.

3. **DDP/FSDP2 wrapping** — After construction, `ComponentManager` wraps the backbone in DDP or FSDP2 per `train.strategy` (defaults to `full`). The model becomes a distributed-aware component accessible via `components["dit"]` in the `TrainContext`.

4. **In-pipeline usage** — During training, the backbone is called by ops (e.g., diffusion/denoising ops in `framework/ops/diffusion.py`). Typical data flow:
   - VAE encoder produces latent `z`.
   - Noise is added to `z` (or the latent is concatenated with low-quality embedding).
   - The noisy latent + conditioning is fed to the backbone as `input_type: noise_one_lq`.
   - Backbone predicts the denoised latent (or noise residual).
   - VAE decoder reconstructs the image.
   - Loss is computed between reconstruction and ground truth.

   The single fixed timestep `[1.0]` suggests the diffusion schedule is either degenerate (single-step refinement) or controlled externally via the op logic rather than by the backbone's internal timestep embedding.

## Integration Points

| Consumer | Mechanism |
|---|---|
| **Top-level experiment configs** (e.g., `configs/f16c64_vae_dit_proj_out_align.yaml`) | `imports.components.dit: configs/network_config/ldm_nch_v3.yaml` — merge into `components.dit` |
| **`framework/config.py`** | Recursive import resolution; deep-merges fragment into parent config |
| **`framework/components.py`** (`ComponentManager.build_all()`) | `target + params` instantiation; optional checkpoint load; DDP/FSDP2 wrap |
| **Training ops** (`framework/ops/`) | Access backbone as `components["dit"]`; call forward with latent + conditioning tensors |
| **External (likely missing) targets** | `network.nch_ldm_v3.NCH_LDM_V3` — Python module removed from repo. Architecture details delegated to `configs/model_config/SR_nch_v3_ti2i_with_guidance_f16c64.yaml` (also likely missing). Checkpoint `new_x_embedder_merge.pth` not present in repo. |

To restore this backbone, one must implement `network.nch_ldm_v3.NCH_LDM_V3` (or provide a compatible replacement) and the referenced `model_cfg_path` YAML, plus place or symlink the checkpoint.
