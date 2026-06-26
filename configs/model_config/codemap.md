# configs/model_config/

## Responsibility

Reusable model component configuration fragments (DiT/VAE/embedding definitions) imported by top-level experiment configs via `imports`. Each file defines a single model component using the `target` + `params` schema consumed by `framework/components.py`. Files are referenced from the top-level YAML's `imports` section under paths like `components.dit`, `components.vae`, etc.

## Design

Single YAML file as of writing: `SR_nch_v3_ti2i_with_guidance_f16c64.yaml`.

### SR_nch_v3_ti2i_with_guidance_f16c64.yaml

**Target**: `models.dit.nch.ldm.transformer_nch_v3_split.NCHTransformer2DModel`

A 37-layer sparse-attention DiT (NCH variant v3, split architecture) for text-image-to-image (ti2i) super-resolution tasks with classifier-free guidance (CFG) support.

**Key architecture params**:

| Param | Value | Semantics |
|---|---|---|
| `patch_size` | 1 | 1x1 patch embedding (pixel-level tokenization) |
| `in_channels` | 1024 | Input latent channels (likely concatenated noisy latents + conditioning, e.g. 256*4 or 64*16) |
| `out_channels` | 256 | Output latent channels (denoised prediction) |
| `num_layers` | 37 | Transformer blocks |
| `attention_head_dim` | 384 | Per-head dimension |
| `num_attention_heads` | 4 | Attention heads |
| `joint_attention_dim` | 1536 | Cross-attention key/value dimension (text encoder output dim) |
| `pooled_projection_dim` | 768 | Pooled text embedding dimension for adaLN modulation |
| `guidance_embeds` | false | When false = CFG mode (no guidance embedding), when true = GFT (guided fine-tuning) |
| `axes_dims_rope` | [48, 168, 168] | Rotary position embedding dimension per axis (3 axes: depth, height, width for 3D latent) |
| `processor_type` | sparse | Attention processor variant (sparse attention for efficiency) |
| `ffn_ratio` | 4 | Feed-forward expansion ratio (4x hidden_dim) |
| `adaln_dim` | 1536 | Adaptive layer norm modulation dim (matches `joint_attention_dim`) |
| `layers_to_retained` | dict | Layer pruning maps at inference budgets {100%, 60%, 40%, 30%}, each listing which transformer blocks to keep |

**Filename convention**: `SR` = super-resolution, `nch_v3` = NCH architecture version 3, `ti2i` = text-image-to-image, `guidance` = CFG support, `f16c64` = feature down factor 16 / base channels 64 (latent space geometry).

### Subdirectory: text_encoder/

See `configs/model_config/text_encoder/codemap.md` for text encoder component configs (nch_trainable_vector.yaml, offline_embedding.yaml). The text_encoder/ directory is an independent sub-component collection; its files define text encoder fragments (e.g. offline precomputed embeddings, trainable vector encoders) that complement the DiT model by providing `joint_attention_dim=1536` aligned outputs.

## Data & Control Flow

1. **Import**: Top-level experiment config (e.g. `configs/f16c64_vae_dit_proj_out_align.yaml`) lists under `imports`:
   ```yaml
   imports:
     components.dit: configs/model_config/SR_nch_v3_ti2i_with_guidance_f16c64.yaml
   ```
2. **Merge**: `framework/config.py` resolves `imports` by reading the fragment and merging its content under the specified dotted path (`components.dit`).
3. **Instantiation**: `framework/components.py` reads `components.dit.target` + `components.dit.params`, imports the class via `importlib`, and calls `NCHTransformer2DModel(**params)`.
4. **Checkpoint** (optional): If the importing config adds a `checkpoint` path under `components.dit`, `ComponentManager` loads weights after construction.
5. **Forward**: During training, the op layer invokes the model via an op (e.g. diffusion forward) reading from `TrainContext`; the model receives noisy latents, timesteps, text encoder hidden states (`joint_attention_dim=1536`), and pooled embeddings (`pooled_projection_dim=768`). It returns predicted denoised output (or noise prediction) of shape `[B, 256, ...]`.

The `layers_to_retained` dict enables inference-time layer pruning at different compute budgets without re-initialization.

## Integration Points

- **Consumed by**: Top-level configs via `imports` section; `framework/components.py` (target+params instantiation, checkpoint loading, parameter statistics).
- **External (missing) target**: `models.dit.nch.ldm.transformer_nch_v3_split.NCHTransformer2DModel` -- this module resides in the removed `models/` package; instantiation will fail until the model code is restored.
- **Text encoder**: See `configs/model_config/text_encoder/codemap.md` for the text encoder configs that produce the `joint_attention_dim=1536` / `pooled_projection_dim=768` embeddings consumed by this DiT.
- **Downstream**: The 256-channel output is consumed by a VAE decoder (defined in a separate config fragment) to produce the final image.
