# configs/model_config/text_encoder/

## Responsibility

Reusable text encoder component configuration sub-fragments. These files define text encoder components — the module that maps text prompts into conditioning embeddings (`encoder_hidden_states`) for the diffusion/transformer backbone. They are imported by top-level training configs or model configs via the `imports:` mechanism.

## Design

Two YAML files, one leaf and one that composes the leaf:

### `nch_trainable_vector.yaml` (leaf)

- **target**: `models.text_encoder.nch.utils_nch.TrainableVector_multitask`
- **params**: 17 boolean `*_fix_tokens` keys (one per task: `bbox`, `ctxt`, `seg`, `sr`, `t2i`, `sun`, `facesr`, `sr20Xbright`, `sr20Xdark`, `sr50X`, `bokeh`, `linedraw`, `moonsr`, `moonpainting`, `refsr`, `dehalo`, `firework`) all set to `True`, plus `token_requires_grad: True`.
- **Role**: Configures a multitask trainable token vector table. Each `*_fix_tokens` flag controls whether that task's token embeddings are frozen. `token_requires_grad` enables gradient flow into the token embeddings.
- **No checkpoint/strict/mode/save** — these are left to the parent config.
- **Consumed as**: an imported sub-fragment mapped into `params.trainable_vector_cfg` of `offline_embedding.yaml`.

### `offline_embedding.yaml` (composite)

- **imports**:
  - `params.trainable_vector_cfg` ← `nch_trainable_vector.yaml`
- **target**: `models.text_encoder.offline_embedding.EmbeddingDB`
- **params**:
  - `embedding_cache_path`: SQLite file path (`embedding_pangu_1b_vl_for_mainsr_v3_3b_ti2i_dehalo.sqlite`) — the persistent cache of pre-computed text embeddings.
  - `trainable_vector_cfg` (injected by import above): the `TrainableVector_multitask` config.
- **Role**: Defines an off-the-shelf embedding database component. At inference time, `EmbeddingDB.get_embedding()` looks up or computes text embeddings from a frozen text encoder (Pangu 1B VL), caches them in SQLite, and optionally combines them with trainable task-specific token vectors.
- **No checkpoint/strict/mode/save** — set by the importing top-level config.

## Data & Control Flow

1. **Config loading** (`framework/config.py`):
   - Top-level config (e.g., `configs/f16c64_vae_dit_proj_out_align.yaml`) declares:
     ```yaml
     imports:
       components.offline_embedding: configs/model_config/text_encoder/offline_embedding.yaml
     ```
   - `framework/config.py` resolves the import: `offline_embedding.yaml` is loaded and its `imports` are recursively resolved (`nch_trainable_vector.yaml` → `params.trainable_vector_cfg`).
   - The merged output lands at `config.components.offline_embedding` with `target` + `params` (including the nested `trainable_vector_cfg`).
   - The top-level config can override params, e.g.:
     ```yaml
     components:
       offline_embedding:
         params:
           task_tokens: dehalo
     ```

2. **Component instantiation** (`framework/components.py`):
   - `ComponentManager.build_all()` reads `config.components.offline_embedding`.
   - Calls `target(models.text_encoder.offline_embedding.EmbeddingDB)` with the merged `params` (trainable_vector_cfg + task_tokens + embedding_cache_path).
   - No checkpoint is loaded at this level; the EmbeddingDB itself manages the SQLite cache internally.

3. **Phase execution** (`framework/phase_runner.py`):
   - A phase op invokes method `get_embedding` on the `offline_embedding` component:
     ```yaml
     text_encode:
       component: offline_embedding
       method: get_embedding
       inputs:
         text: batch.prompt          # str prompt from dataset
         model_id:                   # const injected by resolver
           type: const
           value: pangu_1b_vl
       outputs:
         encoder_hidden_states: encoder_hidden_states
     ```
   - `resolve_kwargs` in `framework/resolver.py` resolves `batch.prompt` (from `ctx["batch.prompt"]`) and `model_id` (a constant).
   - `EmbeddingDB.get_embedding()` returns the text embedding tensor, written to `ctx["encoder_hidden_states"]`.
   - Downstream ops (e.g., `dit`) consume `encoder_hidden_states` as cross-attention conditioning.

## Integration Points

- **Consumed by**: Top-level configs (e.g., `configs/f16c64_vae_dit_proj_out_align.yaml`) via `imports: components.offline_embedding: ...`. Also directly importable by other model configs.
- **Framework layer**: `framework/config.py` (imports/extends resolution), `framework/components.py` (target+params instantiation), `framework/resolver.py` (input resolution for op calls).
- **External targets** (likely removed from repo, paths are non-existent):
  - `models.text_encoder.offline_embedding.EmbeddingDB`
  - `models.text_encoder.nch.utils_nch.TrainableVector_multitask`
- **Context keys produced**: `encoder_hidden_states` (Tensor, shape depends on text encoder output, typically `[B, L, D]` for cross-attention).
- **Context keys consumed**: `batch.prompt` (str) — text prompt from the dataset.
