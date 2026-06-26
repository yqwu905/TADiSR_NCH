# configs/data_config/

## Responsibility

Reusable dataset and dataloader configuration fragments imported by top-level experiment configs via the config system's `imports` mechanism. Each file defines a self-contained `{dataset: ..., dataloader: ...}` tree that gets merged into a dotted config path (e.g. `data.train` or `data.valid`), decoupling data-specific settings from model/optimizer/training config.

## Design

The sole file `infer_simple_dataset.yaml` defines an inference-mode dataset fragment:

```yaml
dataset:
  target: data.inference_image_dataset.InferenceImageDataset
  params:
    input_path: /path/to/images
    size: [2048, 1536]
    recursive: true
    interpolation: bicubic
    square_policy: landscape
    return_meta: true

dataloader:
  batch_size: 1
  shuffle: false
  num_workers: 4
  pin_memory: false
  persistent_workers: false
  prefetch_factor: 1
  timeout: 60
  multiprocessing_context: spawn
```

Key characteristics:

- **dataset.target**: Full Python import path to a dataset class (removed from repo — currently unresolved).
- **dataset.params**: Keyword arguments forwarded to the dataset constructor. `size` is a `[H, W]` list; `square_policy` and `return_meta` indicate per-sample metadata (e.g. original path, dimensions) is preserved.
- **dataloader.***: Passed directly to `torch.utils.data.DataLoader`. Inference-oriented: `batch_size=1`, `shuffle=false`. Explicitly sets all fields recommended by project conventions (`batch_size`, `shuffle`, `num_workers`, `drop_last` defaults to `false` via absence).
- Configuration composition: Imported via `imports: { data.train.dataset: configs/data_config/infer_simple_dataset.yaml }` in a top-level config. The config loader (`framework/config.py`) merges the YAML file's content under the specified dotted key (`data.train.dataset`). The resulting `data.train` subtree thus contains both `dataset` and `dataloader` keys.

## Data & Control Flow

1. Top-level YAML specifies `imports: { data.train.dataset: configs/data_config/infer_simple_dataset.yaml }`.
2. `framework/config.py` resolves the import, reading `infer_simple_dataset.yaml` and assigning its content to `cfg.data.train.dataset`.
3. `framework/engine.py` `Trainer.__init__` accesses `cfg.data.train` and calls `build_dataloader(cfg.data.train, dist_state=self.dist_state)` (line 169).
4. `build_dataloader` (line 79):
   - Calls `instantiate(data_cfg["dataset"])` — instantiates the `target` class with `params`.
   - Merges `dataloader` keys with defaults for `pin_memory`, `persistent_workers`, `prefetch_factor`, `timeout`, `multiprocessing_context` when `num_workers > 0`.
   - In DDP mode, pops `shuffle` to construct a `DistributedSampler`; sets `shuffle=False` and injects the sampler.
   - Constructs `torch.utils.data.DataLoader(dataset, **dl_cfg)`.
5. The dataset yields RGB image tensors and (when `return_meta=true`) metadata dicts. Exact shape/layout depends on `InferenceImageDataset` implementation (removed from repo).

## Integration Points

- **Consumed by**: Top-level configs via `imports`; `framework/engine.py` `build_dataloader` and `Trainer.__init__` (`cfg.data.train`, `cfg.data.valid`).
- **External dependency (missing)**: `data.inference_image_dataset.InferenceImageDataset` — the dataset class is no longer present in the repository. Re-implement or restore it before using this config fragment.
- **Extension pattern**: New data config files follow the same `{dataset: {target, params}, dataloader: {...}}` structure. Dataloader keys should always explicitly set `batch_size`, `shuffle`, `num_workers`, `drop_last` per project convention.
