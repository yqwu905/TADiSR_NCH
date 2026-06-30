"""Run TADiSR inference from a training config and checkpoint."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data.inference_image_dataset import InferenceImageDataset, inference_collate_fn
from data.tadisr_dataset import DEFAULT_PROMPT, _parse_target_size
from framework.components import ComponentManager
from framework.config import load_config
from framework.context import TrainContext
from framework.engine import move_to_device
from framework.phase_runner import PhaseRunner


logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def select_device(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg or "auto")
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        return torch.device("cpu")
    return torch.device(device_cfg)


def _load_config(config_path: Optional[str], checkpoint: Optional[str]):
    if config_path:
        return load_config(config_path)

    if not checkpoint:
        raise ValueError("--config is required when --checkpoint is not provided")

    ckpt_path = Path(checkpoint).expanduser()
    candidates = [ckpt_path / "config.yaml"]
    if ckpt_path.name == "models":
        candidates.append(ckpt_path.parent / "config.yaml")

    ckpt_cfg = next((path for path in candidates if path.exists()), None)
    if ckpt_cfg is None:
        raise ValueError(
            "--config is required because checkpoint config was not found at "
            f"{candidates[0]}"
        )
    return OmegaConf.load(ckpt_cfg)


def _checkpoint_models_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    models_dir = path / "models"
    if models_dir.is_dir():
        return models_dir
    if path.is_dir() and path.name == "models":
        return path
    raise FileNotFoundError(
        f"checkpoint must be a checkpoint directory containing models/ or a models/ "
        f"directory, got {path}"
    )


def apply_checkpoint_to_config(cfg, checkpoint: Optional[str]) -> list[str]:
    if not checkpoint:
        return []

    models_dir = _checkpoint_models_dir(checkpoint)
    loaded = []
    for name, comp_cfg in cfg.get("components", {}).items():
        model_path = models_dir / f"{name}.pt"
        if model_path.exists():
            comp_cfg["checkpoint"] = str(model_path)
            loaded.append(name)
    if not loaded:
        raise FileNotFoundError(f"no component .pt files found in {models_dir}")
    return loaded


def apply_component_checkpoints(cfg, specs: list[str]) -> None:
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "--component-checkpoint must use COMPONENT=/path/to/model.pt"
            )
        name, path = spec.split("=", 1)
        if name not in cfg.components:
            raise KeyError(f"component {name!r} not found in config")
        cfg.components[name]["checkpoint"] = path


def freeze_components_for_inference(cfg) -> None:
    for comp_cfg in cfg.get("components", {}).values():
        comp_cfg["train"] = {"strategy": "frozen"}
        comp_cfg["mode"] = "eval"


def _iter_op_configs(ops_cfg) -> list[dict]:
    ops_cfg = _plain(ops_cfg) or []
    if isinstance(ops_cfg, Mapping):
        out = []
        for name, op_cfg in ops_cfg.items():
            op = dict(_plain(op_cfg) or {})
            op.setdefault("name", name)
            op.setdefault("type", "call")
            out.append(op)
        return out

    out = []
    for op_cfg in ops_cfg:
        op = dict(_plain(op_cfg) or {})
        op.setdefault("type", "call")
        op.setdefault("name", op["type"])
        out.append(op)
    return out


def make_inference_phase(
    train_phase,
    component_names: list[str],
    target_size: tuple[int, int],
    *,
    include_heatmap: bool = True,
):
    phase = dict(_plain(train_phase))
    ops = []
    skip_names = {"rescale_hr"}
    skip_types = {"rescale", "save_image"}

    for op in _iter_op_configs(phase.get("ops", [])):
        op_name = str(op.get("name", op.get("type", "")))
        op_type = str(op.get("type", "call"))
        if op_name in skip_names or op_type in skip_types:
            continue
        if op_type == "make_heatmap" and not include_heatmap:
            continue
        if op_type == "make_heatmap" and target_size is not None:
            op["target_h"], op["target_w"] = int(target_size[0]), int(target_size[1])
        ops.append(op)

    modes = {name: "eval" for name in component_names}
    return {
        "name": f"{phase.get('name', 'phase')}_inference",
        "trainable": [],
        "frozen": list(component_names),
        "modes": modes,
        "zero_grad": [],
        "backward": False,
        "step": [],
        "ops": ops,
        "losses": [],
    }


def _find_first_phase(cfg):
    phases = cfg.train_program.phases
    if not phases:
        raise ValueError("config has no train_program.phases")
    return phases[0]


def infer_target_size(cfg, explicit_size: Optional[str]) -> tuple[int, int]:
    if explicit_size:
        return _parse_target_size(explicit_size)

    size = OmegaConf.select(cfg, "data.train.dataset.params.target_size")
    if size is not None:
        return _parse_target_size(size)

    resolution = OmegaConf.select(cfg, "components.jsd.params.resolution")
    if resolution is not None:
        return int(resolution), int(resolution)

    resolution = OmegaConf.select(cfg, "components.vae_encoder.params.resolution")
    if resolution is not None:
        return int(resolution), int(resolution)

    return 512, 512


def _set_embedding_cache(cfg, cache_path: Optional[str]) -> None:
    if not cache_path:
        return
    if "offline_embedding" not in cfg.components:
        raise KeyError("config has no components.offline_embedding")
    cfg.components.offline_embedding.params.embedding_cache_path = cache_path


def _set_dit_runtime_options(
    phase: dict,
    *,
    timestep: Optional[float],
    enable_skip_level: Optional[str],
    denoise: Optional[bool],
) -> None:
    for op in phase["ops"]:
        if op.get("type") != "nch_mmdit_sr":
            continue
        if timestep is not None:
            op["timestep"] = float(timestep)
        if enable_skip_level is not None:
            op["enable_skip_level"] = str(enable_skip_level)
        if denoise is not None:
            op["denoise"] = bool(denoise)


def _to_uint8(tensor: torch.Tensor, value_range: str) -> Any:
    image = tensor.detach().float().cpu()
    if value_range == "-1_1":
        image = (image + 1.0) * 0.5
    elif value_range == "0_255":
        image = image / 255.0
    elif value_range != "0_1":
        raise ValueError(f"unsupported value_range: {value_range}")

    image = image.clamp(0.0, 1.0)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.permute(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return (image.numpy() * 255.0 + 0.5).astype("uint8")


def save_tensor_image(
    tensor: torch.Tensor,
    path: Path,
    *,
    value_range: str,
    restore_size: Optional[tuple[int, int]] = None,
    colorize: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _to_uint8(tensor, value_range=value_range)
    if restore_size is not None:
        h, w = restore_size
        image = cv2.resize(image, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
    if colorize:
        if image.ndim == 3:
            image = image[..., 0]
        image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    elif image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _batch_list(batch: Mapping[str, Any], key: str, default_prefix: str, n: int) -> list[str]:
    values = batch.get(key)
    if values is None:
        return [f"{default_prefix}_{idx:04d}" for idx in range(n)]
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values]


def save_outputs(
    ctx: TrainContext,
    batch: Mapping[str, Any],
    output_dir: Path,
    *,
    save_lr: bool,
    save_seg: bool,
    save_heatmap: bool,
    restore_input_size: bool,
) -> list[dict]:
    pred = ctx.get("pred.rgb")
    stems = _batch_list(batch, "stem", "sample", int(pred.shape[0]))
    paths = _batch_list(batch, "path", "", int(pred.shape[0]))
    orig_sizes = batch.get("orig_size")

    records = []
    for idx, stem in enumerate(stems):
        restore_size = None
        if restore_input_size and torch.is_tensor(orig_sizes):
            size_tensor = orig_sizes[idx].detach().cpu()
            restore_size = (int(size_tensor[0]), int(size_tensor[1]))

        record = {"input": paths[idx] if idx < len(paths) else "", "stem": stem}

        sr_path = output_dir / "sr" / f"{stem}.png"
        save_tensor_image(pred[idx], sr_path, value_range="-1_1", restore_size=restore_size)
        record["sr"] = str(sr_path)

        if save_lr and ctx.has("batch.lr"):
            lr_path = output_dir / "lr" / f"{stem}.png"
            save_tensor_image(
                ctx.get("batch.lr")[idx],
                lr_path,
                value_range="0_1",
                restore_size=restore_size,
            )
            record["lr"] = str(lr_path)

        if save_seg and ctx.has("pred.seg"):
            seg_path = output_dir / "seg" / f"{stem}.png"
            save_tensor_image(
                ctx.get("pred.seg")[idx],
                seg_path,
                value_range="0_1",
                restore_size=restore_size,
            )
            record["seg"] = str(seg_path)

        if save_heatmap and ctx.has("pred.taca_heatmap"):
            heatmap_path = output_dir / "heatmap" / f"{stem}.png"
            save_tensor_image(
                ctx.get("pred.taca_heatmap")[idx],
                heatmap_path,
                value_range="0_1",
                restore_size=restore_size,
                colorize=True,
            )
            record["heatmap"] = str(heatmap_path)

        records.append(record)
    return records


def run(args) -> None:
    cfg = _load_config(args.config, args.checkpoint)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    _set_embedding_cache(cfg, args.embedding_cache)
    loaded_components = apply_checkpoint_to_config(cfg, args.checkpoint)
    apply_component_checkpoints(cfg, args.component_checkpoint)
    freeze_components_for_inference(cfg)

    target_size = infer_target_size(cfg, args.target_size)
    device = select_device(args.device)
    mixed_precision = args.mixed_precision
    if mixed_precision is None:
        mixed_precision = str(OmegaConf.select(cfg, "runtime.mixed_precision") or "no")
        if device.type == "cpu":
            mixed_precision = "no"

    component_names = list(cfg.get("components", {}).keys())
    phase = make_inference_phase(
        _find_first_phase(cfg),
        component_names,
        target_size,
        include_heatmap=args.save_heatmap,
    )
    _set_dit_runtime_options(
        phase,
        timestep=args.timestep,
        enable_skip_level=args.enable_skip_level,
        denoise=args.denoise,
    )

    logger.info("[inference] device=%s mixed_precision=%s target_size=%s", device, mixed_precision, target_size)
    if loaded_components:
        logger.info("[inference] checkpoint components: %s", ", ".join(loaded_components))

    components = ComponentManager(cfg.get("components", {})).build_all()
    components.apply_gradient_checkpointing()
    components.to(device)
    components.set_initial_modes()
    for _, module in components.items():
        if hasattr(module, "eval"):
            module.eval()

    dataset = InferenceImageDataset(
        input_path=args.input,
        size=target_size,
        prompt=args.prompt,
        recursive=args.recursive,
        interpolation=args.interpolation,
        return_meta=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        collate_fn=inference_collate_fn,
    )

    runner = PhaseRunner(
        components,
        optimizers={},
        schedulers={},
        losses={},
        device=device,
        mixed_precision=mixed_precision,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    total = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for batch in loader:
            if args.max_images is not None and total >= args.max_images:
                break
            remaining = None if args.max_images is None else args.max_images - total
            if remaining is not None and batch["lr"].shape[0] > remaining:
                keep = int(remaining)
                batch = {
                    key: value[:keep] if torch.is_tensor(value) else value[:keep]
                    for key, value in batch.items()
                }

            batch = move_to_device(batch, device)
            ctx = TrainContext(global_step=total, batch=batch)
            with torch.no_grad():
                runner.run(ctx, phase, do_zero_grad=False, do_step=False)

            records = save_outputs(
                ctx,
                batch,
                output_dir,
                save_lr=args.save_lr,
                save_seg=args.save_seg,
                save_heatmap=args.save_heatmap,
                restore_input_size=args.restore_input_size,
            )
            for record in records:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += len(records)
            logger.info("[inference] saved %s/%s", total, len(dataset))

    logger.info("[inference] wrote manifest: %s", manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="training config; optional when checkpoint/config.yaml exists")
    parser.add_argument("--checkpoint", help="checkpoint-* directory or its models/ subdirectory")
    parser.add_argument("--input", required=True, help="input image file or directory")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--target-size", help="inference resize as HxW, e.g. 1024x1024")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--embedding-cache", help="override components.offline_embedding cache path")
    parser.add_argument(
        "--component-checkpoint",
        action="append",
        default=[],
        help="override a component checkpoint as COMPONENT=/path/to/model.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", choices=["no", "fp32", "bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--interpolation", default="bicubic")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--save-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-seg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-heatmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-input-size", action="store_true")
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--enable-skip-level")
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
