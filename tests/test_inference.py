from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from data.inference_image_dataset import InferenceImageDataset, inference_collate_fn
from framework.context import TrainContext
from scripts.infer_tadisr import (
    apply_checkpoint_to_config,
    infer_target_size,
    log_wandb_batch,
    make_inference_phase,
    parse_wandb_images,
)


class InferenceImageDatasetTest(unittest.TestCase):
    def test_reads_file_resizes_and_collates_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            rgb = np.zeros((5, 7, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.assertTrue(cv2.imwrite(str(path), bgr))

            dataset = InferenceImageDataset(
                input_path=str(path),
                size="4x6",
                prompt="hello",
                return_meta=True,
            )
            sample = dataset[0]

            self.assertEqual(sample["lr"].shape, (3, 4, 6))
            self.assertEqual(sample["hr"].shape, (3, 4, 6))
            self.assertEqual(sample["mask"].shape, (1, 4, 6))
            self.assertEqual(sample["prompt"], "hello")
            self.assertEqual(sample["filename"], "sample.png")
            self.assertEqual(sample["orig_size"].tolist(), [5, 7])

            batch = inference_collate_fn([sample])
            self.assertEqual(batch["lr"].shape, (1, 3, 4, 6))
            self.assertEqual(batch["path"], [str(path)])


class InferenceScriptHelpersTest(unittest.TestCase):
    def test_checkpoint_dir_sets_available_component_paths(self):
        cfg = OmegaConf.create(
            {
                "components": {
                    "dit": {"target": "x.DiT", "params": {}},
                    "jsd": {"target": "x.JSD", "params": {}},
                    "vae_encoder": {"target": "x.VAE", "params": {}},
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "checkpoint-last" / "models"
            models_dir.mkdir(parents=True)
            (models_dir / "dit.pt").write_bytes(b"dit")
            (models_dir / "jsd.pt").write_bytes(b"jsd")

            loaded = apply_checkpoint_to_config(cfg, str(models_dir.parent))

            self.assertEqual(loaded, ["dit", "jsd"])
            self.assertEqual(cfg.components.dit.checkpoint, str(models_dir / "dit.pt"))
            self.assertEqual(cfg.components.jsd.checkpoint, str(models_dir / "jsd.pt"))
            self.assertFalse("checkpoint" in cfg.components.vae_encoder)

    def test_inference_phase_skips_training_only_ops_and_losses(self):
        phase = OmegaConf.create(
            {
                "name": "generator",
                "ops": {
                    "vae_encode": {"component": "vae_encoder"},
                    "rescale_hr": {"type": "rescale", "input": "batch.hr"},
                    "make_taca_heatmap": {
                        "type": "make_heatmap",
                        "input": "pred.a_tex",
                        "target_h": 1024,
                        "target_w": 1024,
                    },
                    "save": {"type": "save_image"},
                },
                "losses": ["sr", "seg"],
            }
        )

        infer_phase = make_inference_phase(phase, ["vae_encoder", "dit"], (256, 384))

        self.assertEqual(infer_phase["losses"], [])
        self.assertFalse(infer_phase["backward"])
        self.assertEqual([op["name"] for op in infer_phase["ops"]], ["vae_encode", "make_taca_heatmap"])
        self.assertEqual(infer_phase["ops"][1]["target_h"], 256)
        self.assertEqual(infer_phase["ops"][1]["target_w"], 384)
        self.assertEqual(infer_phase["modes"], {"vae_encoder": "eval", "dit": "eval"})

    def test_inference_phase_can_skip_heatmap_when_not_requested(self):
        phase = OmegaConf.create(
            {
                "name": "generator",
                "ops": {
                    "vae_encode": {"component": "vae_encoder"},
                    "make_taca_heatmap": {"type": "make_heatmap"},
                },
            }
        )

        infer_phase = make_inference_phase(
            phase,
            ["vae_encoder"],
            (256, 256),
            include_heatmap=False,
        )

        self.assertEqual([op["name"] for op in infer_phase["ops"]], ["vae_encode"])

    def test_infer_target_size_prefers_cli_then_config(self):
        cfg = OmegaConf.create(
            {
                "data": {
                    "train": {
                        "dataset": {
                            "params": {
                                "target_size": "1024x768",
                            }
                        }
                    }
                },
                "components": {"jsd": {"params": {"resolution": 512}}},
            }
        )

        self.assertEqual(infer_target_size(cfg, "128x256"), (128, 256))
        self.assertEqual(infer_target_size(cfg, None), (1024, 768))

    def test_parse_wandb_images_validates_values(self):
        self.assertEqual(parse_wandb_images("lr,pred,seg,heatmap"), ["lr", "pred", "seg", "heatmap"])
        self.assertEqual(parse_wandb_images("all"), ["lr", "pred", "seg", "heatmap"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_wandb_images("lr,bad")

    def test_log_wandb_batch_uploads_requested_images(self):
        class FakeImage:
            def __init__(self, data, caption=None):
                self.data = data
                self.caption = caption

        class FakeWandb:
            Image = FakeImage

            def __init__(self):
                self.logs = []

            def log(self, payload, step=None):
                self.logs.append((payload, step))

        batch = {
            "lr": torch.zeros(1, 3, 4, 6),
            "stem": ["sample"],
            "path": ["/tmp/sample.png"],
        }
        ctx = TrainContext(batch=batch)
        ctx.set("pred.rgb", torch.zeros(1, 3, 4, 6))
        ctx.set("pred.seg", torch.ones(1, 1, 4, 6))
        ctx.set("pred.taca_heatmap", torch.rand(1, 1, 4, 6))

        fake = FakeWandb()
        logged = log_wandb_batch(
            fake,
            ctx,
            batch,
            step=7,
            image_names=["lr", "pred", "seg", "heatmap"],
        )

        self.assertEqual(logged, 1)
        self.assertEqual(len(fake.logs), 1)
        payload, step = fake.logs[0]
        self.assertEqual(step, 7)
        self.assertEqual(
            sorted(k for k in payload if k.startswith("inference/")),
            [
                "inference/heatmap",
                "inference/lr",
                "inference/pred",
                "inference/sample_index",
                "inference/seg",
            ],
        )
        self.assertEqual(payload["inference/lr"].data.shape, (4, 6, 3))
        self.assertEqual(payload["inference/seg"].data.shape, (4, 6))
        self.assertEqual(payload["inference/heatmap"].data.shape, (4, 6, 3))
        self.assertIn("sample", payload["inference/pred"].caption)


if __name__ == "__main__":
    unittest.main()
