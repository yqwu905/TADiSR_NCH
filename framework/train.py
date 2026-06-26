import logging
import torch
IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
if IS_NPU:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
import argparse
from omegaconf import OmegaConf
from .engine import Trainer
from .distributed import cleanup_distributed
from framework.config import load_config


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        override_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    try:
        trainer = Trainer(cfg)
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
