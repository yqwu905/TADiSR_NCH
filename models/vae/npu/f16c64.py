import torch
import torch.nn as nn
from .mj64_vae import Encoder, Decoder

SHIFTING_FACTOR = -0.02453034371137619
SCALING_FACTOR = 1 / 0.7610968947410583

class VaeEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Encoder(**kwargs)

    def forward(self, x):
        out = self.model(x)
        mean, _ = torch.chunk(out, chunks=2, dim=1)
        out_dict = {
            "latent": (mean - SHIFTING_FACTOR) * SCALING_FACTOR,
        }
        return out_dict

    def get_fsdp_wrap_module_list(self):
        encoder = self.model
        modules = [encoder]
        for level in encoder.down:
            modules.extend(level.block)   # ResnetBlock 列表
            modules.extend(level.attn)    # AttnBlock 列表，为空时无副作用
        modules.extend([encoder.mid.block_1, encoder.mid.attn_1, encoder.mid.block_2])
        return modules


class VaeDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Decoder(**kwargs)

    def forward(self, x):
        out = self.model(x)
        out_dict = {
            "recon": 1 / SCALING_FACTOR * out + SHIFTING_FACTOR,
        }
        return out_dict

    def get_fsdp_wrap_module_list(self):
        decoder = self.model
        modules = [decoder]
        for level in decoder.up:
            modules.extend(level.block)
            modules.extend(level.attn)
        modules.extend([decoder.mid.block_1, decoder.mid.attn_1, decoder.mid.block_2])
        return modules
