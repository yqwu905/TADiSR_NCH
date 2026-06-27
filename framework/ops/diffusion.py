from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from framework.registry import register_op
from framework.resolver import resolve_input, resolve_kwargs
from framework.ops.common import _write_outputs


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


@register_op("sample_timestep")
class SampleTimestepOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        ref = ctx.get(self.cfg["ref"]) if "ref" in self.cfg else ctx.get("batch.gt")
        batch_size = ref.shape[0]
        device = ref.device
        distribution = self.cfg.get("distribution", "uniform")
        eps = float(self.cfg.get("eps", 1e-5))

        if distribution == "uniform":
            t = torch.rand(batch_size, device=device)
        elif distribution == "logit_normal":
            mean = float(self.cfg.get("mean", 0.0))
            std = float(self.cfg.get("std", 1.0))
            t = torch.randn(batch_size, device=device) * std + mean
            t = torch.sigmoid(t)
        else:
            raise ValueError(f"unknown timestep distribution: {distribution}")

        t = t.clamp(eps, 1.0 - eps)
        ctx.set(self.cfg.get("output", "noise.t"), t)


@register_op("flow_matching_prepare")
class FlowMatchingPrepareOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        x1 = ctx.get(self.cfg["x1"])
        x0_spec = self.cfg.get("x0")
        x0 = resolve_input(x0_spec, ctx) if x0_spec is not None else torch.randn_like(x1)
        t = ctx.get(self.cfg["t"])
        view_shape = [t.shape[0]] + [1] * (x1.ndim - 1)
        tv = t.view(*view_shape)
        xt = (1.0 - tv) * x0 + tv * x1
        target_v = x1 - x0
        ctx.set(self.cfg.get("xt_output", "latent.noisy"), xt)
        ctx.set(self.cfg.get("target_output", "target.v"), target_v)


@register_op("dmd_proxy_target")
class DMDProxyTargetOp:
    """
    Minimal target-trick helper:
        target = (pred - grad * scale).detach()
    Then 0.5 * mse(pred, target) gives gradient approximately grad * scale.
    The actual teacher/guidance score computation should be implemented in a task-specific op
    and written to ctx as `grad`.
    """

    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        pred = ctx.get(self.cfg["pred"])
        grad = ctx.get(self.cfg["grad"])
        scale = float(self.cfg.get("scale", 1.0))
        target = (pred - grad * scale).detach()
        ctx.set(self.cfg.get("output", "dmd.target"), target)


@register_op("nch_ldm_v3_two_step")
class NCHLDMV3TwoStepOp:
    """
    Task-specific two-step NCH LDM training flow.

    The op owns no parameters. It reads latents and text context from TrainContext,
    calls a DiT component for each configured timestep, and writes the final latent
    back to TrainContext. LoRA/checkpoint/FSDP ownership stays with the DiT component.
    """

    _INPUT_TYPES = {"noise_one_lq", "noise_zero_lq", "lq_one_lq", "lq_zero_lq"}

    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))
        self.component_name = self.cfg.get("component") or self.cfg.get("dit")
        if not self.component_name:
            raise ValueError("nch_ldm_v3_two_step requires 'component' or 'dit'")

        self.input_type = str(self.cfg["input_type"])
        if self.input_type not in self._INPUT_TYPES:
            raise ValueError(
                f"unsupported input_type for nch_ldm_v3_two_step: {self.input_type!r}"
            )

        self.timesteps = [float(t) for t in self.cfg["timesteps"]]
        if not self.timesteps:
            raise ValueError("nch_ldm_v3_two_step requires at least one timestep")

        self.enable_skip_level = self.cfg.get("enable_skip_level")
        self.mask_repeat = int(self.cfg.get("mask_repeat", 2))
        if self.mask_repeat < 1:
            raise ValueError("nch_ldm_v3_two_step mask_repeat must be >= 1")

        self.concat_dim = int(self.cfg.get("concat_dim", 1))
        self.model_output_index = self.cfg.get("model_output_index", 0)
        self.model_output_key = self.cfg.get("model_output_key")

    def __call__(self, ctx, components):
        inputs = dict(self.cfg.get("inputs", {}) or {})
        if "hidden_states" not in inputs:
            raise KeyError("nch_ldm_v3_two_step inputs must include hidden_states")
        if "encoder_hidden_states" not in inputs:
            raise KeyError(
                "nch_ldm_v3_two_step inputs must include encoder_hidden_states"
            )

        hidden_states = resolve_input(inputs["hidden_states"], ctx)
        encoder_hidden_states = resolve_input(inputs["encoder_hidden_states"], ctx)

        if not torch.is_tensor(hidden_states):
            raise TypeError(
                "nch_ldm_v3_two_step hidden_states must resolve to a tensor, "
                f"got {type(hidden_states)}"
            )
        if hidden_states.ndim < 2:
            raise ValueError(
                "nch_ldm_v3_two_step hidden_states must have batch and channel dims"
            )

        if torch.is_tensor(encoder_hidden_states):
            encoder_hidden_states = encoder_hidden_states.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        if self.input_type.startswith("noise"):
            x_start = torch.randn_like(hidden_states)
        else:
            x_start = hidden_states

        if "_zero_" in self.input_type:
            mask = torch.zeros_like(hidden_states)
        else:
            mask = torch.ones_like(hidden_states)

        repeat_shape = [1] * hidden_states.ndim
        repeat_shape[self.concat_dim] = self.mask_repeat
        mask = torch.tile(mask, tuple(repeat_shape))
        mask_image_latents = hidden_states

        model = components[self.component_name]
        extra_kwargs = resolve_kwargs(self.cfg.get("extra_inputs", {}) or {}, ctx)
        dims = [1] * (hidden_states.ndim - 1)
        ts = self.timesteps + [0.0]

        generated_image = None
        for timestep_value, next_timestep_value in zip(ts[:-1], ts[1:]):
            timestep = torch.full(
                (hidden_states.shape[0],),
                timestep_value,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            timestep_diff = torch.full(
                (hidden_states.shape[0],),
                timestep_value - next_timestep_value,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            model_input = torch.cat(
                [x_start, mask, mask_image_latents],
                dim=self.concat_dim,
            )
            model_kwargs = dict(extra_kwargs)
            model_kwargs.update(
                {
                    "hidden_states": model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )
            if self.enable_skip_level is not None:
                model_kwargs["enable_skip_level"] = self.enable_skip_level

            model_result = model(**model_kwargs)
            model_output = self._select_model_output(model_result)
            timestep_diff = timestep_diff.view(timestep_diff.size(0), *dims)
            generated_image = x_start - timestep_diff * model_output
            x_start = generated_image

        result = {"out": generated_image}
        outputs = self.cfg.get("outputs")
        if outputs is not None:
            _write_outputs(ctx, result, outputs)
        else:
            ctx.set(self.cfg.get("output", "pred.out"), generated_image)

    def _select_model_output(self, model_result):
        if self.model_output_key is not None:
            return model_result[self.model_output_key]

        if isinstance(model_result, (tuple, list)):
            return model_result[int(self.model_output_index)]

        return model_result


@register_op("nch_mmdit_sr")
class NCHMMDiTSROp:
    """
    NCH MMDiT super-resolution denoising op.

    Adapts the new NCHTransformer2DModel forward signature
    (hidden_states, encoder_hidden_states, timestep, enable_skip_level) ->
    Transformer2DModelOutput(sample) to the framework op protocol.

    Input construction (per user spec):
        The DiT x_embedder expects in_channels=1024. This is produced by
        concatenating [x_start, mask, mask_image_latents] along channel dim,
        where x_start is the noisy/identity latent (64ch), mask is 2x the
        latent channels (128ch), and mask_image_latents is the LQ latent
        (64ch). 64*(1+2+1)=256, then a 2x2 patchify inside the DiT yields
        256*4=1024 channels matching x_embedder.

        - input_type "lq": x_start = lq latent (LQ-as-start, no noise)
        - input_type "noise": x_start = randn_like(lq latent)
        - mask_repeat controls the mask channel multiplier (default 2)

    Flow-matching single-step denoising:
        timestep t in [0,1]; the op runs the DiT once at the configured
        timestep and writes the model output (Transformer2DModelOutput.sample)
        back to context. The model output is a velocity (v) prediction; the
        denoised latent is x0 = x_start - (1-t)*v for LQ-as-start, or
        x0 = x_start + (1-t)*v depending on convention. We expose the raw
        model output and a `denoise` flag to control post-processing.

    Device handling: encoder_hidden_states from EmbeddingDB may be on CPU;
    this op moves it to the latent's device before calling the model.
    """

    _INPUT_TYPES = {"noise", "lq"}

    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))
        self.component_name = self.cfg.get("component") or self.cfg.get("dit")
        if not self.component_name:
            raise ValueError("nch_mmdit_sr requires 'component' or 'dit'")

        self.input_type = str(self.cfg.get("input_type", "lq"))
        if self.input_type not in self._INPUT_TYPES:
            raise ValueError(
                f"unsupported input_type for nch_mmdit_sr: {self.input_type!r}; "
                f"must be one of {self._INPUT_TYPES}"
            )

        self.timestep = float(self.cfg.get("timestep", 1.0))
        self.mask_repeat = int(self.cfg.get("mask_repeat", 2))
        if self.mask_repeat < 1:
            raise ValueError("nch_mmdit_sr mask_repeat must be >= 1")

        self.concat_dim = int(self.cfg.get("concat_dim", 1))
        self.enable_skip_level = self.cfg.get("enable_skip_level")
        self.denoise = bool(self.cfg.get("denoise", True))
        self.extract_a_tex = bool(self.cfg.get("extract_a_tex", False))

    def __call__(self, ctx, components):
        inputs = dict(self.cfg.get("inputs", {}) or {})
        if "latent" not in inputs:
            raise KeyError("nch_mmdit_sr inputs must include 'latent' (lq latent)")
        if "encoder_hidden_states" not in inputs:
            raise KeyError(
                "nch_mmdit_sr inputs must include 'encoder_hidden_states'"
            )

        latent = resolve_input(inputs["latent"], ctx)
        encoder_hidden_states = resolve_input(inputs["encoder_hidden_states"], ctx)

        if not torch.is_tensor(latent):
            raise TypeError(
                f"nch_mmdit_sr latent must resolve to a tensor, got {type(latent)}"
            )
        if latent.ndim < 4:
            raise ValueError(
                "nch_mmdit_sr latent must have batch and channel dims (NCHW)"
            )

        if torch.is_tensor(encoder_hidden_states):
            encoder_hidden_states = encoder_hidden_states.to(
                device=latent.device, dtype=latent.dtype
            )

        if self.input_type == "noise":
            x_start = torch.randn_like(latent)
        else:
            x_start = latent

        mask = torch.ones_like(latent)
        repeat_shape = [1] * latent.ndim
        repeat_shape[self.concat_dim] = self.mask_repeat
        mask = torch.tile(mask, tuple(repeat_shape))
        mask_image_latents = latent

        model_input = torch.cat(
            [x_start, mask, mask_image_latents],
            dim=self.concat_dim,
        )

        bs = latent.shape[0]
        timestep = torch.full(
            (bs,),
            self.timestep,
            device=latent.device,
            dtype=latent.dtype,
        )

        model = components[self.component_name]
        model_kwargs = {
            "hidden_states": model_input,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }
        if self.enable_skip_level is not None:
            model_kwargs["enable_skip_level"] = self.enable_skip_level
        if self.extract_a_tex:
            model_kwargs["extract_a_tex"] = True

        model_result = model(**model_kwargs)

        sample = self._extract_sample(model_result)

        a_tex = None
        if self.extract_a_tex and isinstance(model_result, dict):
            a_tex = model_result.get("a_tex")

        if self.denoise:
            dt = 1.0 - self.timestep
            dt_view = [1] * sample.ndim
            dt_view[0] = bs
            sample = x_start - dt * sample

        result = {"latent_denoised": sample}
        if a_tex is not None:
            result["a_tex"] = a_tex
        outputs = self.cfg.get("outputs")
        if outputs is not None:
            _write_outputs(ctx, result, outputs)
        else:
            ctx.set(self.cfg.get("output", "pred.latent_denoised"), sample)
            if a_tex is not None:
                ctx.set(self.cfg.get("a_tex_output", "pred.a_tex"), a_tex)

    @staticmethod
    def _extract_sample(model_result):
        if hasattr(model_result, "sample"):
            return model_result.sample
        if isinstance(model_result, dict):
            return model_result["sample"]
        if isinstance(model_result, (tuple, list)):
            return model_result[0]
        return model_result


@register_op("jsd_decode")
class JSDDecodeOp:
    """
    Joint Segmentation Decoder op for TADiSR stage 3.

    Calls a ``JointSegDecoder`` component with the denoised latent and the
    text-aware attention feature ``a_tex`` produced by the DiT TACA path,
    writing both the super-resolved image (``recon``) and the text
    segmentation mask (``seg``) back to context.

    Inputs (under ``inputs``):
        latent  : denoised latent tensor  [B, C, H, W]
        a_tex   : text-aware attention    [B, S_img, a_tex_channels]

    Outputs (under ``outputs``):
        recon   : super-resolved image    [B, 3, H, W]
        seg     : text segmentation mask  [B, 1, H, W] in [0, 1]
    """

    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))
        self.component_name = self.cfg.get("component") or self.cfg.get("jsd")
        if not self.component_name:
            raise ValueError("jsd_decode requires 'component' or 'jsd'")

    def __call__(self, ctx, components):
        inputs = dict(self.cfg.get("inputs", {}) or {})
        if "latent" not in inputs:
            raise KeyError("jsd_decode inputs must include 'latent'")
        if "a_tex" not in inputs:
            raise KeyError("jsd_decode inputs must include 'a_tex'")

        latent = resolve_input(inputs["latent"], ctx)
        a_tex = resolve_input(inputs["a_tex"], ctx)

        if not torch.is_tensor(latent):
            raise TypeError(
                f"jsd_decode latent must resolve to a tensor, got {type(latent)}"
            )
        if not torch.is_tensor(a_tex):
            raise TypeError(
                f"jsd_decode a_tex must resolve to a tensor, got {type(a_tex)}"
            )

        a_tex = a_tex.to(device=latent.device, dtype=latent.dtype)

        model = components[self.component_name]
        result = model(latent, a_tex)

        outputs = self.cfg.get("outputs")
        if outputs is not None:
            _write_outputs(ctx, result, outputs)
        else:
            recon = result["recon"] if isinstance(result, dict) else result
            ctx.set(self.cfg.get("recon_output", "pred.rgb"), recon)
            if isinstance(result, dict) and "seg" in result:
                ctx.set(self.cfg.get("seg_output", "pred.seg"), result["seg"])
