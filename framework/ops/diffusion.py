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
