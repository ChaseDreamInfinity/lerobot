#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorRT inference for PI0.5.

Three engines, in one directory:

- ``vision.engine``: SigLIP + projector for all camera images of a prediction in one batch (fp16).
- ``kv.engine``: the prefix KV-cache fill (Gemma 2B, 18 layers). Exported from an fp16 copy of
  the model and built strongly typed, so the model's explicit fp32 upcasts in normalization and
  attention survive. Its hidden states peak at ~22k, within fp16's range (the last layer's output
  overflows, but the KV cache does not use it). A plain fp16 build returns silently wrong K/V.
- ``step.engine``: one denoising step of the action expert, reading fp16 K/V (fp16, strongly
  typed for the same reason). The expert's adaptive norms take their per-timestep modulations as
  an input (``expert_modulations``), computed once per timestep at attach time: they depend only
  on the timestep, and computing them in the engine would re-read 475 MB of fp32 weights per step.

fp16 is more accurate here than the model's own bf16: actions are several times closer to an
fp32 reference. At run time the engines plug into the CUDA-graph sampler of ``cuda_graph_pi05``,
so Real-Time Chunking works unchanged and the empty camera slot comes from its cache. Each engine
is checked against the PyTorch model when attached.

Build the engines once per checkpoint, camera count and device (about 4 min on an AGX Orin)::

    uv run --no-sync --with onnx python -m lerobot.policies.pi05.tensorrt_pi05 \\
        --policy-path=<repo_or_dir> --engine-dir=pi05_trt --num-cameras=2

Then attach them with ``attach_tensorrt_engines(policy, "pi05_trt")``
(``lerobot-rollout --tensorrt_engine=<dir>`` does this).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import logging
import tempfile
from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.utils.tensorrt import TensorRTEngine, build_engine, export_onnx

from . import modeling_pi05
from .cuda_graph_pi05 import GraphedSampler, _capture_with_dummy_prediction, pytorch_fill_kv
from .modeling_pi05 import PI05Policy

logger = logging.getLogger(__name__)

VISION_ENGINE, KV_ENGINE, STEP_ENGINE = "vision.engine", "kv.engine", "step.engine"
KV_INPUTS = ["prefix_embs", "prefix_pad_masks", "prefix_att_masks"]
# Load-time self-check tolerance, relative to each reference output's magnitude. Measured fp16
# errors against the bf16 model on the SO-101 checkpoint are 0.004-0.03; a mismatched checkpoint
# is O(1).
DEFAULT_SELF_CHECK_RTOL = 0.1


def _kv_names(n_layers: int) -> list[str]:
    return [f"{kind}_{i}" for i in range(n_layers) for kind in ("k", "v")]


def _step_inputs(n_layers: int) -> list[str]:
    return ["x_t", "modulations", "prefix_pad_masks", *_kv_names(n_layers)]


def _expert_norms(model: modeling_pi05.PI05Pytorch) -> list[nn.Module]:
    """The action expert's adaptive RMSNorms, in a fixed order: per layer input and
    post-attention norm, then the final norm."""
    expert = model.paligemma_with_expert.gemma_expert.model
    return [
        norm for layer in expert.layers for norm in (layer.input_layernorm, layer.post_attention_layernorm)
    ] + [expert.norm]


@torch.no_grad()
def expert_modulations(model: modeling_pi05.PI05Pytorch, timestep: Tensor) -> Tensor:
    """Per-norm AdaRMS modulations (scale, shift, gate) for one timestep: ``[n_norms, 3 * width]``.

    They depend only on the timestep, so they are computed here once per timestep, with the
    model's own ops (including the float64 time embedding, which TensorRT does not support), and
    fed to the step engine. Inside the engine they would re-read 37 fp32 3072x1024 weight
    matrices (475 MB) on every denoising step.
    """
    _, _, _, adarms_cond = model.embed_suffix(
        torch.zeros(1, model.config.chunk_size, model.config.max_action_dim, device=timestep.device), timestep
    )
    return torch.cat([norm.dense(adarms_cond) for norm in _expert_norms(model)])


@contextlib.contextmanager
def _norms_read_modulations(model: modeling_pi05.PI05Pytorch, modulations: list[Tensor]):
    """Make each expert norm use its precomputed modulation instead of ``dense(cond)``."""
    norms = _expert_norms(model)

    def modulated_forward(norm, modulation):
        def forward(x, cond=None):
            dtype = x.dtype
            normed = norm._norm(x)
            mod = modulation.unsqueeze(1) if len(x.shape) == 3 else modulation
            scale, shift, gate = mod.chunk(3, dim=-1)
            normed = normed * (1 + scale.float()) + shift.float()
            return normed.to(dtype), gate.to(dtype)

        return forward

    for norm, modulation in zip(norms, modulations, strict=True):
        norm.forward = modulated_forward(norm, modulation)
    try:
        yield
    finally:
        for norm in norms:
            del norm.forward  # back to the class's forward


def _att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """``make_att_2d_masks`` in int32: TensorRT cannot sum or multiply bool tensors."""
    pad = pad_masks.to(torch.int32)
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    return (cumsum[:, None, :] <= cumsum[:, :, None]) & (pad[:, None, :] * pad[:, :, None] > 0)


class VisionExport(nn.Module):
    """``PaliGemmaWithExpertModel.embed_image`` for a batch of camera images (SigLIP + projector)."""

    def __init__(self, paligemma_with_expert: nn.Module):
        super().__init__()
        self.paligemma_with_expert = paligemma_with_expert

    def forward(self, images: Tensor) -> Tensor:
        return self.paligemma_with_expert.embed_image(images)


class KVFillExport(nn.Module):
    """Prefix pass of ``PI05Pytorch.sample_actions``. Returns per-layer (K, V) as a flat tuple."""

    def __init__(self, model: modeling_pi05.PI05Pytorch):
        super().__init__()
        self.model = model

    def forward(self, prefix_embs: Tensor, prefix_pad_masks: Tensor, prefix_att_masks: Tensor):
        model = self.model
        language_model = model.paligemma_with_expert.paligemma.model.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        pad = prefix_pad_masks.to(torch.int32)
        # PiGemmaModel casts its input to bf16 only for bf16 weights (see StepExport).
        weight_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=model._prepare_attention_masks_4d(
                _att_2d_masks(prefix_pad_masks, prefix_att_masks)
            ),
            position_ids=(torch.cumsum(pad, dim=1) - 1).to(torch.int64),
            past_key_values=None,
            inputs_embeds=[prefix_embs.to(weight_dtype), None],
            use_cache=True,
        )
        return tuple(t for layer in cache.layers for t in (layer.keys, layer.values))


class StepExport(nn.Module):
    """``PI05Pytorch.denoise_step`` with the KV cache and the norm modulations
    (``expert_modulations``) as flat inputs, mask arithmetic in int32."""

    def __init__(self, model: modeling_pi05.PI05Pytorch):
        super().__init__()
        self.model = model

    def forward(self, x_t: Tensor, modulations: Tensor, prefix_pad_masks: Tensor, *kv: Tensor) -> Tensor:
        from transformers.cache_utils import DynamicCache

        model = self.model
        cache = DynamicCache(ddp_cache_data=[(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)])
        # embed_suffix for pi05: the action tokens are action_in_proj(x_t), all valid, one
        # attention block ([1, 0, 0, ...]); the timestep only enters through the modulations.
        expert = model.paligemma_with_expert.gemma_expert.model
        # PiGemmaModel casts its input to bf16 only for bf16 weights; cast to the weights' own
        # dtype, so an fp16 export copy works too (identical for bf16).
        suffix_embs = model.action_in_proj(x_t).to(expert.layers[0].self_attn.q_proj.weight.dtype)
        batch_size, suffix_len = suffix_embs.shape[:2]
        prefix_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_pad_masks.shape[1])
        suffix_2d = torch.ones(batch_size, suffix_len, suffix_len, dtype=torch.bool, device=x_t.device)
        full_2d = torch.cat([prefix_2d, suffix_2d], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int32), dim=-1)[:, None]
        steps = torch.arange(1, suffix_len + 1, dtype=torch.int32, device=x_t.device)[None, :]
        position_ids = prefix_offsets + steps - 1
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        with _norms_read_modulations(model, list(modulations[:, None, :])):
            outputs_embeds, _ = model.paligemma_with_expert.forward(
                attention_mask=model._prepare_attention_masks_4d(full_2d),
                position_ids=position_ids.to(torch.int64),
                past_key_values=cache,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                # Not None, so the norms take the adaptive path; their modulations are inputs.
                adarms_cond=[None, modulations[:1]],
            )
        suffix_out = outputs_embeds[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)


def fp16_copy(model: nn.Module) -> nn.Module:
    """Copy with bf16 weights and buffers converted to fp16; fp32 modules stay fp32, as in bf16."""
    model = copy.deepcopy(model).eval()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.dtype == torch.bfloat16:
            tensor.data = tensor.data.half()
    return model


def _example_inputs(policy: PI05Policy, n_cameras: int, device):
    """Placeholder ``sample_actions`` inputs: ``n_cameras`` images, the remaining slots empty."""
    config = policy.config
    height, width = config.image_resolution
    generator = torch.Generator(device=device).manual_seed(0)
    images = [
        torch.rand(1, 3, height, width, device=device, generator=generator) * 2 - 1
        if i < n_cameras
        else torch.full((1, 3, height, width), -1.0, device=device)
        for i in range(len(config.image_features))
    ]
    img_masks = [torch.tensor([i < n_cameras], device=device) for i in range(len(images))]
    tokens = torch.randint(0, 1000, (1, config.tokenizer_max_length), device=device, generator=generator)
    masks = torch.zeros(1, config.tokenizer_max_length, dtype=torch.bool, device=device)
    masks[:, : config.tokenizer_max_length // 4] = True
    return images, img_masks, tokens, masks


@torch.no_grad()
def export_onnx_files(policy: PI05Policy, out_dir: str | Path, n_cameras: int) -> dict[str, Path]:
    """Export the three engines' ONNX files. Needs the ``onnx`` package and a CUDA policy.

    The KV fill exceeds ONNX's 2 GB single-file limit, so its weights go to external data files
    next to ``kv.onnx``; each ONNX file gets its own subdirectory.
    """
    out_dir = Path(out_dir)
    model = policy.model
    device = next(policy.parameters()).device
    images, img_masks, tokens, masks = _example_inputs(policy, n_cameras, device)
    prefix_embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    n_layers = len(model.paligemma_with_expert.gemma_expert.model.layers)

    # The vision tower already runs in fp32; TensorRT picks fp16 kernels for it.
    paths = {
        "vision": export_onnx(
            VisionExport(model.paligemma_with_expert),
            [torch.cat(images[:n_cameras])],
            out_dir / "vision" / "vision.onnx",
            ["images"],
            ["image_embs"],
        )
    }
    model16 = fp16_copy(model)
    kv_args = [prefix_embs, pad_masks, att_masks]
    paths["kv"] = export_onnx(
        KVFillExport(model16), kv_args, out_dir / "kv" / "kv.onnx", KV_INPUTS, _kv_names(n_layers)
    )
    kv = KVFillExport(model16)(*kv_args)
    x_t = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device=device)
    modulations = expert_modulations(model16, torch.full((1,), 0.7, device=device))
    paths["step"] = export_onnx(
        StepExport(model16),
        [x_t, modulations, pad_masks, *kv],
        out_dir / "step" / "step.onnx",
        _step_inputs(n_layers),
        ["v_t"],
    )
    return paths


def build_engines(policy: PI05Policy, engine_dir: str | Path, n_cameras: int) -> Path:
    """Export and build the three engines into ``engine_dir``. ONNX files are not kept (they go to
    a temporary directory next to ``engine_dir``: the KV fill's is ~3.5 GB)."""
    engine_dir = Path(engine_dir)
    engine_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=engine_dir.parent) as tmp:
        onnx = export_onnx_files(policy, tmp, n_cameras)
        build_engine(onnx["vision"], engine_dir / VISION_ENGINE, fp16=True)
        build_engine(onnx["kv"], engine_dir / KV_ENGINE, strongly_typed=True)
        build_engine(onnx["step"], engine_dir / STEP_ENGINE, strongly_typed=True)
    return engine_dir


def _relative_error(actual: Tensor, reference: Tensor) -> float:
    return (
        (actual.float() - reference.float()).abs().max() / reference.float().abs().max().clamp_min(1e-6)
    ).item()


@torch.no_grad()
def attach_tensorrt_engines(
    policy: PI05Policy,
    engine_dir: str | Path,
    present_image_keys: list[str] | None = None,
    rtol: float = DEFAULT_SELF_CHECK_RTOL,
) -> GraphedSampler:
    """Run PI0.5's image encoding, KV fill and denoising steps on TensorRT engines.

    Each engine is checked against the PyTorch model first, so engines from another checkpoint or
    a broken build fail here instead of driving the robot. ``present_image_keys`` captures the
    CUDA graphs now (see ``cuda_graph_pi05.enable_cuda_graphs``).
    """
    if policy.name != "pi05":
        raise ValueError(f"Expected a PI0.5 policy, got policy type '{policy.name}'")
    model = policy.model
    config = model.config
    device = next(policy.parameters()).device
    n_layers = len(model.paligemma_with_expert.gemma_expert.model.layers)
    engine_dir = Path(engine_dir)
    vision = TensorRTEngine(engine_dir / VISION_ENGINE, ["images"], ["image_embs"], device=device)
    kv_engine = TensorRTEngine(engine_dir / KV_ENGINE, KV_INPUTS, _kv_names(n_layers), device=device)
    step_engine = TensorRTEngine(engine_dir / STEP_ENGINE, _step_inputs(n_layers), ["v_t"], device=device)
    n_cameras = vision.shapes["images"][0]
    embed_image = model.paligemma_with_expert.embed_image

    # The denoising loop only visits these timesteps (GraphedSampler runs the configured number of
    # steps), so their modulations are a table, looked up on the device inside the graph.
    num_steps = config.num_inference_steps
    times = [1.0 - step / num_steps for step in range(num_steps)]
    modulation_table = torch.stack(
        [expert_modulations(model, torch.full((1,), t, device=device)) for t in times]
    )

    def encode_images(images: Tensor) -> Tensor:
        if tuple(images.shape) != vision.shapes["images"]:  # other camera count: eager
            return torch.cat([embed_image(img[None]) for img in images])
        return vision(images)

    def fill_kv(embs: Tensor, pad_masks: Tensor, att_masks: Tensor):
        return kv_engine(embs, pad_masks, att_masks)

    def denoise(x_t: Tensor, timestep: Tensor, pad_masks: Tensor, kv) -> Tensor:
        index = torch.round((1.0 - timestep) * num_steps).long().clamp(0, num_steps - 1)
        return step_engine(x_t, modulation_table.index_select(0, index)[0], pad_masks, *kv)

    # Self-check each engine against PyTorch on placeholder inputs.
    images, img_masks, tokens, masks = _example_inputs(policy, n_cameras, device)
    errors = {
        "vision": _relative_error(
            vision(torch.cat(images[:n_cameras])), torch.cat([embed_image(i) for i in images[:n_cameras]])
        )
    }
    embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    reference_kv = pytorch_fill_kv(model)(embs, pad_masks, att_masks)
    engine_kv = fill_kv(embs, pad_masks, att_masks)
    flat_reference_kv = [t for layer in reference_kv.layers for t in (layer.keys, layer.values)]
    errors["kv"] = max(_relative_error(a, b) for a, b in zip(engine_kv, flat_reference_kv, strict=True))
    x_t = torch.randn(1, config.chunk_size, config.max_action_dim, device=device)
    timestep = torch.full((1,), times[num_steps // 2], device=device)
    reference_v = model.denoise_step(pad_masks, reference_kv, x_t, timestep)
    errors["step"] = _relative_error(denoise(x_t, timestep, pad_masks, flat_reference_kv), reference_v)
    for part, error in errors.items():
        if not error <= rtol:  # also catches NaN
            raise RuntimeError(
                f"TensorRT {part} engine in {engine_dir} disagrees with the PyTorch policy (relative error "
                f"{error:.4f} > {rtol}). It was probably built from another checkpoint; rebuild it."
            )

    sampler = GraphedSampler(model, fill_kv=fill_kv, denoise=denoise, encode_images=encode_images)
    model.sample_actions = sampler
    if present_image_keys is not None:
        _capture_with_dummy_prediction(policy, sampler, present_image_keys)
    logger.info(
        "TensorRT engines in %s attached for %d cameras (self-check relative error: %s)",
        engine_dir,
        n_cameras,
        ", ".join(f"{part} {error:.4f}" for part, error in errors.items()),
    )
    return sampler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True, help="Hub repo id or local checkpoint directory")
    parser.add_argument("--engine-dir", required=True, type=Path, help="Directory to write the engines to")
    parser.add_argument(
        "--num-cameras", type=int, default=2, help="Camera images per prediction (not empty slots)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    policy = PI05Policy.from_pretrained(args.policy_path).to("cuda").eval()
    logger.info("Exporting and building TensorRT engines (takes a few minutes)")
    build_engines(policy, args.engine_dir, args.num_cameras)
    attach_tensorrt_engines(policy, args.engine_dir)
    print(f"Engines written to {args.engine_dir}")


if __name__ == "__main__":
    main()
