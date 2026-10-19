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
"""TensorRT inference for SmolVLA.

Three engines, in one directory:

- ``vision.engine``: SigLIP + connector for all camera images of a prediction in one batch (fp16).
- ``kv.engine``: the prefix KV-cache fill (16 VLM layers). Exported from an fp16 copy of the model
  and built strongly typed, so the model's explicit fp32 upcasts in normalization and attention
  survive: prefix activations reach ~7e3, and a plain fp16 build overflows inside layers and
  returns silently wrong K/V.
- ``step.engine``: one denoising step of the action expert, reading fp16 K/V (fp16).

At run time they plug into the CUDA-graph sampler of ``cuda_graph_smolvla``: the KV engine and
the step engine are replayed from CUDA graphs, the flow-matching loop stays in Python so Real-Time
Chunking works unchanged, and empty camera slots come from its constant-image cache. Each engine
is checked against the PyTorch model when attached.

Build the engines once per checkpoint, camera count and device::

    uv run --no-sync --with onnx python -m lerobot.policies.smolvla.tensorrt_smolvla \\
        --policy-path=<repo_or_dir> --engine-dir=smolvla_trt --num-cameras=2

Then attach them with ``attach_tensorrt_engines(policy, "smolvla_trt")``
(``lerobot-rollout --tensorrt_engine=<dir>`` does this).
"""

from __future__ import annotations

import argparse
import copy
import logging
import tempfile
import types
from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.utils.tensorrt import TensorRTEngine, build_engine, export_onnx

from . import modeling_smolvla
from .cuda_graph_smolvla import GraphedSampler, _capture_with_dummy_prediction, pytorch_fill_kv
from .modeling_smolvla import SmolVLAPolicy

logger = logging.getLogger(__name__)

VISION_ENGINE, KV_ENGINE, STEP_ENGINE = "vision.engine", "kv.engine", "step.engine"
KV_INPUTS = ["prefix_embs", "prefix_pad_masks", "prefix_att_masks"]
# Load-time self-check tolerance, relative to each reference output's magnitude. Measured fp16
# errors on the SO-101 checkpoint are 0.002-0.03; a mismatched checkpoint is O(1).
DEFAULT_SELF_CHECK_RTOL = 0.1


def _kv_names(n_layers: int) -> list[str]:
    return [f"{kind}_{i}" for i in range(n_layers) for kind in ("k", "v")]


def _step_inputs(n_layers: int) -> list[str]:
    return ["x_t", "timestep", "prefix_pad_masks", *_kv_names(n_layers)]


def _siglip_embeddings_forward(self, pixel_values: Tensor, patch_attention_mask: Tensor) -> Tensor:
    """``SmolVLMVisionEmbeddings.forward`` (transformers 5.x), exportable to ONNX.

    The original merges position ids with boolean-mask indexing, which ONNX/TensorRT cannot
    express with static shapes; ``torch.where`` gives the same result. Only used on export copies.
    """
    batch_size, _, max_im_h, max_im_w = pixel_values.shape
    embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
    nb_h, nb_w = max_im_h // self.patch_size, max_im_w // self.patch_size
    boundaries = torch.arange(
        1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side, device=pixel_values.device
    )
    step_h = 1.0 / patch_attention_mask[:, :, 0].sum(dim=1)
    step_w = 1.0 / patch_attention_mask[:, 0, :].sum(dim=1)
    h_idx = torch.arange(nb_h, device=pixel_values.device, dtype=torch.float32)
    w_idx = torch.arange(nb_w, device=pixel_values.device, dtype=torch.float32)
    coords_h = torch.clamp(h_idx[None, :] * step_h[:, None], max=(1.0 - 1e-6)).to(pixel_values.dtype)
    coords_w = torch.clamp(w_idx[None, :] * step_w[:, None], max=(1.0 - 1e-6)).to(pixel_values.dtype)
    bucket_h = torch.bucketize(coords_h, boundaries, right=True)
    bucket_w = torch.bucketize(coords_w, boundaries, right=True)
    pos_ids = (bucket_h[:, :, None] * self.num_patches_per_side + bucket_w[:, None, :]).reshape(
        batch_size, -1
    )
    position_ids = torch.where(patch_attention_mask.view(batch_size, -1), pos_ids, torch.zeros_like(pos_ids))
    return embeddings + self.position_embedding(position_ids.to(torch.int64))


class VisionExport(nn.Module):
    """``embed_image`` for a batch of full images, through the vision stages without a mask (as
    eager does for all-true masks). Patches the embeddings of the module it is given: pass a copy."""

    def __init__(self, vlm: nn.Module):
        super().__init__()
        self.vlm = vlm
        embeddings = vlm.vision_model.embeddings
        embeddings.forward = types.MethodType(_siglip_embeddings_forward, embeddings)

    def forward(self, images: Tensor) -> Tensor:
        vision = self.vlm.vision_model
        patch = vision.patch_size
        mask = torch.ones(
            images.shape[0],
            images.shape[2] // patch,
            images.shape[3] // patch,
            dtype=torch.bool,
            device=images.device,
        )
        weight_dtype = vision.embeddings.patch_embedding.weight.dtype
        hidden = vision.embeddings(pixel_values=images.to(weight_dtype), patch_attention_mask=mask)
        hidden = vision.encoder(inputs_embeds=hidden, attention_mask=None).last_hidden_state
        return self.vlm.connector(vision.post_layernorm(hidden))


class KVFillExport(nn.Module):
    """Prefix pass of ``VLAFlowMatching.sample_actions``, mask arithmetic in int32 (TensorRT cannot
    sum or multiply bool tensors). Returns per-layer (K, V) as a flat tuple."""

    def __init__(self, model: modeling_smolvla.VLAFlowMatching):
        super().__init__()
        self.model = model

    def forward(self, prefix_embs: Tensor, prefix_pad_masks: Tensor, prefix_att_masks: Tensor):
        pad = prefix_pad_masks.to(torch.int32)
        cumsum = torch.cumsum(prefix_att_masks.to(torch.int32), dim=1)
        att_2d = (cumsum[:, None, :] <= cumsum[:, :, None]) & (pad[:, None, :] * pad[:, :, None] > 0)
        _, kv = self.model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=(torch.cumsum(pad, dim=1) - 1).to(torch.int64),
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return tuple(t for i in range(len(kv)) for t in (kv[i]["key_states"], kv[i]["value_states"]))


class StepExport(nn.Module):
    """``VLAFlowMatching.denoise_step`` with the K/V cache as flat inputs, mask arithmetic in int32."""

    def __init__(self, model: modeling_smolvla.VLAFlowMatching):
        super().__init__()
        self.model = model

    def forward(self, x_t: Tensor, timestep: Tensor, prefix_pad_masks: Tensor, *kv: Tensor) -> Tensor:
        model = self.model
        past = {i: {"key_states": kv[2 * i], "value_states": kv[2 * i + 1]} for i in range(len(kv) // 2)}
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)
        batch_size, suffix_len = suffix_pad_masks.shape
        prefix_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_pad_masks.shape[1])
        pad = suffix_pad_masks.to(torch.int32)
        cumsum = torch.cumsum(suffix_att_masks.to(torch.int32), dim=1)
        suffix_2d = (cumsum[:, None, :] <= cumsum[:, :, None]) & (pad[:, None, :] * pad[:, :, None] > 0)
        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int32), dim=-1)[:, None]
        outputs, _ = model.vlm_with_expert.forward(
            attention_mask=torch.cat([prefix_2d, suffix_2d], dim=2),
            position_ids=(prefix_offsets + torch.cumsum(pad, dim=1) - 1).to(torch.int64),
            past_key_values=past,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        return model.action_out_proj(outputs[1][:, -model.config.chunk_size :].to(torch.float32))


def fp16_copy(model: nn.Module) -> nn.Module:
    """Copy with bf16 weights and buffers converted to fp16; fp32 modules stay fp32, as in bf16."""
    model = copy.deepcopy(model).eval()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.dtype == torch.bfloat16:
            tensor.data = tensor.data.half()
    return model


def _example_inputs(policy: SmolVLAPolicy, n_cameras: int, device):
    """Placeholder model inputs shaped like ``prepare_images``/tokenizer output (values random)."""
    config = policy.config
    height, width = config.resize_imgs_with_padding or next(iter(config.image_features.values())).shape[1:]
    n_images = n_cameras + min(len(config.image_features) - n_cameras, config.empty_cameras)
    generator = torch.Generator(device=device).manual_seed(0)
    images = [
        torch.rand(1, 3, height, width, device=device, generator=generator) * 2 - 1 for _ in range(n_images)
    ]
    img_masks = [torch.ones(1, dtype=torch.bool, device=device) for _ in range(n_images)]
    lang_tokens = torch.randint(0, 1000, (1, config.tokenizer_max_length), device=device, generator=generator)
    lang_masks = torch.ones(1, config.tokenizer_max_length, dtype=torch.bool, device=device)
    state = torch.randn(1, config.max_state_dim, device=device, generator=generator)
    return images, img_masks, lang_tokens, lang_masks, state


@torch.no_grad()
def export_onnx_files(policy: SmolVLAPolicy, out_dir: str | Path, n_cameras: int) -> dict[str, Path]:
    """Export the three engines' ONNX files. Needs the ``onnx`` package and a CUDA policy."""
    out_dir = Path(out_dir)
    model = policy.model
    device = next(policy.parameters()).device
    images, img_masks, lang_tokens, lang_masks, state = _example_inputs(policy, n_cameras, device)
    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )

    vision = VisionExport(copy.deepcopy(model.vlm_with_expert.get_vlm_model()).float().cpu().eval())
    paths = {
        "vision": export_onnx(
            vision,
            [torch.cat(images[:n_cameras]).float().cpu()],
            out_dir / "vision.onnx",
            ["images"],
            ["image_embs"],
        )
    }

    model16 = fp16_copy(model)
    n_layers = len(model16.vlm_with_expert.lm_expert.layers)
    kv_args = [prefix_embs.half(), pad_masks, att_masks]
    paths["kv"] = export_onnx(
        KVFillExport(model16), kv_args, out_dir / "kv.onnx", KV_INPUTS, _kv_names(n_layers)
    )
    kv = KVFillExport(model16)(*kv_args)
    # TensorRT has no float64; the sinusoidal time embedding is computed in float32 for export.
    get_safe_dtype = modeling_smolvla.get_safe_dtype
    modeling_smolvla.get_safe_dtype = lambda dtype, device: torch.float32 if dtype == torch.float64 else dtype
    try:
        x_t = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device=device)
        paths["step"] = export_onnx(
            StepExport(model16),
            [x_t, torch.tensor([0.7], device=device), pad_masks, *kv],
            out_dir / "step.onnx",
            _step_inputs(n_layers),
            ["v_t"],
        )
    finally:
        modeling_smolvla.get_safe_dtype = get_safe_dtype
    return paths


def build_engines(policy: SmolVLAPolicy, engine_dir: str | Path, n_cameras: int) -> Path:
    """Export and build the three engines into ``engine_dir``. ONNX files are not kept."""
    engine_dir = Path(engine_dir)
    with tempfile.TemporaryDirectory() as tmp:
        onnx = export_onnx_files(policy, tmp, n_cameras)
        build_engine(onnx["vision"], engine_dir / VISION_ENGINE, fp16=True)
        build_engine(onnx["kv"], engine_dir / KV_ENGINE, strongly_typed=True)
        build_engine(onnx["step"], engine_dir / STEP_ENGINE, fp16=True)
    return engine_dir


def _relative_error(actual: Tensor, reference: Tensor) -> float:
    return (
        (actual.float() - reference.float()).abs().max() / reference.float().abs().max().clamp_min(1e-6)
    ).item()


@torch.no_grad()
def attach_tensorrt_engines(
    policy: SmolVLAPolicy,
    engine_dir: str | Path,
    present_image_keys: list[str] | None = None,
    rtol: float = DEFAULT_SELF_CHECK_RTOL,
) -> GraphedSampler:
    """Run SmolVLA's image encoding, KV fill and denoising steps on TensorRT engines.

    Each engine is checked against the PyTorch model first, so engines from another checkpoint or
    a broken build fail here instead of driving the robot. ``present_image_keys`` captures the
    CUDA graphs now (see ``cuda_graph_smolvla.enable_cuda_graphs``).
    """
    if policy.name != "smolvla":
        raise ValueError(f"Expected a SmolVLA policy, got policy type '{policy.name}'")
    model = policy.model
    device = next(policy.parameters()).device
    n_layers = len(model.vlm_with_expert.lm_expert.layers)
    engine_dir = Path(engine_dir)
    vision = TensorRTEngine(engine_dir / VISION_ENGINE, ["images"], ["image_embs"], device=device)
    kv_engine = TensorRTEngine(engine_dir / KV_ENGINE, KV_INPUTS, _kv_names(n_layers), device=device)
    step_engine = TensorRTEngine(engine_dir / STEP_ENGINE, _step_inputs(n_layers), ["v_t"], device=device)
    n_cameras = vision.shapes["images"][0]
    embed_dtype = next(model.vlm_with_expert.get_vlm_model().connector.parameters()).dtype

    def encode_images(images: Tensor) -> Tensor:
        if tuple(images.shape) != vision.shapes["images"]:  # other camera count: eager
            return torch.cat([model.vlm_with_expert.embed_image(img[None]) for img in images])
        return vision(images).to(embed_dtype)

    def fill_kv(embs: Tensor, pad_masks: Tensor, att_masks: Tensor):
        return kv_engine(embs, pad_masks, att_masks)

    def denoise(x_t: Tensor, timestep: Tensor, pad_masks: Tensor, kv) -> Tensor:
        return step_engine(x_t, timestep, pad_masks, *kv)

    # Self-check each engine against PyTorch on placeholder inputs.
    images, img_masks, lang_tokens, lang_masks, state = _example_inputs(policy, n_cameras, device)
    real = torch.cat(images[:n_cameras])
    errors = {
        "vision": _relative_error(
            vision(real), torch.cat([model.vlm_with_expert.embed_image(i) for i in images[:n_cameras]])
        )
    }
    embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    reference_kv = pytorch_fill_kv(model)(embs, pad_masks, att_masks)
    engine_kv = fill_kv(embs, pad_masks, att_masks)
    errors["kv"] = max(
        _relative_error(engine_kv[2 * i + j], reference_kv[i][key])
        for i in range(n_layers)
        for j, key in enumerate(("key_states", "value_states"))
    )
    x_t = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device=device)
    timestep = torch.full((1,), 0.5, device=device)
    reference_v = model.denoise_step(
        x_t=x_t, prefix_pad_masks=pad_masks, past_key_values=reference_kv, timestep=timestep
    )
    flat_reference_kv = [
        reference_kv[i][key] for i in range(n_layers) for key in ("key_states", "value_states")
    ]
    errors["step"] = _relative_error(denoise(x_t, timestep, pad_masks, flat_reference_kv), reference_v)
    for part, error in errors.items():
        if error > rtol:
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

    policy = SmolVLAPolicy.from_pretrained(args.policy_path).to("cuda").eval()
    logger.info("Exporting and building TensorRT engines (takes a few minutes)")
    build_engines(policy, args.engine_dir, args.num_cameras)
    attach_tensorrt_engines(policy, args.engine_dir)
    print(f"Engines written to {args.engine_dir}")


if __name__ == "__main__":
    main()
