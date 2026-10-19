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
"""TensorRT inference for Diffusion Policy.

Two engines replace the two expensive parts of a prediction:

- ``rgb_encoder.engine``: observation encoding (``_prepare_global_conditioning``: image backbone,
  spatial softmax, concatenation with the state), run once per prediction.
- ``unet.engine``: one denoising step of the conditional UNet, run ``num_inference_steps`` times.

The denoising loop and noise scheduler stay in PyTorch, so the sampler (DDPM/DDIM) and the number
of steps can still be changed at run time without rebuilding. Observation and action queues,
pre/post-processing are untouched.

Build the engines once per checkpoint and device (they are tied to the GPU model, the TensorRT
version and the camera resolution)::

    uv run --no-sync --with onnx python -m lerobot.policies.diffusion.tensorrt_diffusion \\
        --policy-path=<repo_or_dir> --engine-dir=diffusion_fp16

Then attach them at inference time with ``attach_tensorrt_engines(policy, "diffusion_fp16")``
(``lerobot-rollout --tensorrt_engine=<dir>`` does this).
"""

from __future__ import annotations

import argparse
import copy
import logging
import tempfile
from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.tensorrt import TensorRTEngine, build_engine, export_onnx

from .configuration_diffusion import DiffusionConfig
from .modeling_diffusion import DiffusionModel, DiffusionPolicy

logger = logging.getLogger(__name__)

ENCODER_ENGINE = "rgb_encoder.engine"
UNET_ENGINE = "unet.engine"
UNET_INPUTS = ["x", "t", "global_cond"]
# Load-time self-check tolerance, relative to the reference output's magnitude. fp16 engines
# measure ~1e-3 on SO-101 checkpoints; a mismatched checkpoint is O(1).
DEFAULT_SELF_CHECK_RTOL = 0.05

# Engine input name -> batch key, in binding order.
_ENCODER_KEYS = {"state": OBS_STATE, "images": OBS_IMAGES, "env_state": OBS_ENV_STATE}


def encoder_input_names(config: DiffusionConfig) -> list[str]:
    names = ["state"]
    if config.image_features:
        names.append("images")
    if config.env_state_feature:
        names.append("env_state")
    return names


def _example_batch(config: DiffusionConfig, device: torch.device | str) -> dict[str, Tensor]:
    """Random normalized observations, stacked the way ``generate_actions`` receives them."""
    s = config.n_obs_steps
    batch = {OBS_STATE: torch.randn(1, s, *config.robot_state_feature.shape, device=device)}
    if config.image_features:
        shape = next(iter(config.image_features.values())).shape
        batch[OBS_IMAGES] = torch.randn(1, s, len(config.image_features), *shape, device=device)
    if config.env_state_feature:
        batch[OBS_ENV_STATE] = torch.randn(1, s, *config.env_state_feature.shape, device=device)
    return batch


def _example_unet_inputs(config: DiffusionConfig, global_cond: Tensor) -> list[Tensor]:
    x = torch.randn(1, config.horizon, config.action_feature.shape[0], device=global_cond.device)
    t = torch.full((1,), config.num_train_timesteps // 2, dtype=torch.long, device=global_cond.device)
    return [x, t, global_cond]


class _EncoderModule(nn.Module):
    def __init__(self, model: DiffusionModel, input_names: list[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *inputs: Tensor) -> Tensor:
        batch = {_ENCODER_KEYS[name]: x for name, x in zip(self.input_names, inputs, strict=True)}
        return self.model._prepare_global_conditioning(batch)


class _UNetModule(nn.Module):
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.unet = model.unet

    def forward(self, x: Tensor, t: Tensor, global_cond: Tensor) -> Tensor:
        return self.unet(x, t, global_cond=global_cond)


class _TensorRTUNet(nn.Module):
    """Stands in for ``DiffusionModel.unet`` (same call signature)."""

    def __init__(self, engine: TensorRTEngine):
        super().__init__()
        self.engine = engine

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        return self.engine(x, timestep.reshape(1), global_cond)


def export_onnx_files(policy: DiffusionPolicy, out_dir: str | Path) -> tuple[Path, Path]:
    """Export the encoder and one UNet step with a fixed batch size of one. Needs ``onnx``."""
    out_dir = Path(out_dir)
    model = copy.deepcopy(policy.diffusion).cpu().eval()
    names = encoder_input_names(policy.config)
    batch = _example_batch(policy.config, "cpu")
    encoder_inputs = [batch[_ENCODER_KEYS[name]] for name in names]
    with torch.no_grad():
        global_cond = model._prepare_global_conditioning(batch)
    encoder_onnx = export_onnx(
        _EncoderModule(model, names), encoder_inputs, out_dir / "rgb_encoder.onnx", names, ["global_cond"]
    )
    unet_onnx = export_onnx(
        _UNetModule(model),
        _example_unet_inputs(policy.config, global_cond),
        out_dir / "unet.onnx",
        UNET_INPUTS,
        ["eps"],
    )
    return encoder_onnx, unet_onnx


def build_engines(policy: DiffusionPolicy, engine_dir: str | Path, fp16: bool = True) -> Path:
    """Export and build both engines into ``engine_dir``. The ONNX files (~1 GB) are not kept."""
    engine_dir = Path(engine_dir)
    with tempfile.TemporaryDirectory() as tmp:
        encoder_onnx, unet_onnx = export_onnx_files(policy, tmp)
        build_engine(encoder_onnx, engine_dir / ENCODER_ENGINE, fp16=fp16)
        build_engine(unet_onnx, engine_dir / UNET_ENGINE, fp16=fp16)
    return engine_dir


def _relative_error(actual: Tensor, reference: Tensor) -> float:
    return ((actual - reference).abs().max() / reference.abs().max().clamp_min(1.0)).item()


@torch.no_grad()
def attach_tensorrt_engines(
    policy: DiffusionPolicy, engine_dir: str | Path, rtol: float = DEFAULT_SELF_CHECK_RTOL
) -> tuple[TensorRTEngine, TensorRTEngine]:
    """Route the policy's observation encoding and denoising steps through TensorRT engines.

    Both engines are checked against the PyTorch model first, so engines built from a different
    checkpoint, or a broken build, fail here instead of driving the robot. The PyTorch UNet is
    dropped afterwards, freeing its GPU memory (~1 GB in fp32 for the SO-101 checkpoint).
    """
    if policy.name != "diffusion":
        raise ValueError(f"Expected a diffusion policy, got policy type '{policy.name}'")
    policy.eval()
    model = policy.diffusion
    device = next(policy.parameters()).device
    names = encoder_input_names(policy.config)
    engine_dir = Path(engine_dir)
    encoder = TensorRTEngine(engine_dir / ENCODER_ENGINE, names, ["global_cond"], device=device)
    unet = TensorRTEngine(engine_dir / UNET_ENGINE, UNET_INPUTS, ["eps"], device=device)

    # Check each engine on its own: a full sampling run would not be comparable under DDPM,
    # which draws fresh noise at every step.
    batch = _example_batch(policy.config, device)
    reference_cond = model._prepare_global_conditioning(batch)
    encoder_error = _relative_error(encoder(*[batch[_ENCODER_KEYS[n]] for n in names]), reference_cond)
    unet_inputs = _example_unet_inputs(policy.config, reference_cond)
    unet_error = _relative_error(unet(*unet_inputs), model.unet(*unet_inputs[:2], global_cond=unet_inputs[2]))
    for part, error in (("encoder", encoder_error), ("UNet", unet_error)):
        if error > rtol:
            raise RuntimeError(
                f"TensorRT {part} engine in {engine_dir} disagrees with the PyTorch policy (relative "
                f"error {error:.4f} > {rtol}). It was probably built from another checkpoint; rebuild it."
            )

    model._prepare_global_conditioning = lambda b: encoder(*[b[_ENCODER_KEYS[n]] for n in names])
    model.unet = _TensorRTUNet(unet)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(
        "TensorRT engines in %s attached (self-check relative error: encoder %.4f, UNet %.4f)",
        engine_dir,
        encoder_error,
        unet_error,
    )
    return encoder, unet


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True, help="Hub repo id or local checkpoint directory")
    parser.add_argument("--engine-dir", required=True, type=Path, help="Directory to write both engines to")
    parser.add_argument("--fp32", action="store_true", help="Build fp32 engines instead of fp16")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    policy = DiffusionPolicy.from_pretrained(args.policy_path).to("cuda").eval()
    logger.info("Exporting and building %s engines (takes a few minutes)", "fp32" if args.fp32 else "fp16")
    build_engines(policy, args.engine_dir, fp16=not args.fp32)
    attach_tensorrt_engines(policy, args.engine_dir)
    print(f"Engines written to {args.engine_dir}")


if __name__ == "__main__":
    main()
