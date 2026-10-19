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

- ``frame_encoder.engine``: encodes one observation frame (image backbone + spatial softmax over
  every camera, concatenated with the state). A prediction's conditioning is the concatenation of
  its ``n_obs_steps`` frames.
- ``unet.engine``: one denoising step of the conditional UNet, run ``num_inference_steps`` times.

At run time, on top of the engines:

- Each frame is encoded once and cached. The tick before a prediction encodes its newest frame,
  so the prediction tick itself only encodes the current frame (half the encoder work with
  ``n_obs_steps=2``).
- The denoising loop (UNet engine + noise scheduler) is captured in a CUDA graph, removing the
  per-step Python and launch overhead. It falls back to the eager loop if capture fails.

The scheduler stays in PyTorch, so the sampler (DDPM/DDIM) and the number of steps can be changed
without rebuilding the engines (the graph is captured for the configured values when the engines
are attached). Observation and action queues, pre/post-processing are untouched.

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

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.tensorrt import TensorRTEngine, build_engine, export_onnx

from .configuration_diffusion import DiffusionConfig
from .modeling_diffusion import DiffusionModel, DiffusionPolicy

logger = logging.getLogger(__name__)

FRAME_ENCODER_ENGINE = "frame_encoder.engine"
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


def _example_batch(
    config: DiffusionConfig, device: torch.device | str, n_frames: int | None = None
) -> dict[str, Tensor]:
    """Random normalized observations, stacked the way ``generate_actions`` receives them."""
    s = config.n_obs_steps if n_frames is None else n_frames
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


class _FrameEncoder:
    """Encode observation frames with the engine, caching each frame's encoding.

    A frame is identified by the tensor objects the policy keeps in its observation queues, which
    hold the same objects for as long as the frame is in the history. The cache keeps references
    to those tensors, so their ids cannot be reused while cached.
    """

    def __init__(self, engine: TensorRTEngine, input_names: list[str]):
        self.engine = engine
        self.keys = [_ENCODER_KEYS[name] for name in input_names]
        self._cache: dict[tuple[int, ...], tuple[tuple[Tensor, ...], Tensor]] = {}

    def __call__(self, frame: dict[str, Tensor]) -> Tensor:
        """Encode one frame: each value has a batch dim of 1 and no time dim. Returns (1, D)."""
        return self.engine(*[frame[key].unsqueeze(1) for key in self.keys])

    def cached(self, frame: dict[str, Tensor]) -> Tensor:
        key = tuple(id(frame[k]) for k in self.keys)
        if key not in self._cache:
            self._cache[key] = (tuple(frame[k] for k in self.keys), self(frame))
        return self._cache[key][1]

    def retain(self, frames: list[dict[str, Tensor]]) -> None:
        """Drop cached encodings of frames that have left the observation history."""
        live = {tuple(id(f[k]) for k in self.keys) for f in frames}
        self._cache = {k: v for k, v in self._cache.items() if k in live}

    def stacked(self, batch: dict[str, Tensor]) -> Tensor:
        """Uncached ``_prepare_global_conditioning`` for a (1, n_obs_steps, ...) batch."""
        n_frames = batch[self.keys[0]].shape[1]
        frames = [{k: batch[k][:, i] for k in self.keys} for i in range(n_frames)]
        return torch.cat([self(f) for f in frames], dim=-1)


class _GraphedSampler:
    """CUDA graph of ``DiffusionModel.conditional_sample`` for a batch size of one.

    Captured for the scheduler and step count configured at capture time. Calls the eager loop for
    anything the graph does not cover (other batch sizes, an explicit generator).
    """

    def __init__(self, model: DiffusionModel, global_cond_dim: int):
        self.eager = model.conditional_sample
        config = model.config
        device = next(model.parameters()).device
        self.global_cond = torch.zeros(1, global_cond_dim, device=device)
        self.noise = torch.zeros(1, config.horizon, config.action_feature.shape[0], device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.eager(1, global_cond=self.global_cond, noise=self.noise)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.eager(1, global_cond=self.global_cond, noise=self.noise)

    def __call__(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        if batch_size != 1 or generator is not None or global_cond is None:
            return self.eager(batch_size, global_cond=global_cond, generator=generator, noise=noise)
        self.global_cond.copy_(global_cond)
        if noise is None:
            self.noise.normal_()  # fresh prior for every prediction, as the eager loop draws
        else:
            self.noise.copy_(noise)
        self.graph.replay()
        return self.output.clone()


def export_onnx_files(policy: DiffusionPolicy, out_dir: str | Path) -> tuple[Path, Path]:
    """Export the per-frame encoder and one UNet step with a batch size of one. Needs ``onnx``."""
    out_dir = Path(out_dir)
    model = copy.deepcopy(policy.diffusion).cpu().eval()
    names = encoder_input_names(policy.config)
    frame = _example_batch(policy.config, "cpu", n_frames=1)
    with torch.no_grad():
        frame_cond = model._prepare_global_conditioning(frame)
    encoder_onnx = export_onnx(
        _EncoderModule(model, names),
        [frame[_ENCODER_KEYS[name]] for name in names],
        out_dir / "frame_encoder.onnx",
        names,
        ["frame_cond"],
    )
    global_cond = frame_cond.repeat(1, policy.config.n_obs_steps)
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
        build_engine(encoder_onnx, engine_dir / FRAME_ENCODER_ENGINE, fp16=fp16)
        build_engine(unet_onnx, engine_dir / UNET_ENGINE, fp16=fp16)
    return engine_dir


def _relative_error(actual: Tensor, reference: Tensor) -> float:
    return ((actual - reference).abs().max() / reference.abs().max().clamp_min(1.0)).item()


@torch.no_grad()
def attach_tensorrt_engines(
    policy: DiffusionPolicy,
    engine_dir: str | Path,
    rtol: float = DEFAULT_SELF_CHECK_RTOL,
    cuda_graph: bool = True,
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
    config = policy.config
    device = next(policy.parameters()).device
    names = encoder_input_names(config)
    engine_dir = Path(engine_dir)
    encoder_path = engine_dir / FRAME_ENCODER_ENGINE
    if not encoder_path.is_file():
        raise FileNotFoundError(
            f"{encoder_path} not found. Engine directories from before per-frame encoding hold "
            "rgb_encoder.engine instead; rebuild the engines."
        )
    frame_engine = TensorRTEngine(encoder_path, names, ["frame_cond"], device=device)
    unet_engine = TensorRTEngine(engine_dir / UNET_ENGINE, UNET_INPUTS, ["eps"], device=device)
    encoder = _FrameEncoder(frame_engine, names)

    # Check each engine on its own: a full sampling run would not be comparable under DDPM,
    # which draws fresh noise at every step.
    batch = _example_batch(config, device)
    reference_cond = model._prepare_global_conditioning(batch)
    encoder_error = _relative_error(encoder.stacked(batch), reference_cond)
    unet_inputs = _example_unet_inputs(config, reference_cond)
    unet_error = _relative_error(
        unet_engine(*unet_inputs), model.unet(*unet_inputs[:2], global_cond=unet_inputs[2])
    )
    for part, error in (("encoder", encoder_error), ("UNet", unet_error)):
        if error > rtol:
            raise RuntimeError(
                f"TensorRT {part} engine in {engine_dir} disagrees with the PyTorch policy (relative "
                f"error {error:.4f} > {rtol}). It was probably built from another checkpoint; rebuild it."
            )

    model._prepare_global_conditioning = encoder.stacked
    model.unet = _TensorRTUNet(unet_engine)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if cuda_graph and device.type == "cuda":
        try:
            model.conditional_sample = _GraphedSampler(model, reference_cond.shape[-1])
        except Exception as e:  # noqa: BLE001 - an eager loop is slower but still correct
            logger.warning("CUDA graph capture of the denoising loop failed, running it eagerly: %s", e)

    obs_keys = encoder.keys
    n_obs, n_action = config.n_obs_steps, config.n_action_steps

    def predict_action_chunk(batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        frames = [{k: policy._queues[k][i] for k in obs_keys} for i in range(n_obs)]
        global_cond = torch.cat([encoder.cached(f) for f in frames], dim=-1)
        encoder.retain(frames)
        actions = model.conditional_sample(1, global_cond=global_cond, noise=noise)
        return actions[:, n_obs - 1 : n_obs - 1 + n_action]

    select_action = policy.select_action

    def select_action_with_prefetch(batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        action = select_action(batch, noise=noise)
        # If the next tick predicts, this tick's frame will be in its history: encode it now, on
        # a tick that has time to spare, instead of on the prediction tick.
        if len(policy._queues[ACTION]) < n_obs - 1:
            encoder.cached({k: policy._queues[k][-1] for k in obs_keys})
        return action

    policy.predict_action_chunk = predict_action_chunk
    policy.select_action = select_action_with_prefetch
    logger.info(
        "TensorRT engines in %s attached (self-check relative error: encoder %.4f, UNet %.4f; "
        "denoising loop %s)",
        engine_dir,
        encoder_error,
        unet_error,
        "in a CUDA graph" if isinstance(model.conditional_sample, _GraphedSampler) else "eager",
    )
    return frame_engine, unet_engine


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
