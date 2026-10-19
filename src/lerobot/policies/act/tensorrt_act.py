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
"""TensorRT inference for ACT.

ACT inference is launch-bound on small GPUs such as Jetson: the eager forward pass spends most of
its time in the CPU dispatching kernels, so mixed precision alone does not help. A TensorRT engine
runs the whole forward as one fused graph.

Only ``ACT.forward`` (normalized observations -> normalized action chunk) goes into the engine.
Pre/post-processing, the action queue and temporal ensembling stay in PyTorch, so the policy
behaves exactly as before apart from numerical precision.

Build an engine once per checkpoint and device. Engines are tied to the GPU model and the TensorRT
version that built them, and to the input resolution in the checkpoint config::

    uv run --no-sync --with onnx python -m lerobot.policies.act.tensorrt_act \\
        --policy-path=<repo_or_dir> --engine=act_fp16.engine

Then attach it at inference time with ``attach_tensorrt_engine(policy, "act_fp16.engine")``
(``lerobot-rollout --tensorrt_engine=...`` does this).
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
from lerobot.utils.tensorrt import TensorRTEngine, build_engine, export_onnx as export_module_onnx

from .configuration_act import ACTConfig
from .modeling_act import ACTPolicy

logger = logging.getLogger(__name__)

OUTPUT_NAME = "actions"
# Max |difference| from the PyTorch model, in normalized action units, tolerated by the load-time
# self-check. fp16 engines measure ~0.01 on SO-101 checkpoints; a mismatched checkpoint is O(1).
DEFAULT_SELF_CHECK_ATOL = 0.05


def engine_input_keys(config: ACTConfig) -> list[str]:
    """Batch keys fed to the engine, in binding order."""
    keys = []
    if config.robot_state_feature:
        keys.append(OBS_STATE)
    if config.env_state_feature:
        keys.append(OBS_ENV_STATE)
    keys.extend(config.image_features)
    return keys


class _ACTChunkModule(nn.Module):
    """Positional-argument wrapper around ``ACT.forward`` for ONNX export."""

    def __init__(self, policy: ACTPolicy):
        super().__init__()
        self.model = policy.model
        self.config = policy.config
        self.input_keys = engine_input_keys(policy.config)

    def forward(self, *inputs: Tensor) -> Tensor:
        batch = dict(zip(self.input_keys, inputs, strict=True))
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model(batch)[0]


def _example_inputs(config: ACTConfig, device: torch.device | str) -> list[Tensor]:
    """Random normalized inputs shaped like a batch of one from the checkpoint's features."""
    features = config.input_features
    return [torch.randn(1, *features[key].shape, device=device) for key in engine_input_keys(config)]


def export_onnx(policy: ACTPolicy, onnx_path: str | Path) -> Path:
    """Export ``ACT.forward`` with a fixed batch size of one. Needs the ``onnx`` package."""
    # Export a CPU copy, so the caller's policy stays where it is.
    module = _ACTChunkModule(copy.deepcopy(policy).cpu().eval())
    return export_module_onnx(
        module, _example_inputs(policy.config, "cpu"), onnx_path, module.input_keys, [OUTPUT_NAME]
    )


class TensorRTActionChunker:
    """Run a serialized ACT engine as a drop-in for ``predict_action_chunk``."""

    def __init__(self, engine_path: str | Path, config: ACTConfig, device: torch.device | str = "cuda"):
        self.input_keys = engine_input_keys(config)
        self.engine = TensorRTEngine(engine_path, self.input_keys, [OUTPUT_NAME], device=device)
        self.device = self.engine.device

    def __call__(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if kwargs:
            raise NotImplementedError(f"TensorRT ACT engine does not support {sorted(kwargs)}")
        return self.engine(*[batch[key] for key in self.input_keys])


@torch.no_grad()
def self_check(policy: ACTPolicy, chunker: TensorRTActionChunker, trials: int = 3) -> float:
    """Max |difference| between the engine and the PyTorch model on random normalized inputs."""
    max_diff = 0.0
    for _ in range(trials):
        batch = dict(zip(chunker.input_keys, _example_inputs(policy.config, chunker.device), strict=True))
        reference = policy.predict_action_chunk(batch)
        max_diff = max(max_diff, (chunker(batch) - reference).abs().max().item())
    return max_diff


def attach_tensorrt_engine(
    policy: ACTPolicy, engine_path: str | Path, atol: float = DEFAULT_SELF_CHECK_ATOL
) -> TensorRTActionChunker:
    """Route ``policy.predict_action_chunk`` through a TensorRT engine.

    The engine is checked against the PyTorch model first, so an engine built from a different
    checkpoint, or a broken build, fails here instead of driving the robot.
    """
    if policy.name != "act":
        raise ValueError(f"TensorRT engines are only supported for ACT, got policy type '{policy.name}'")
    policy.eval()
    chunker = TensorRTActionChunker(engine_path, policy.config, device=next(policy.parameters()).device)
    max_diff = self_check(policy, chunker)
    if max_diff > atol:
        raise RuntimeError(
            f"TensorRT engine {engine_path} disagrees with the PyTorch policy (max |diff| "
            f"{max_diff:.4f} > {atol} in normalized action units). It was probably built from "
            "another checkpoint; rebuild it."
        )
    logger.info("TensorRT engine %s attached (self-check max |diff| %.4f)", engine_path, max_diff)
    policy.predict_action_chunk = chunker
    return chunker


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True, help="Hub repo id or local checkpoint directory")
    parser.add_argument("--engine", required=True, type=Path, help="Where to write the TensorRT engine")
    parser.add_argument("--onnx", type=Path, help="Keep the ONNX file here (default: not kept)")
    parser.add_argument("--fp32", action="store_true", help="Build an fp32 engine instead of fp16")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    policy = ACTPolicy.from_pretrained(args.policy_path).to("cuda").eval()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = args.onnx or Path(tmp) / "act.onnx"
        logger.info("Exporting ONNX to %s", onnx_path)
        export_onnx(policy, onnx_path)
        logger.info("Building %s TensorRT engine (takes a few minutes)", "fp32" if args.fp32 else "fp16")
        build_engine(onnx_path, args.engine, fp16=not args.fp32)
    attach_tensorrt_engine(policy, args.engine)
    print(f"Engine written to {args.engine}")


if __name__ == "__main__":
    main()
