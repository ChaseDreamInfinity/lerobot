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
import importlib
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_act import ACTConfig
from .modeling_act import ACTPolicy

logger = logging.getLogger(__name__)

OUTPUT_NAME = "actions"
# Max |difference| from the PyTorch model, in normalized action units, tolerated by the load-time
# self-check. fp16 engines measure ~0.01 on SO-101 checkpoints; a mismatched checkpoint is O(1).
DEFAULT_SELF_CHECK_ATOL = 0.05


def import_tensorrt():
    """Import ``tensorrt``, falling back to the bindings JetPack installs for the system Python.

    JetPack ships TensorRT as a Debian package under ``/usr/lib/python3.X/dist-packages``, which a
    virtualenv does not see. Only ``tensorrt`` is imported from there: the directory is removed
    from ``sys.path`` again so no other system package can shadow the environment's own.
    """
    try:
        return importlib.import_module("tensorrt")
    except ImportError:
        pass
    system_dir = f"/usr/lib/python3.{sys.version_info.minor}/dist-packages"
    if not (Path(system_dir) / "tensorrt").is_dir():
        raise ImportError(
            "TensorRT Python bindings not found. On Jetson they come with JetPack "
            "(python3-libnvinfer); elsewhere install the `tensorrt` package."
        )
    sys.path.append(system_dir)
    try:
        return importlib.import_module("tensorrt")
    finally:
        sys.path.remove(system_dir)


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


def export_onnx(policy: ACTPolicy, onnx_path: str | Path, opset: int = 17) -> Path:
    """Export ``ACT.forward`` with a fixed batch size of one. Needs the ``onnx`` package."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Export from a CPU copy: constant folding during export trips over mixed devices otherwise,
    # and the caller's policy stays where it is.
    module = _ACTChunkModule(copy.deepcopy(policy).cpu().eval()).eval()
    inputs = tuple(_example_inputs(policy.config, "cpu"))
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(onnx_path),
            input_names=module.input_keys,
            output_names=[OUTPUT_NAME],
            opset_version=opset,
            dynamo=False,
        )
    return onnx_path


def build_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True) -> Path:
    """Build a serialized TensorRT engine from an ONNX file."""
    trt = import_tensorrt()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    build_config = builder.create_builder_config()
    if fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return engine_path


class TensorRTActionChunker:
    """Run a serialized ACT engine on CUDA tensors, as a drop-in for ``predict_action_chunk``."""

    def __init__(self, engine_path: str | Path, config: ACTConfig, device: torch.device | str = "cuda"):
        trt = import_tensorrt()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRT engines run on CUDA, got device={self.device}")
        self.engine_path = Path(engine_path)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Could not load TensorRT engine {self.engine_path}. Engines only load on the GPU "
                f"model and TensorRT version ({trt.__version__}) that built them; rebuild it here."
            )
        self.context = self.engine.create_execution_context()

        self.input_keys = engine_input_keys(config)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        if sorted(names) != sorted([*self.input_keys, OUTPUT_NAME]):
            raise ValueError(
                f"Engine {self.engine_path} has tensors {names}, but the policy expects inputs "
                f"{self.input_keys} and output '{OUTPUT_NAME}'. Was it built from another checkpoint?"
            )
        self.input_shapes = {key: tuple(self.engine.get_tensor_shape(key)) for key in self.input_keys}
        out_shape = tuple(self.engine.get_tensor_shape(OUTPUT_NAME))
        self.output = torch.empty(out_shape, dtype=torch.float32, device=self.device)
        self.context.set_tensor_address(OUTPUT_NAME, self.output.data_ptr())
        # TensorRT adds extra synchronization when enqueued on the default stream.
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if kwargs:
            raise NotImplementedError(f"TensorRT ACT engine does not support {sorted(kwargs)}")
        inputs = []  # keep references alive until the engine has run
        for key in self.input_keys:
            tensor = batch[key]
            if tuple(tensor.shape) != self.input_shapes[key]:
                raise ValueError(
                    f"'{key}' has shape {tuple(tensor.shape)} but the engine was built for "
                    f"{self.input_shapes[key]}. Match the camera resolution to the checkpoint, "
                    "or rebuild the engine."
                )
            tensor = tensor.to(device=self.device, dtype=torch.float32).contiguous()
            self.context.set_tensor_address(key, tensor.data_ptr())
            inputs.append(tensor)
        # Order the engine after the input copies and everything after it, so no host sync is needed.
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")
        current.wait_stream(self.stream)
        # The output buffer is reused by every call, while select_action keeps views of the chunk
        # in its action queue: hand back a copy.
        return self.output.clone()


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
    parser.add_argument("--onnx", type=Path, help="Where to keep the ONNX file (default: next to the engine)")
    parser.add_argument("--fp32", action="store_true", help="Build an fp32 engine instead of fp16")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    policy = ACTPolicy.from_pretrained(args.policy_path).to("cuda").eval()
    onnx_path = args.onnx or args.engine.with_suffix(".onnx")
    logger.info("Exporting ONNX to %s", onnx_path)
    export_onnx(policy, onnx_path)
    logger.info("Building %s TensorRT engine (takes a few minutes)", "fp32" if args.fp32 else "fp16")
    build_engine(onnx_path, args.engine, fp16=not args.fp32)
    attach_tensorrt_engine(policy, args.engine)
    print(f"Engine written to {args.engine}")


if __name__ == "__main__":
    main()
