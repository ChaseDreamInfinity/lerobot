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
"""Shared TensorRT helpers: import, ONNX export, engine build, and a torch-tensor engine runner.

Policy-specific wrappers (what goes into an engine, and how it is attached to a policy) live next to
each policy, e.g. ``lerobot.policies.act.tensorrt_act``.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn


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


def export_onnx(
    module: nn.Module,
    example_inputs: Sequence[Tensor],
    onnx_path: str | Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset: int = 17,
) -> Path:
    """Export ``module`` with fixed input shapes. Needs the ``onnx`` package.

    Export from a CPU module and CPU inputs: constant folding during export trips over mixed
    devices otherwise.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module.eval(),
            tuple(example_inputs),
            str(onnx_path),
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=opset,
            dynamo=False,
        )
    return onnx_path


def build_engine(
    onnx_path: str | Path, engine_path: str | Path, fp16: bool = True, strongly_typed: bool = False
) -> Path:
    """Build a serialized TensorRT engine from an ONNX file.

    ``fp16`` lets TensorRT pick fp16 for any layer. ``strongly_typed`` instead makes it keep the
    ONNX file's own types exactly: export from an fp16 model to get fp16 compute with the model's
    explicit fp32 upcasts (e.g. in normalization and attention) preserved.
    """
    trt = import_tensorrt()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    build_config = builder.create_builder_config()
    if fp16 and not strongly_typed:
        build_config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return engine_path


class TensorRTEngine:
    """Run a serialized engine on CUDA tensors with fixed shapes.

    Inputs are passed positionally in ``input_names`` order and checked against the engine's
    shapes; outputs come back as fresh tensors (one, or a tuple in ``output_names`` order).
    """

    def __init__(
        self,
        engine_path: str | Path,
        input_names: Sequence[str],
        output_names: Sequence[str],
        device: torch.device | str = "cuda",
    ):
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

        self.input_names = list(input_names)
        self.output_names = list(output_names)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        if sorted(names) != sorted([*self.input_names, *self.output_names]):
            raise ValueError(
                f"Engine {self.engine_path} has tensors {names}, but expected inputs "
                f"{self.input_names} and outputs {self.output_names}. Was it built from another model?"
            )
        dtypes = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        self.shapes = {name: tuple(self.engine.get_tensor_shape(name)) for name in names}
        self.dtypes = {name: dtypes[self.engine.get_tensor_dtype(name)] for name in names}
        self.outputs = [
            torch.empty(self.shapes[name], dtype=self.dtypes[name], device=self.device)
            for name in self.output_names
        ]
        for name, buffer in zip(self.output_names, self.outputs, strict=True):
            self.context.set_tensor_address(name, buffer.data_ptr())
        # TensorRT adds extra synchronization when enqueued on the default stream.
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        if len(inputs) != len(self.input_names):
            raise ValueError(f"Expected inputs {self.input_names}, got {len(inputs)} tensors")
        keep_alive = []  # converted copies must outlive the enqueue
        for name, tensor in zip(self.input_names, inputs, strict=True):
            if tuple(tensor.shape) != self.shapes[name]:
                raise ValueError(
                    f"'{name}' has shape {tuple(tensor.shape)} but the engine was built for "
                    f"{self.shapes[name]}. Match the camera resolution to the checkpoint, "
                    "or rebuild the engine."
                )
            tensor = tensor.to(device=self.device, dtype=self.dtypes[name]).contiguous()
            self.context.set_tensor_address(name, tensor.data_ptr())
            keep_alive.append(tensor)
        # Order the engine after the input copies and everything after it, so no host sync is needed.
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")
        current.wait_stream(self.stream)
        # Output buffers are reused by every call, and callers may keep results (e.g. in an
        # action queue): hand back copies.
        outputs = tuple(buffer.clone() for buffer in self.outputs)
        return outputs[0] if len(outputs) == 1 else outputs
