#!/usr/bin/env python
"""Speed probe: does quantizing the SmolVLA step engine's weights help on the Jetson Orin?

Rewrites the fp16 step ONNX (from lerobot.policies.smolvla.tensorrt_smolvla.export_onnx_files) so
the projection MatMuls read quantized weights, for timing with trtexec. Modes:
- int8-woq: per-output-channel INT8 weights, dequantized before the MatMul (weight-only);
- w8a8: the same weights, plus per-tensor Q/DQ on the MatMul input, so the GEMM runs in INT8;
- int4-woq: block-wise (128) INT4 weights, weight-only (opset 21).
Activation scales are placeholders: these files are for timing only, not for running a policy.
Result on AGX Orin / TensorRT 10.3 (none is faster than fp16) is in SMOLVLA_ORIN_OPTIMIZATION.md.

    uv run --no-sync --with onnx --with numpy==1.26.4 python scripts/local/smolvla_step_quant_probe.py \\
        step.onnx step_w8a8.onnx --mode=w8a8 [--only=gate_proj,up_proj]
    /usr/src/tensorrt/bin/trtexec --onnx=step_w8a8.onnx --fp16 --int8 --useCudaGraph
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def quantize(src: str, dst: str, mode: str, only: list[str] | None) -> int:
    model = onnx.load(src)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    nodes, replaced = [], set()
    for node in graph.node:
        weight = node.input[1] if node.op_type == "MatMul" else None
        if (
            weight not in inits
            or inits[weight].data_type != TensorProto.FLOAT16
            or len(inits[weight].dims) != 2
            or (only and not any(name in node.name for name in only))
        ):
            nodes.append(node)
            continue
        w = numpy_helper.to_array(inits[weight]).astype(np.float32)  # (in, out)
        if mode == "int4-woq":
            block = 128 if w.shape[0] % 128 == 0 else w.shape[0]
            blocks = w.reshape(w.shape[0] // block, block, w.shape[1])
            scale = (np.maximum(np.abs(blocks).max(axis=1), 1e-8) / 7.0).astype(np.float16)
            q = np.clip(np.round(blocks / scale[:, None, :].astype(np.float32)), -8, 7).reshape(w.shape)
            q_init = helper.make_tensor(f"{weight}_q", TensorProto.INT4, w.shape, q.astype(np.int8).flatten())
            dq_attrs = {"axis": 0, "block_size": block}
        else:
            scale = (np.maximum(np.abs(w).max(axis=0), 1e-8) / 127.0).astype(np.float16)
            q = np.clip(np.round(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
            q_init = numpy_helper.from_array(q, f"{weight}_q")
            dq_attrs = {"axis": 1}
        graph.initializer.extend([q_init, numpy_helper.from_array(scale, f"{weight}_s")])
        nodes.append(
            helper.make_node("DequantizeLinear", [f"{weight}_q", f"{weight}_s"], [f"{weight}_dq"], **dq_attrs)
        )
        if mode == "w8a8":  # per-tensor input Q/DQ; q/k/v share an input, so names are per node
            act, prefix = node.input[0], node.name
            graph.initializer.extend(
                [
                    numpy_helper.from_array(np.array(8.0 / 127.0, np.float16), f"{prefix}_as"),
                    numpy_helper.from_array(np.array(0, np.int8), f"{prefix}_az"),
                ]
            )
            qdq = [f"{prefix}_as", f"{prefix}_az"]
            nodes.append(helper.make_node("QuantizeLinear", [act, *qdq], [f"{prefix}_aq"]))
            nodes.append(helper.make_node("DequantizeLinear", [f"{prefix}_aq", *qdq], [f"{prefix}_adq"]))
            node.input[0] = f"{prefix}_adq"
        node.input[1] = f"{weight}_dq"
        replaced.add(weight)
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    kept = [i for i in graph.initializer if i.name not in replaced]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for opset in model.opset_import:  # fp16 scales need opset 19, INT4 and block_size opset 21
        if opset.domain in ("", "ai.onnx"):
            opset.version = 21 if mode == "int4-woq" else 19
    onnx.save(model, dst)
    return len(replaced)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--mode", choices=["int8-woq", "w8a8", "int4-woq"], required=True)
    parser.add_argument("--only", help="Comma-separated MatMul name filters, e.g. gate_proj,up_proj")
    args = parser.parse_args()
    n = quantize(args.src, args.dst, args.mode, args.only.split(",") if args.only else None)
    print(f"{args.mode}: quantized {n} MatMul weights -> {args.dst}")
