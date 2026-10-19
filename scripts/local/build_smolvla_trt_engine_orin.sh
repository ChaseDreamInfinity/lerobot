#!/usr/bin/env bash
# Build TensorRT engines for a SmolVLA checkpoint on the Jetson Orin.
#
# Three engines, written to one directory: image encoder (all cameras in one batch), prefix KV
# fill (strongly typed fp16, so the model's fp32 upcasts survive; plain fp16 overflows), and one
# denoising step. On AGX Orin a prediction takes ~60 ms instead of ~115 ms with --cuda-graph
# (~855 ms eager). Engines only load on the GPU model and TensorRT version that built them, and
# for the camera count they were built for, so they are tagged "orin" and live under outputs/
# (git-ignored). Rebuild after a JetPack upgrade, a new checkpoint, or a camera change.
#
# Usage: bash scripts/local/build_smolvla_trt_engine_orin.sh
#   POLICY_PATH=chasedreaminf/smolvla_so101-other NUM_CAMERAS=2 bash scripts/local/build_smolvla_trt_engine_orin.sh
#
# Then run: bash scripts/local/eval_pick_block_smolvla_orin.sh --trt
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/smolvla_so101-pick-color-block-in-cup-2cam-run2-v3}"
NUM_CAMERAS="${NUM_CAMERAS:-2}"
ENGINE_DIR="${ENGINE_DIR:-outputs/trt_engines/$(basename -- "${POLICY_PATH}")_orin_fp16}"

# shellcheck source=scripts/local/_cuda_guard.sh
source scripts/local/_cuda_guard.sh
require_cuda cuda

# ONNX export needs the onnx package, which is not a project dependency. Layer it on for this one
# command instead of installing it: `uv sync` would swap the Jetson CUDA torch for a CPU wheel.
# numpy is pinned to the environment's own version, because the Jetson torch build is compiled
# against numpy 1.x and an unpinned overlay pulls in 2.x.
numpy_version="$(uv run --no-sync python -c 'import numpy; print(numpy.__version__)')"

printf 'Policy: %s\nCameras: %s\nEngine directory: %s\n' "${POLICY_PATH}" "${NUM_CAMERAS}" "${ENGINE_DIR}"
uv run --no-sync --with onnx --with "numpy==${numpy_version}" \
  python -m lerobot.policies.smolvla.tensorrt_smolvla \
  --policy-path="${POLICY_PATH}" --engine-dir="${ENGINE_DIR}" --num-cameras="${NUM_CAMERAS}"
