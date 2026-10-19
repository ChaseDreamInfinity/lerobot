#!/usr/bin/env bash
# Build a TensorRT engine for an ACT checkpoint on the Jetson Orin.
#
# The engine replaces ACT's forward pass in lerobot-rollout (fp16: ~5 ms vs ~37 ms eager on
# AGX Orin). It only loads on the GPU model and TensorRT version that built it, so it is tagged
# "orin" and lives under outputs/ (git-ignored). Rebuild after a JetPack upgrade or a new checkpoint.
#
# Usage: bash scripts/local/build_act_trt_engine_orin.sh
#   POLICY_PATH=chasedreaminf/act_so101-2cam-run2 bash scripts/local/build_act_trt_engine_orin.sh
#   PRECISION=fp32 bash scripts/local/build_act_trt_engine_orin.sh
#
# Then run: TRT_ENGINE=<printed path> bash scripts/local/eval_pick_block_act_orin.sh
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/act_so101-pick-block-in-cup-2cam-run1}"
PRECISION="${PRECISION:-fp16}"
ENGINE="${ENGINE:-outputs/trt_engines/$(basename -- "${POLICY_PATH}")_orin_${PRECISION}.engine}"

case "${PRECISION}" in
  fp16) precision_args=() ;;
  fp32) precision_args=(--fp32) ;;
  *) echo "PRECISION must be fp16 or fp32, got: ${PRECISION}" >&2; exit 2 ;;
esac

# shellcheck source=scripts/local/_cuda_guard.sh
source scripts/local/_cuda_guard.sh
require_cuda cuda

# ONNX export needs the onnx package, which is not a project dependency. Layer it on for this one
# command instead of installing it: `uv sync` would swap the Jetson CUDA torch for a CPU wheel.
# numpy is pinned to the environment's own version, because the Jetson torch build is compiled
# against numpy 1.x and an unpinned overlay pulls in 2.x.
numpy_version="$(uv run --no-sync python -c 'import numpy; print(numpy.__version__)')"

printf 'Policy: %s\nPrecision: %s\nEngine: %s\n' "${POLICY_PATH}" "${PRECISION}" "${ENGINE}"
uv run --no-sync --with onnx --with "numpy==${numpy_version}" \
  python -m lerobot.policies.act.tensorrt_act \
  --policy-path="${POLICY_PATH}" --engine="${ENGINE}" "${precision_args[@]}"
