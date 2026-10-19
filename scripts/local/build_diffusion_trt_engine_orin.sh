#!/usr/bin/env bash
# Build TensorRT engines for a Diffusion Policy checkpoint on the Jetson Orin.
#
# Two engines, written to one directory: the per-frame observation encoder and one UNet
# denoising step. fp16 on AGX Orin: a DDIM/5 prediction takes ~23 ms instead of ~244 ms eager.
# The sampler and step count stay adjustable at run time. Engines only load on the GPU model and TensorRT
# version that built them, so they are tagged "orin" and live under outputs/ (git-ignored).
# Rebuild after a JetPack upgrade, a new checkpoint, or a camera resolution change.
#
# Usage: bash scripts/local/build_diffusion_trt_engine_orin.sh
#   POLICY_PATH=chasedreaminf/diffusion_so101-other bash scripts/local/build_diffusion_trt_engine_orin.sh
#   PRECISION=fp32 bash scripts/local/build_diffusion_trt_engine_orin.sh
#
# Then run: bash scripts/local/eval_pick_block_diffusion_orin.sh --trt
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/diffusion_so101-pick-block-in-cup-2cam-run1}"
PRECISION="${PRECISION:-fp16}"
ENGINE_DIR="${ENGINE_DIR:-outputs/trt_engines/$(basename -- "${POLICY_PATH}")_orin_${PRECISION}}"

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

printf 'Policy: %s\nPrecision: %s\nEngine directory: %s\n' "${POLICY_PATH}" "${PRECISION}" "${ENGINE_DIR}"
uv run --no-sync --with onnx --with "numpy==${numpy_version}" \
  python -m lerobot.policies.diffusion.tensorrt_diffusion \
  --policy-path="${POLICY_PATH}" --engine-dir="${ENGINE_DIR}" "${precision_args[@]}"
