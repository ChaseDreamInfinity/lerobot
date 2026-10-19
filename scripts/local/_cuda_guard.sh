#!/usr/bin/env bash
# Shared CUDA preflight for the eval_pick_block_*_orin.sh scripts.
#
# Source it *after* the cd into the repo root, then call:
#   require_cuda "${DEVICE}"
#
# Why this is more than a torch.cuda.is_available() check: the usual failure on
# this box is silent. `uv sync` re-resolves torch from pypi.org (uv.lock gates
# the CUDA deps behind platform_machine == 'x86_64'), so the jetson-ai-lab CUDA
# wheel gets replaced by the generic aarch64 CPU build. torch still imports and
# every script still runs -- it just never touches the GPU. That case needs a
# different fix than "the driver is unhappy", so name the two apart.
#
# Both wheels share the same recorded version (2.8.0), so recovery must force
# the swap with --reinstall-package; a plain pin looks already-satisfied.

JETSON_TORCH_INDEX="${JETSON_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
JETSON_TORCH_VERSION="${JETSON_TORCH_VERSION:-2.8.0}"
JETSON_TORCHVISION_VERSION="${JETSON_TORCHVISION_VERSION:-0.23.0}"

require_cuda() {
  local device="${1:?require_cuda: DEVICE argument is required}"
  [[ "${device}" == cuda* ]] || return 0

  local probe
  if ! probe="$(uv run --no-sync python -c 'import torch; print(torch.__version__, torch.version.cuda or "none", torch.cuda.is_available())' 2>&1)"; then
    echo "DEVICE=${device} was requested, but importing torch failed:" >&2
    echo "${probe}" >&2
    echo >&2
    echo "Install dependencies first:  uv sync --locked" >&2
    echo "then restore the Jetson CUDA wheels (see below)." >&2
    _print_jetson_torch_recovery
    return 1
  fi

  # Ignore anything torch logged before the result; the answer is the last line.
  local torch_version cuda_build cuda_available
  read -r torch_version cuda_build cuda_available <<<"${probe##*$'\n'}"

  [[ "${cuda_available}" == "True" ]] && return 0

  if [[ "${cuda_build}" == "none" ]]; then
    echo "DEVICE=${device} was requested, but the installed PyTorch is a CPU-only build." >&2
    echo "  torch ${torch_version} -- built without CUDA (torch.version.cuda is None)." >&2
    echo >&2
    echo "This is not a driver problem. The GPU is almost certainly fine; check with: nvidia-smi" >&2
    echo "A bare 'uv sync' replaces the Jetson CUDA wheel with the PyPI CPU wheel." >&2
    echo "Use 'uv sync --no-install-package torch --no-install-package torchvision'," >&2
    echo "or re-run the recovery below afterwards." >&2
    _print_jetson_torch_recovery
    return 1
  fi

  echo "DEVICE=${device} was requested, but PyTorch cannot reach the GPU." >&2
  echo "  torch ${torch_version} -- built against CUDA ${cuda_build}, but torch.cuda.is_available() is False." >&2
  echo >&2
  echo "The wheel is right, so this is a runtime/driver issue. Check, in order:" >&2
  echo "  nvidia-smi                       # does the driver respond at all?" >&2
  echo "  ls -l /dev/nvhost-gpu /dev/nvgpu # present, and readable by ${USER}?" >&2
  echo "  scripts/local/orin_max_perf.sh   # perf mode resets on reboot" >&2
  echo "  cat /etc/nv_tegra_release        # L4T major version vs. the wheel's JetPack" >&2
  return 1
}

_print_jetson_torch_recovery() {
  echo >&2
  echo "Restore the CUDA wheels for this Jetson (JetPack 6 / CUDA 12.6 / cp310):" >&2
  echo >&2
  echo "  UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} uv pip install --no-deps \\" >&2
  echo "    --reinstall-package torch --reinstall-package torchvision \\" >&2
  echo "    --index-url ${JETSON_TORCH_INDEX} \\" >&2
  echo "    torch==${JETSON_TORCH_VERSION} torchvision==${JETSON_TORCHVISION_VERSION}" >&2
  echo >&2
  echo "--reinstall-package is required, not optional: the CPU wheel records its" >&2
  echo "version as plain '${JETSON_TORCH_VERSION}' (the +cpu suffix is in torch.__version__ but" >&2
  echo "not in the dist-info METADATA), so uv sees the pin as already satisfied and" >&2
  echo "no-ops with 'Checked 2 packages'. Watch for an actual download." >&2
  echo >&2
  echo "Then verify:" >&2
  echo "  UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache} uv run --no-sync \\" >&2
  echo "    python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())'" >&2
  echo >&2
  echo "Expected: 12.6 True" >&2
  echo "If you then hit a numpy ABI error, pin it back: uv pip install 'numpy<2'" >&2
}
