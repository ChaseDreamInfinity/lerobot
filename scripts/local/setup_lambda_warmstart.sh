#!/usr/bin/env bash
# Lambda Labs warm-start setup for subsequent instances.
#
# Run this each time you spin up a new Lambda instance that has your
# persistent filesystem (set up via setup_lambda_persistent.sh) attached.
# This restores the env vars, ensures uv is on PATH, and runs a
# (typically near-instant) uv sync against the warm cache.
#
# Usage:
#   PERSISTED=/home/ubuntu/persisted source scripts/local/setup_lambda_warmstart.sh
#
# IMPORTANT: source this script (don't bash it) if you want the exports
# (UV_CACHE_DIR, HF_HOME, LD_LIBRARY_PATH, PATH) to persist in your
# current shell. Otherwise, copy the exports into your shell manually
# or run the script via `eval "$(bash <this-script>)"` pattern.

set -euo pipefail

PERSISTED="${PERSISTED:-/home/ubuntu/persisted}"
REPO_DIR="${REPO_DIR:-$PERSISTED/lerobot}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:---extra training --extra pi --extra eo1}"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: $REPO_DIR not found. Did you attach the persistent filesystem?"
    echo "If this is a fresh persistent fs, run setup_lambda_persistent.sh first."
    return 1 2>/dev/null || exit 1
fi

export UV_CACHE_DIR="$PERSISTED/.cache/uv"
export HF_HOME="$PERSISTED/.cache/huggingface"
export LD_LIBRARY_PATH="$PERSISTED/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

# Re-install uv if not on PATH (the binary lives in ~/.local/bin which is
# ephemeral on Lambda; this is a 30-sec re-install)
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, re-installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "=== Warm-start ==="
echo "Repo dir       : $REPO_DIR"
echo "UV_CACHE_DIR   : $UV_CACHE_DIR  ($(du -sh "$UV_CACHE_DIR" 2>/dev/null | cut -f1))"
echo "HF_HOME        : $HF_HOME  ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1))"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "uv             : $(uv --version)"

cd "$REPO_DIR"
echo "Pulling latest on current branch..."
git pull --ff-only || echo "  (skip pull on conflict)"

echo "Running uv sync against warm cache..."
# shellcheck disable=SC2086
uv sync --locked $LEROBOT_EXTRAS

# Quick sanity check
echo
echo "=== Sanity check ==="
uv run python -c "
import torch, torchcodec
print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}, torchcodec={torchcodec.__version__}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)')
"

# Confirm HF auth survived
if uv run hf auth whoami >/dev/null 2>&1; then
    echo "  HF: logged in as $(uv run hf auth whoami | head -1)"
else
    echo "  HF: NOT logged in. Run: uv run hf auth login"
fi

echo
echo "=== Warm-start complete ==="
echo "To train: bash scripts/local/train_pi05_pick_color_block_run2_v3.sh"
