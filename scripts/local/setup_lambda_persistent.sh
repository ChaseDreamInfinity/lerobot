#!/usr/bin/env bash
# Lambda Labs first-time setup with a persistent filesystem.
#
# Run this ONCE on a fresh Lambda instance that has a persistent filesystem
# attached. After this completes, your repo + uv cache + HF cache + ffmpeg-shared
# all live on the persistent volume and survive instance termination.
#
# Subsequent instance starts only need scripts/local/setup_lambda_warmstart.sh.
#
# Usage:
#   PERSISTED=/home/ubuntu/persisted scripts/local/setup_lambda_persistent.sh
#
# Defaults assume:
#   - persistent filesystem mounted at /home/ubuntu/persisted
#   - the lerobot fork is github.com/ChaseDreamInfinity/lerobot
#   - the branch you want is training-experiments
# Override via env vars if your setup differs.

set -euo pipefail

# === Configuration (override via env) ===
PERSISTED="${PERSISTED:-/home/ubuntu/persisted}"
REPO_URL="${REPO_URL:-https://github.com/ChaseDreamInfinity/lerobot.git}"
BRANCH="${BRANCH:-training-experiments}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:---extra training --extra pi --extra eo1}"
FFMPEG_TARBALL_URL="${FFMPEG_TARBALL_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl-shared.tar.xz}"

echo "=== Lambda persistent setup ==="
echo "PERSISTED dir : $PERSISTED"
echo "Repo          : $REPO_URL"
echo "Branch        : $BRANCH"
echo "Python        : $PYTHON_VERSION"
echo "uv extras     : $LEROBOT_EXTRAS"
echo

# Sanity check: persistent dir must exist
if [[ ! -d "$PERSISTED" ]]; then
    echo "ERROR: $PERSISTED does not exist. Mount your persistent filesystem there first."
    echo "  In Lambda Cloud UI: attach filesystem to the instance and set mount path."
    exit 1
fi

# Directories on persistent storage
REPO_DIR="$PERSISTED/lerobot"
UV_CACHE_DIR="$PERSISTED/.cache/uv"
HF_HOME="$PERSISTED/.cache/huggingface"
FFMPEG_DIR="$PERSISTED/ffmpeg-shared"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
export UV_CACHE_DIR HF_HOME

# 1. Install uv (binary lives in ~/.local/bin which is ephemeral; re-installed
#    each instance start. Cache lives in $UV_CACHE_DIR on persistent.)
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2. Clone the repo onto persistent storage
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "Cloning $REPO_URL -> $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repo already cloned at $REPO_DIR; pulling latest"
    git -C "$REPO_DIR" fetch origin
fi
git -C "$REPO_DIR" checkout "$BRANCH"
git -C "$REPO_DIR" pull origin "$BRANCH" || true

# 3. Pin python and sync deps (this populates $UV_CACHE_DIR on persistent)
cd "$REPO_DIR"
uv python pin "$PYTHON_VERSION"
echo "Running uv sync (this is the longest first-time step)..."
# shellcheck disable=SC2086
uv sync --locked $LEROBOT_EXTRAS

# 4. Install BtbN ffmpeg shared libs (torchcodec needs ffmpeg 5-8 .so files;
#    Ubuntu 22.04 only ships ffmpeg 4. Lambda images may differ; this is a
#    safe always-install.)
if [[ ! -f "$FFMPEG_DIR/lib/libavutil.so" ]]; then
    echo "Downloading BtbN ffmpeg shared libs to $FFMPEG_DIR"
    mkdir -p "$FFMPEG_DIR"
    curl -fsSL -o /tmp/ffmpeg.tar.xz "$FFMPEG_TARBALL_URL"
    tar -xJf /tmp/ffmpeg.tar.xz -C "$FFMPEG_DIR" --strip-components=1
    rm -f /tmp/ffmpeg.tar.xz
    echo "ffmpeg libs installed:"
    ls "$FFMPEG_DIR/lib/" | grep -E '^libav(util|codec|format)\.so\.[0-9]+$' | sed 's/^/  /'
else
    echo "ffmpeg-shared already present at $FFMPEG_DIR"
fi

# 5. Verify torch + cuda + torchcodec all import cleanly with persistent env
echo
echo "=== Verifying installation ==="
LD_LIBRARY_PATH="$FFMPEG_DIR/lib:${LD_LIBRARY_PATH:-}" uv run python -c "
import torch, torchcodec
print(f'torch        : {torch.__version__}')
print(f'cuda avail   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu          : {torch.cuda.get_device_name(0)}')
    print(f'gpu memory   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'torchcodec   : {torchcodec.__version__}')
"

# 6. HF auth (interactive). Skip if already authed (token survives via HF_HOME).
if ! uv run hf auth whoami >/dev/null 2>&1; then
    echo
    echo "=== HF auth ==="
    echo "Run this manually to log in (token goes to $HF_HOME and survives):"
    echo "  uv run hf auth login"
    echo
    echo "Or if you have a token in an env var: uv run hf auth login --token \$HF_TOKEN"
else
    echo
    echo "HF auth: already logged in as $(uv run hf auth whoami | head -1)"
fi

echo
echo "=== First-time setup complete ==="
echo "Persistent paths:"
echo "  Repo       : $REPO_DIR"
echo "  uv cache   : $UV_CACHE_DIR  ($(du -sh "$UV_CACHE_DIR" 2>/dev/null | cut -f1))"
echo "  HF cache   : $HF_HOME  ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1))"
echo "  ffmpeg     : $FFMPEG_DIR"
echo
echo "For subsequent instances, run: bash scripts/local/setup_lambda_warmstart.sh"
echo "To start training:             cd $REPO_DIR && bash scripts/local/train_pi05_pick_color_block_run2_v3.sh"
