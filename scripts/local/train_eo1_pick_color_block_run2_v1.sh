#!/bin/bash
# EO-1 (Qwen2.5-VL 3B + flow matching) finetune on the 200-episode color-disambiguation dataset.
# Untested architecture for SO-101; trying because:
#   * Larger VLM than SmolVLA (3B vs 500M) -> more language capacity
#   * Native language conditioning via complementary_data["task"]
#   * Dynamic image resolution via Qwen processor (50K-100K pixels per image)
#   * Camera-key agnostic: picks up native dataset keys, no rename_map needed
# Dataset: chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315
#   - eps 0-49:   brown block, single-block scenes
#   - eps 50-99:  B1, single-block green/blue
#   - eps 100-149: B2, two-block scenes
#   - eps 150-199: B3, three-block scenes
# Goal: language-conditioned color disambiguation across {brown, green, blue}.
set -o pipefail

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

# === Single knob: bump VERSION to start a new tracked run within run2 (200-ep dataset) ===
VERSION=v1
NAME=eo1_so101-pick-color-block-in-cup-2cam-run2-${VERSION}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=outputs/train/${NAME}_${TIMESTAMP}
LOG_FILE=outputs/train/${NAME}_${TIMESTAMP}.log
SCRIPT_COPY=outputs/train/${NAME}_${TIMESTAMP}.script.sh
mkdir -p outputs/train
cp -- "$0" "$SCRIPT_COPY"

# To hold out eval episodes (95-99, 145-149, 195-199), uncomment:
# TRAIN_EPS=$(python3 -c "print(list(range(0,95))+list(range(100,145))+list(range(150,195)))")
# Then add:    --dataset.episodes="$TRAIN_EPS"

uv run lerobot-train \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --dataset.image_transforms.enable=true \
    --policy.type=eo1 \
    --policy.vlm_base=Qwen/Qwen2.5-VL-3B-Instruct \
    --policy.dtype=bfloat16 \
    --policy.attn_implementation=sdpa \
    --policy.gradient_checkpointing=true \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/${NAME} \
    --output_dir="$OUTPUT_DIR" \
    --job_name="${NAME}" \
    --batch_size=4 \
    --steps=60000 \
    --policy.scheduler_decay_steps=60000 \
    --save_freq=10000 \
    --log_freq=200 \
    --wandb.enable=false 2>&1 | tee "$LOG_FILE"
