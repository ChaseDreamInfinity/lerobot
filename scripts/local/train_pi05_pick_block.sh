#!/bin/bash
# Finetune Pi0.5 (π₀.₅) policy for pick-block-in-cup task with 2 cameras
# Uses lerobot/pi05_base as the pretrained backbone
set -o pipefail

# torchcodec 0.10 needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
# Use the BtbN shared build extracted to ~/ffmpeg-shared (libavutil.so.60 = ffmpeg 8).
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=outputs/train/pi05_pick-block-in-cup-2cam-run1_$TIMESTAMP
LOG_FILE=outputs/train/pi05_pick-block-in-cup-2cam-run1_$TIMESTAMP.log
mkdir -p outputs/train

uv run lerobot-train \
    --policy.path=lerobot/pi05_base \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/pi05_so101-pick-block-in-cup-2cam-run1-v2 \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=true \
    --output_dir="$OUTPUT_DIR" \
    --job_name=pi05_pick-block-in-cup-2cam-run1 \
    --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --dataset.image_transforms.enable=true \
    --batch_size=4 \
    --steps=30000 \
    --save_freq=5000 \
    --log_freq=200 \
    --wandb.enable=false 2>&1 | tee "$LOG_FILE"
