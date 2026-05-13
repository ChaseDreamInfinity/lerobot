#!/bin/bash
# Train ACT policy for pick-block-in-cup task with 2 cameras

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
# BtbN shared build extracted to ~/ffmpeg-shared.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

uv run lerobot-train \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --policy.type=act \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/act_so101-pick-block-in-cup-2cam-run1 \
    --output_dir=outputs/train/act_pick-block-in-cup-2cam-run1 \
    --job_name=act_pick-block-in-cup-2cam-run1 \
    --batch_size=8 \
    --steps=50000 \
    --save_freq=5000 \
    --log_freq=200 \
    --wandb.enable=false
