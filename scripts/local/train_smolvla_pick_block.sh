#!/bin/bash
# Finetune SmolVLA policy for pick-block-in-cup task with 2 cameras
# Uses lerobot/smolvla_base as the pretrained backbone

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
# BtbN shared build extracted to ~/ffmpeg-shared.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

uv run lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/smolvla_so101-pick-block-in-cup-2cam-run1 \
    --policy.empty_cameras=1 \
    --output_dir=outputs/train/smolvla_pick-block-in-cup-2cam-run1 \
    --job_name=smolvla_pick-block-in-cup-2cam-run1 \
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --batch_size=4 \
    --steps=20000 \
    --save_freq=2000 \
    --log_freq=200 \
    --wandb.enable=false
