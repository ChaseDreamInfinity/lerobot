#!/bin/bash
# SmolVLA v2: longer training + reversed camera mapping
# Compare against v1 to see which mapping is correct

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
# BtbN shared build extracted to ~/ffmpeg-shared.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

uv run lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/smolvla_so101-pick-block-in-cup-2cam-run1-v2 \
    --policy.empty_cameras=1 \
    --output_dir=outputs/train/smolvla_pick-block-in-cup-2cam-run1-v2 \
    --job_name=smolvla_pick-block-in-cup-2cam-run1-v2 \
    --rename_map='{"observation.images.wrist":"observation.images.camera1","observation.images.top":"observation.images.camera2"}' \
    --batch_size=4 \
    --steps=60000 \
    --save_freq=5000 \
    --log_freq=200 \
    --wandb.enable=false
