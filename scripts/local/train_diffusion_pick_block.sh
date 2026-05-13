#!/bin/bash
# Train Diffusion policy for pick-block-in-cup task with 2 cameras

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
# BtbN shared build extracted to ~/ffmpeg-shared.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

uv run lerobot-train \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --policy.type=diffusion \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/diffusion_so101-pick-block-in-cup-2cam-run1 \
    --output_dir=outputs/train/diffusion_pick-block-in-cup-2cam-run1 \
    --job_name=diffusion_pick-block-in-cup-2cam-run1 \
    --policy.use_amp=true \
    --policy.noise_scheduler_type=DDIM \
    --policy.num_inference_steps=10 \
    --batch_size=32 \
    --num_workers=4 \
    --steps=200000 \
    --save_freq=10000 \
    --log_freq=200 \
    --wandb.enable=false
