#!/bin/bash
# Pi0.5 finetune on the TOP-CROPPED 200-episode color-disambiguation dataset.
# v2 vs v1: same hyperparams, but the top camera in the dataset has been cropped
# from 640x480 to 352x352 (region 288,0,352,352) via preprocess_crop_top_cam.py.
# Wrist camera unchanged. Expected effect: ~1.9x linear pixel-per-object boost
# on top cam after pi0.5's 224x224 resize -> reduces localization offset from
# 2-3 cm to ~1-1.5 cm. Does NOT fix grasp-commit hesitation (architectural).
# Inference must apply the SAME crop to live top-cam frames before passing to
# the policy. See eval_pick_block_pi05_orin.sh for the corresponding edit.
# Source dataset (pre-crop): chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315
# Cropped dataset (this script trains on): chasedreaminf/record-pick-block-in-cup-2cam-topcrop_20260512
#   - same 200 episodes:
#   - eps 0-49:   brown block, single-block scenes
#   - eps 50-99:  B1, single-block green/blue
#   - eps 100-149: B2, two-block scenes
#   - eps 150-199: B3, three-block scenes
# Goal: language-conditioned color disambiguation across {brown, green, blue}.
set -o pipefail

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

# === Single knob: bump VERSION to start a new tracked run within run2 (extended 200-ep dataset) ===
VERSION=v2
NAME=pi05_so101-pick-color-block-in-cup-2cam-run2-${VERSION}

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
    --policy.path=lerobot/pi05_base \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-topcrop_20260512 \
    --dataset.image_transforms.enable=true \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/${NAME} \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=true \
    --output_dir="$OUTPUT_DIR" \
    --job_name="${NAME}" \
    --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --batch_size=4 \
    --steps=60000 \
    --policy.scheduler_decay_steps=60000 \
    --save_freq=10000 \
    --log_freq=200 \
    --wandb.enable=false 2>&1 | tee "$LOG_FILE"
