#!/bin/bash
# SmolVLA finetune on the 200-episode color-disambiguation dataset.
# v3 vs v2: SAME unfrozen-VLM config, but EXCLUDE eps 0-49 (the brown single-block
# anti-signal). v2 showed only occasional correct-color picks; the 50 original
# brown-only single-block episodes were training the model to ignore language
# (every episode says "pick brown" with brown as the only choice). Removing them
# leaves only B1/B2/B3 (eps 50-199, 150 eps) where the color word is the
# disambiguator. Brown becomes least-represented (only in B2/B3 multi-block
# scenes, ~24 eps) but every brown occurrence carries language signal.
# Dataset: chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315
#   - eps 0-49:   brown block, single-block scenes (EXCLUDED)
#   - eps 50-99:  B1, single-block green/blue
#   - eps 100-149: B2, two-block scenes
#   - eps 150-199: B3, three-block scenes
# Goal: language-conditioned color disambiguation across {brown, green, blue}.
set -o pipefail

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

# === Single knob: bump VERSION to start a new tracked run within run2 (extended 200-ep dataset) ===
VERSION=v3
NAME=smolvla_so101-pick-color-block-in-cup-2cam-run2-${VERSION}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=outputs/train/${NAME}_${TIMESTAMP}
LOG_FILE=outputs/train/${NAME}_${TIMESTAMP}.log
SCRIPT_COPY=outputs/train/${NAME}_${TIMESTAMP}.script.sh
mkdir -p outputs/train
cp -- "$0" "$SCRIPT_COPY"

# Drop eps 0-49 (brown single-block, language-anti-signal). Train on eps 50-199.
TRAIN_EPS=$(python3 -c "print(list(range(50, 200)))")

uv run lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --dataset.episodes="$TRAIN_EPS" \
    --dataset.image_transforms.enable=true \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/${NAME} \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.optimizer_lr=5e-5 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="${NAME}" \
    --rename_map='{"observation.images.wrist":"observation.images.camera1","observation.images.top":"observation.images.camera2"}' \
    --batch_size=4 \
    --steps=60000 \
    --policy.scheduler_decay_steps=60000 \
    --save_freq=5000 \
    --log_freq=200 \
    --wandb.enable=false 2>&1 | tee "$LOG_FILE"
