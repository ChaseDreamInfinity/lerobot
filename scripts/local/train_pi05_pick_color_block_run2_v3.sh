#!/bin/bash
# Pi0.5 FULL FINE-TUNING on the 200-episode color-disambiguation dataset.
# v3 vs v1: unfreezes the entire model (vision tower + LLM + action expert).
# Target hardware: H100 80GB or A100 80GB. Does NOT fit on A10G 23GB.
#
# Why full-tuning:
#   * v1 (train_expert_only=true) gives correct language conditioning but
#     2-3 cm localization offset and occasional grasp-commit hesitation
#   * EO-1 (smaller, also fully unfrozen) confirmed A10G insufficient
#   * Hypothesis: unfreezing the PaliGemma vision tower lets the model adapt
#     its visual features to SO-101 viewpoints, fixing both precision and
#     grasp-commit at the source rather than working around them.
#
# Memory budget (rough):
#   ~4B trainable params, bf16 weights (~8 GB) + bf16 grads (~8 GB)
#   + fp32 AdamW states (~32 GB) + activations (~3-5 GB) -> ~55-60 GB total
#   Fits comfortably on H100 80GB / A100 80GB at bs=4.
#
# Overfitting mitigation:
#   * LR dropped 5x to 5e-6 (vs default 2.5e-5)
#   * save_freq=5000 to catch sweet spot before drift
#   * image_transforms enabled
#
# Dataset: chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315
#   - eps 0-49:   brown block, single-block scenes (existing)
#   - eps 50-99:  B1, single-block green/blue
#   - eps 100-149: B2, two-block scenes
#   - eps 150-199: B3, three-block scenes
set -o pipefail

# torchcodec needs ffmpeg 5-8 shared libs; Ubuntu 22.04 ships ffmpeg 4.
# On the cloud machine, you may need to set up the same ffmpeg-shared install.
export LD_LIBRARY_PATH="$HOME/ffmpeg-shared/lib:${LD_LIBRARY_PATH:-}"

cd "$(dirname "$0")/../.."

# === Single knob: bump VERSION to start a new tracked run within run2 (200-ep dataset) ===
VERSION=v3
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
    --dataset.repo_id=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \
    --dataset.image_transforms.enable=true \
    --policy.device=cuda \
    --policy.repo_id=chasedreaminf/${NAME} \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.optimizer_lr=5e-6 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="${NAME}" \
    --rename_map='{"observation.images.top":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --batch_size=4 \
    --steps=60000 \
    --policy.scheduler_decay_steps=60000 \
    --save_freq=5000 \
    --log_freq=200 \
    --wandb.enable=false 2>&1 | tee "$LOG_FILE"
