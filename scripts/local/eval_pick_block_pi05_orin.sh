#!/usr/bin/env bash
# Run the trained Pi-0.5 (pi05) policy on the SO-101 from the Jetson Orin.
#
# Twin of eval_pick_block_smolvla_orin.sh — same hardware setup, different
# VLA. Pi-0.5 is a flow-matching VLA like SmolVLA: predict_action_chunk
# takes the observation batch directly and is RTC-compatible (unlike
# diffusion). RTC runs inference in a background thread so the main control
# loop ticks at FPS Hz with no inference stalls.
#
# Heavier than SmolVLA: pi-0.5's Gemma-based language tower runs slower
# per inference on Orin (~1.4s/chunk), so defaults are more conservative —
# FPS=15 and a higher RTC queue threshold so the queue doesn't drain
# during inference.
#
# RTC_ENABLED=false (append mode) is the default here, NOT replace mode:
# this fine-tune was not trained with prev_chunk_left_over conditioning,
# so RTC's mid-chunk replacement at action 21 produces large visible
# corrections every ~1.4s ("jump back" between chunks). Append mode keeps
# the async inference thread (no control-loop stall, no camera buffer
# overflow) and lets each chunk play to completion before the next is
# appended. The discontinuity moves to the chunk boundary every ~3.3s
# and is much smaller than the mid-chunk replace correction. Re-enable
# replace (RTC_ENABLED=true) only if you re-fine-tune with prev-chunk
# conditioning. Sync mode works for execution but the 1.4s freeze
# overflows the V4L2 camera buffer and cameras drop within ~15s.
#
# Camera-key rename: pi-0.5's input_features use base_0_rgb / left_wrist_0_rgb
# from its pretraining schema. Fine-tuning baked in:
# top → base_0_rgb, wrist → left_wrist_0_rgb. We pass the same map at deploy
# time so the robot's top/wrist keys feed the policy's expected slots.
#
# Override hardware paths without editing the file, for example:
#   ROBOT_PORT=/dev/serial/by-id/... \
#     scripts/local/eval_pick_block_pi05_orin.sh
#
# Performance tuning examples:
#   FPS=20 scripts/local/eval_pick_block_pi05_orin.sh
#   RTC_QUEUE_THRESHOLD=35 scripts/local/eval_pick_block_pi05_orin.sh
#   RTC_ENABLED=true scripts/local/eval_pick_block_pi05_orin.sh   # only if re-tuned with prev_chunk_left_over
set -euo pipefail

cd /home/nvidia/Workspace/lerobot

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/pi05_so101-pick-block-in-cup-2cam-run1-v2}"
ROBOT_PORT="${ROBOT_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68008652-if00}"
ROBOT_ID="${ROBOT_ID:-my_follower}"
WRIST_CAMERA="${WRIST_CAMERA:-/dev/video0}"
TOP_CAMERA="${TOP_CAMERA:-/dev/video2}"
DEVICE="${DEVICE:-cuda}"
FPS="${FPS:-15}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"
INFERENCE_TYPE="${INFERENCE_TYPE:-rtc}"
INTERPOLATION_MULTIPLIER="${INTERPOLATION_MULTIPLIER:-1}"
NUM_ACTION_STEPS="${NUM_ACTION_STEPS:-50}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_ENABLED="${RTC_ENABLED:-false}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-30}"
DURATION="${DURATION:-120}"
TASK="${TASK:-pick up the brown block and place it into the cup}"

if [[ ! -e "${ROBOT_PORT}" ]]; then
  echo "Robot port not found: ${ROBOT_PORT}" >&2
  echo "Check that the SO-101 follower is plugged in and run: ls -l /dev/serial/by-id /dev/ttyACM*" >&2
  exit 1
fi

if [[ ! -r "${ROBOT_PORT}" || ! -w "${ROBOT_PORT}" ]]; then
  echo "Robot port is not readable/writable by this user: ${ROBOT_PORT}" >&2
  echo "Permanent fix: sudo usermod -aG dialout ${USER} && log out/in." >&2
  echo "Temporary fix for this plug-in: sudo chmod a+rw $(readlink -f "${ROBOT_PORT}")" >&2
  exit 1
fi

for camera in "${WRIST_CAMERA}" "${TOP_CAMERA}"; do
  if [[ ! -e "${camera}" ]]; then
    echo "Camera not found: ${camera}" >&2
    echo "Check available cameras with: ls -l /dev/video*" >&2
    exit 1
  fi
done

if [[ "${DEVICE}" == cuda* ]]; then
  uv run --no-sync python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" || {
    echo "CUDA was requested with DEVICE=${DEVICE}, but PyTorch cannot access CUDA." >&2
    echo "Verify with: UV_CACHE_DIR=${UV_CACHE_DIR} uv run --no-sync python -c 'import torch; print(torch.cuda.is_available())'" >&2
    exit 1
  }
fi

args=(
  uv run --no-sync lerobot-rollout
  --policy.path="${POLICY_PATH}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{
    wrist: {type: opencv, index_or_path: \"${WRIST_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}},
    top:   {type: opencv, index_or_path: \"${TOP_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}}
  }" \
  --strategy.type=base \
  --inference.type="${INFERENCE_TYPE}" \
  --policy.n_action_steps="${NUM_ACTION_STEPS}" \
  --task="${TASK}" \
  --fps="${FPS}" \
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}" \
  --duration="${DURATION}" \
  --device="${DEVICE}" \
  --display_data=false \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
)

if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  args+=(
    --inference.rtc.enabled="${RTC_ENABLED}"
    --inference.rtc.execution_horizon="${RTC_EXECUTION_HORIZON}"
    --inference.queue_threshold="${RTC_QUEUE_THRESHOLD}"
  )
fi

"${args[@]}"
