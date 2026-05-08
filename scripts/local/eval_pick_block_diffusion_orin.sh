#!/usr/bin/env bash
# Run the trained Diffusion Policy on the SO-101 from the Jetson Orin.
#
# Twin of eval_pick_block_act_orin.sh (ACT) — same hardware setup, different
# policy. Uses the sync inference engine: diffusion's predict_action_chunk
# reads from internal observation queues that only select_action populates,
# which the RTC engine bypasses.
#
# Sampler override: the trained policy ships with DDPM/100 inference steps
# (~5s per chunk on Orin), which blows any usable per-chunk budget. At
# deploy time we use DDIM/5 (~250ms per chunk) — same weights, faster
# sampler. With n_action_steps=8 cached actions per chunk, FPS=15 leaves
# a 533ms refill budget, comfortably fitting the ~250ms denoise.
#
# Override hardware paths without editing the file, for example:
#   ROBOT_PORT=/dev/serial/by-id/... \
#     scripts/local/eval_pick_block_diffusion_orin.sh
#
# Performance / quality tuning:
#   NUM_INFERENCE_STEPS=10 scripts/local/eval_pick_block_diffusion_orin.sh   # closer to DDPM/100, slower
#   NOISE_SCHEDULER_TYPE=DDPM NUM_INFERENCE_STEPS=100 FPS=1 ...              # faithful to training, unusable
set -euo pipefail

cd /home/nvidia/Workspace/lerobot

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/diffusion_so101-pick-block-in-cup-2cam-run1}"
ROBOT_PORT="${ROBOT_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68008652-if00}"
ROBOT_ID="${ROBOT_ID:-my_follower}"
WRIST_CAMERA="${WRIST_CAMERA:-/dev/video2}"
TOP_CAMERA="${TOP_CAMERA:-/dev/video0}"
DEVICE="${DEVICE:-cuda}"
FPS="${FPS:-15}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"
INFERENCE_TYPE="${INFERENCE_TYPE:-sync}"
INTERPOLATION_MULTIPLIER="${INTERPOLATION_MULTIPLIER:-1}"
NOISE_SCHEDULER_TYPE="${NOISE_SCHEDULER_TYPE:-DDIM}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-5}"
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
  --policy.noise_scheduler_type="${NOISE_SCHEDULER_TYPE}" \
  --policy.num_inference_steps="${NUM_INFERENCE_STEPS}" \
  --task="${TASK}" \
  --fps="${FPS}" \
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}" \
  --duration="${DURATION}" \
  --device="${DEVICE}" \
  --display_data=false
)

if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  echo "INFERENCE_TYPE=rtc is not supported for the diffusion policy:" >&2
  echo "  predict_action_chunk reads from observation queues populated only by select_action," >&2
  echo "  which the RTC engine bypasses → empty-stack RuntimeError. Use sync instead." >&2
  exit 1
fi

"${args[@]}"
