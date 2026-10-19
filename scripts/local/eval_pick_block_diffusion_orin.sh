#!/usr/bin/env bash
# Run the trained Diffusion Policy on the SO-101 from the Jetson Orin.
#
# Twin of eval_pick_block_act_orin.sh (ACT) — same hardware setup, different
# policy. Uses the sync inference engine: diffusion's predict_action_chunk
# reads from internal observation queues that only select_action populates,
# which the RTC engine bypasses.
#
# Defaults to DDIM/5 for faster sampling with the existing checkpoint.
# Synchronous prediction blocks the control loop at each chunk boundary;
# FPS=15 is a target, not a guarantee that prediction fits in one tick.
# Use --help for overrides or --dry-run to inspect the command.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

DRY_RUN=false
case "${1:-}" in
  --help|-h)
    cat <<'HELP'
Usage: bash scripts/local/eval_pick_block_diffusion_orin.sh [--dry-run]

Defaults: Diffusion pick-block-in-cup policy, CUDA, 120 seconds, no display.
Control: FPS=15 target, INTERPOLATION_MULTIPLIER=1; synchronous inference.
Sampler: NOISE_SCHEDULER_TYPE=DDIM, NUM_INFERENCE_STEPS=5.
Cameras: CAMERA_WIDTH=640, CAMERA_HEIGHT=480, CAMERA_FOURCC=MJPG.
Both cameras use CAMERA_FPS=30; WRIST_CAMERA_FPS and TOP_CAMERA_FPS override it.

Environment overrides:
  POLICY_PATH   Hugging Face repository or local checkpoint directory
  N_ACTION_STEPS   Actions executed per prediction (unset: checkpoint value)
  ROBOT_PORT, ROBOT_ID, WRIST_CAMERA, TOP_CAMERA   Hardware configuration
  DEVICE, FPS, DURATION, TASK, DISPLAY_DATA       Rollout configuration
  CAMERA_MAX_FRAME_AGE_MS=2000, CAMERA_RECONNECT_TIMEOUT_S=5.0

Examples:
  N_ACTION_STEPS=8 bash scripts/local/eval_pick_block_diffusion_orin.sh
  TOP_CAMERA_FPS=15 bash scripts/local/eval_pick_block_diffusion_orin.sh
  DURATION=30 bash scripts/local/eval_pick_block_diffusion_orin.sh
  POLICY_PATH=chasedreaminf/diffusion_so101-pick-block-in-cup-2cam-run1 bash scripts/local/eval_pick_block_diffusion_orin.sh --dry-run

Check a different checkpoint's task and camera features before running it.
--dry-run prints the command without accessing hardware or loading the policy.
HELP
    exit 0
    ;;
  --dry-run) DRY_RUN=true ;;
  "") ;;
  *) echo "Unknown option: $1 (use --help)" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Expected at most one option (use --help)." >&2
  exit 2
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/diffusion_so101-pick-block-in-cup-2cam-run1}"
ROBOT_PORT="${ROBOT_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68008652-if00}"
ROBOT_ID="${ROBOT_ID:-my_follower}"
# Cameras default to /dev/v4l/by-path/... rather than /dev/videoN: the numbering
# reshuffles whenever a camera re-enumerates, and this rig's wrist camera does that
# under load. by-path is tied to the physical USB port, so the wrist/top roles stay put.
#   wrist -> Orin port 4.4 (Microdia 0c45:6366)
#   top   -> Orin port 4.1 (ARC 05a3:9230)
WRIST_CAMERA="${WRIST_CAMERA:-/dev/v4l/by-path/platform-3610000.usb-usb-0:4.4:1.0-video-index0}"
TOP_CAMERA="${TOP_CAMERA:-/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1:1.0-video-index0}"

DEVICE="${DEVICE:-cuda}"
FPS="${FPS:-15}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
# Both cameras default to CAMERA_FPS (30); override either rate independently.
WRIST_CAMERA_FPS="${WRIST_CAMERA_FPS:-${CAMERA_FPS}}"
TOP_CAMERA_FPS="${TOP_CAMERA_FPS:-${CAMERA_FPS}}"
# Cached images expire after 2 seconds, even if reconnection is still trying.
CAMERA_MAX_FRAME_AGE_MS="${CAMERA_MAX_FRAME_AGE_MS:-2000}"
CAMERA_RECONNECT_TIMEOUT_S="${CAMERA_RECONNECT_TIMEOUT_S:-5.0}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"
INFERENCE_TYPE="${INFERENCE_TYPE:-sync}"
if [[ "${INFERENCE_TYPE}" != sync ]]; then
  echo "Diffusion requires INFERENCE_TYPE=sync: its observation queues are populated by select_action." >&2
  exit 2
fi
INTERPOLATION_MULTIPLIER="${INTERPOLATION_MULTIPLIER:-1}"
NOISE_SCHEDULER_TYPE="${NOISE_SCHEDULER_TYPE:-DDIM}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-5}"
DURATION="${DURATION:-120}"
TASK="${TASK:-pick up the brown block and place it into the cup}"

DISPLAY_DATA="${DISPLAY_DATA:-false}"

if [[ "${DRY_RUN}" == false ]]; then
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
      echo "Check available cameras with: ls -l /dev/v4l/by-path/ /dev/video*" >&2
      exit 1
    fi
  done

  # shellcheck source=scripts/local/_cuda_guard.sh
  source scripts/local/_cuda_guard.sh
  require_cuda "${DEVICE}"
fi

args=(
  uv run --no-sync lerobot-rollout
  --policy.path="${POLICY_PATH}"
  --robot.type=so101_follower
  --robot.port="${ROBOT_PORT}"
  --robot.id="${ROBOT_ID}"
  --robot.cameras="{
    wrist: {type: opencv, backend: V4L2, index_or_path: \"${WRIST_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${WRIST_CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}, max_frame_age_ms: ${CAMERA_MAX_FRAME_AGE_MS}, reconnect_timeout_s: ${CAMERA_RECONNECT_TIMEOUT_S}},
    top:   {type: opencv, backend: V4L2, index_or_path: \"${TOP_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${TOP_CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}, max_frame_age_ms: ${CAMERA_MAX_FRAME_AGE_MS}, reconnect_timeout_s: ${CAMERA_RECONNECT_TIMEOUT_S}}
  }"
  --strategy.type=base
  --inference.type="${INFERENCE_TYPE}"
  --policy.noise_scheduler_type="${NOISE_SCHEDULER_TYPE}"
  --policy.num_inference_steps="${NUM_INFERENCE_STEPS}"
  --task="${TASK}"
  --fps="${FPS}"
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}"
  --duration="${DURATION}"
  --device="${DEVICE}"
  --display_data="${DISPLAY_DATA}"
)

if [[ -n "${N_ACTION_STEPS:-}" ]]; then
  args+=(--policy.n_action_steps="${N_ACTION_STEPS}")
fi

printf 'Policy: %s\nDevice: %s | Inference: %s | Duration: %s s\n' \
  "${POLICY_PATH}" "${DEVICE}" "${INFERENCE_TYPE}" "${DURATION}"
printf 'Sampler: %s | Denoising steps: %s | Action steps: %s\n' \
  "${NOISE_SCHEDULER_TYPE}" "${NUM_INFERENCE_STEPS}" "${N_ACTION_STEPS:-checkpoint default}"
printf 'Control target: %s Hz | Interpolation multiplier: %s\n' "${FPS}" "${INTERPOLATION_MULTIPLIER}"
printf 'Cameras: %sx%s %s | Wrist: %s FPS | Top: %s FPS\n' \
  "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}" "${CAMERA_FOURCC}" "${WRIST_CAMERA_FPS}" "${TOP_CAMERA_FPS}"

if [[ "${DRY_RUN}" == true ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

exec "${args[@]}"
