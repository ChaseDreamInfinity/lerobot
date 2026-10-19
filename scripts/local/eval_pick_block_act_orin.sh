#!/usr/bin/env bash
# Run the trained ACT policy on the SO-101 from the Jetson Orin.
#
# Run with --help for overrides, --trt for the TensorRT forward pass, or --dry-run to
# inspect the command.
# Latency log fields and the TensorRT progress log: scripts/local/ACT_ORIN_OPTIMIZATION.md
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

DRY_RUN=false
USE_TRT=false
for option in "$@"; do
  case "${option}" in
    --help|-h)
      cat <<'HELP'
Usage: bash scripts/local/eval_pick_block_act_orin.sh [--trt] [--dry-run]

Defaults: ACT pick-block-in-cup policy, CUDA, 120 seconds, no display.
Control: FPS=30, INTERPOLATION_MULTIPLIER=1 (30 action updates/second).
Cameras: CAMERA_WIDTH=640, CAMERA_HEIGHT=480, CAMERA_FOURCC=MJPG.
Both cameras use CAMERA_FPS=30; WRIST_CAMERA_FPS and TOP_CAMERA_FPS override it.

--trt runs the forward pass on a TensorRT fp16 engine (~9 ms vs ~42 ms per inference tick).
The engine is built automatically on first use (about 80 s), at
outputs/trt_engines/<checkpoint name>_orin_fp16.engine, and reused afterwards.

Environment overrides:
  POLICY_PATH   Hugging Face repository or local checkpoint directory
  N_ACTION_STEPS   Actions executed per prediction (unset: checkpoint value)
  TRT_ENGINE   Use this TensorRT engine instead (implies --trt; must already exist)
  ROBOT_PORT, ROBOT_ID, WRIST_CAMERA, TOP_CAMERA   Hardware configuration
  DEVICE, FPS, DURATION, TASK, DISPLAY_DATA       Rollout configuration
  LATENCY_LOG_INTERVAL_S=5   Seconds between per-stage latency log lines (0: off)
  CAMERA_MAX_FRAME_AGE_MS=2000, CAMERA_RECONNECT_TIMEOUT_S=5.0

Examples:
  N_ACTION_STEPS=50 bash scripts/local/eval_pick_block_act_orin.sh
  TOP_CAMERA_FPS=15 bash scripts/local/eval_pick_block_act_orin.sh
  bash scripts/local/eval_pick_block_act_orin.sh --trt
  N_ACTION_STEPS=25 bash scripts/local/eval_pick_block_act_orin.sh --trt
  DURATION=30 bash scripts/local/eval_pick_block_act_orin.sh
  POLICY_PATH=chasedreaminf/act_so101-2cam-run2 bash scripts/local/eval_pick_block_act_orin.sh --dry-run

Check a different checkpoint's task and camera features before running it.
--dry-run prints the command without accessing hardware or loading the policy.
HELP
      exit 0
      ;;
    --trt) USE_TRT=true ;;
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown option: ${option} (use --help)" >&2; exit 2 ;;
  esac
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/act_so101-pick-block-in-cup-2cam-run1}"
# TensorRT: an explicit TRT_ENGINE must exist; with --trt alone the engine path follows the
# checkpoint (same naming as build_act_trt_engine_orin.sh) and is built if missing.
BUILD_TRT_ENGINE=false
if [[ -z "${TRT_ENGINE:-}" && "${USE_TRT}" == true ]]; then
  TRT_ENGINE="outputs/trt_engines/$(basename -- "${POLICY_PATH}")_orin_fp16.engine"
  [[ -f "${TRT_ENGINE}" ]] || BUILD_TRT_ENGINE=true
fi
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
FPS="${FPS:-30}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
# Both cameras default to CAMERA_FPS (30); override either rate independently.
WRIST_CAMERA_FPS="${WRIST_CAMERA_FPS:-${CAMERA_FPS}}"
TOP_CAMERA_FPS="${TOP_CAMERA_FPS:-${CAMERA_FPS}}"
# The control loop can use a cached image for up to 2 seconds during recovery.
# It stops at that limit even if the 5-second reconnect budget has time remaining.
CAMERA_MAX_FRAME_AGE_MS="${CAMERA_MAX_FRAME_AGE_MS:-2000}"
CAMERA_RECONNECT_TIMEOUT_S="${CAMERA_RECONNECT_TIMEOUT_S:-5.0}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"
INFERENCE_TYPE="${INFERENCE_TYPE:-sync}"
INTERPOLATION_MULTIPLIER="${INTERPOLATION_MULTIPLIER:-1}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_ENABLED="${RTC_ENABLED:-true}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-30}"
DURATION="${DURATION:-120}"
TASK="${TASK:-pick up the brown block and place it into the cup}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
LATENCY_LOG_INTERVAL_S="${LATENCY_LOG_INTERVAL_S:-5}"

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

  if [[ -n "${TRT_ENGINE:-}" && ! -f "${TRT_ENGINE}" && "${BUILD_TRT_ENGINE}" == false ]]; then
    echo "TensorRT engine not found: ${TRT_ENGINE}" >&2
    echo "Build it with: POLICY_PATH=${POLICY_PATH} bash scripts/local/build_act_trt_engine_orin.sh" >&2
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

  if [[ "${BUILD_TRT_ENGINE}" == true ]]; then
    echo "No TensorRT engine for this checkpoint yet; building ${TRT_ENGINE} (about 80 s)..."
    POLICY_PATH="${POLICY_PATH}" ENGINE="${TRT_ENGINE}" PRECISION=fp16 \
      bash scripts/local/build_act_trt_engine_orin.sh
  fi
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
  --task="${TASK}"
  --fps="${FPS}"
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}"
  --duration="${DURATION}"
  --device="${DEVICE}"
  --display_data="${DISPLAY_DATA}"
  --latency_log_interval_s="${LATENCY_LOG_INTERVAL_S}"
)

if [[ -n "${N_ACTION_STEPS:-}" ]]; then
  args+=(--policy.n_action_steps="${N_ACTION_STEPS}")
fi

if [[ -n "${TRT_ENGINE:-}" ]]; then
  args+=(--tensorrt_engine="${TRT_ENGINE}")
fi

if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  args+=(
    --inference.rtc.enabled="${RTC_ENABLED}"
    --inference.rtc.execution_horizon="${RTC_EXECUTION_HORIZON}"
    --inference.queue_threshold="${RTC_QUEUE_THRESHOLD}"
  )
fi

printf 'Policy: %s\nDevice: %s | Inference: %s | Duration: %s s\n' \
  "${POLICY_PATH}" "${DEVICE}" "${INFERENCE_TYPE}" "${DURATION}"
if [[ -n "${TRT_ENGINE:-}" ]]; then
  printf 'Forward pass: TensorRT %s\n' "${TRT_ENGINE}"
else
  printf 'Forward pass: PyTorch eager\n'
fi
if [[ "${DRY_RUN}" == true && "${BUILD_TRT_ENGINE}" == true ]]; then
  echo "  (engine not built yet: a real run builds it first, about 80 s)"
fi
printf 'Control target: %s Hz | Interpolation multiplier: %s\n' "${FPS}" "${INTERPOLATION_MULTIPLIER}"
printf 'Cameras: %sx%s %s | Wrist: %s FPS | Top: %s FPS\n' \
  "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}" "${CAMERA_FOURCC}" "${WRIST_CAMERA_FPS}" "${TOP_CAMERA_FPS}"

if [[ "${DRY_RUN}" == true ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

exec "${args[@]}"
