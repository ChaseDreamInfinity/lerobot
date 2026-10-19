#!/usr/bin/env bash
# Run the trained SmolVLA policy on the SO-101 from the Jetson Orin.
#
# Twin of eval_pick_block_act_orin.sh — same hardware setup, different policy. SmolVLA is a
# vision-language-action model: it reads the task text, so it can tell the brown block from the
# blue and green ones in the training data. It runs with Real-Time Chunking (RTC): inference runs
# in a background thread while the control loop executes the queued actions, so the loop never
# stalls on inference. --cuda-graph replays sampling from CUDA graphs (~130 ms per chunk instead
# of ~850 ms); see scripts/local/SMOLVLA_ORIN_OPTIMIZATION.md.
#
# Camera keys: SmolVLA names its cameras camera1/camera2/... The checkpoint was fine-tuned with
# wrist -> camera1 and top -> camera2 (its train_config.json and saved preprocessor both say so).
# --rename_map overrides the saved map, so it must repeat it exactly; a swapped map feeds the
# policy each view in the other's slot. Extra slots (camera3, empty_camera_0) are padded by the
# model up to config.empty_cameras.
#
# FPS defaults to 30, the rate the training data was recorded at: each predicted action is one
# 1/30 s step, so running slower plays the plan back in slow motion.
#
# Run with --help for overrides, or --dry-run to inspect the command.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

DRY_RUN=false
USE_CUDA_GRAPH=false
for option in "$@"; do
  case "${option}" in
    --help|-h)
      cat <<'HELP'
Usage: bash scripts/local/eval_pick_block_smolvla_orin.sh [--cuda-graph] [--dry-run]

Defaults: SmolVLA pick-color-block policy, CUDA, RTC inference, 120 seconds, no display.
Control: FPS=30 (the dataset's rate), INTERPOLATION_MULTIPLIER=1.
Sampling: NUM_STEPS=10 flow-matching steps, NUM_ACTION_STEPS=50 actions per chunk.
RTC: RTC_ENABLED=true, RTC_EXECUTION_HORIZON=10, RTC_QUEUE_THRESHOLD=22.
Cameras: CAMERA_WIDTH=640, CAMERA_HEIGHT=480, CAMERA_FOURCC=MJPG, wrist -> camera1, top -> camera2.

--cuda-graph replays the KV-cache fill and each denoising step from CUDA graphs captured at
startup: ~130 ms per chunk instead of ~850 ms, with identical outputs.

Environment overrides:
  POLICY_PATH   Hugging Face repository or local checkpoint directory
  ROBOT_PORT, ROBOT_ID, WRIST_CAMERA, TOP_CAMERA   Hardware configuration
  DEVICE, FPS, DURATION, TASK                       Rollout configuration
  NUM_STEPS, NUM_ACTION_STEPS, INFERENCE_TYPE        Sampling and inference engine
  RTC_ENABLED, RTC_EXECUTION_HORIZON, RTC_QUEUE_THRESHOLD
  LATENCY_LOG_INTERVAL_S=5   Seconds between control-loop latency log lines (0: off)
  CAMERA_MAX_FRAME_AGE_MS=2000, CAMERA_RECONNECT_TIMEOUT_S=5.0

Examples:
  bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph
  TASK="pick up the green block and place it into the cup" bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph
  DURATION=30 bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph --dry-run

Check a different checkpoint's camera rename map (train_config.json) before running it.
--dry-run prints the command without accessing hardware or loading the policy.
HELP
      exit 0
      ;;
    --cuda-graph) USE_CUDA_GRAPH=true ;;
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown option: ${option} (use --help)" >&2; exit 2 ;;
  esac
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/smolvla_so101-pick-color-block-in-cup-2cam-run2-v3}"
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
# Dropout tolerance. This rig's wrist camera intermittently falls off the USB bus and
# re-enumerates in ~1s. The camera read thread reopens the device instead of dying, and
# read_latest() keeps serving the last frame during the gap -- so a rollout rides out a
# dropout on a stale wrist image rather than crashing. Note the policy IS being fed a
# stale frame for up to this long, so keep it tight; set to 500 (the library default) to
# make dropouts fatal again.
CAMERA_MAX_FRAME_AGE_MS="${CAMERA_MAX_FRAME_AGE_MS:-2000}"
CAMERA_RECONNECT_TIMEOUT_S="${CAMERA_RECONNECT_TIMEOUT_S:-5.0}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"
INFERENCE_TYPE="${INFERENCE_TYPE:-rtc}"
INTERPOLATION_MULTIPLIER="${INTERPOLATION_MULTIPLIER:-1}"
NUM_ACTION_STEPS="${NUM_ACTION_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_ENABLED="${RTC_ENABLED:-true}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-22}"
DURATION="${DURATION:-120}"
TASK="${TASK:-pick up the brown block and place it into the cup}"
LATENCY_LOG_INTERVAL_S="${LATENCY_LOG_INTERVAL_S:-5}"
# Must match the checkpoint's training rename map (see the header).
RENAME_MAP='{"observation.images.wrist": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}'

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
  --policy.n_action_steps="${NUM_ACTION_STEPS}"
  --policy.num_steps="${NUM_STEPS}"
  --task="${TASK}"
  --fps="${FPS}"
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}"
  --duration="${DURATION}"
  --device="${DEVICE}"
  --display_data=false
  --latency_log_interval_s="${LATENCY_LOG_INTERVAL_S}"
  --rename_map="${RENAME_MAP}"
)

if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  args+=(
    --inference.rtc.enabled="${RTC_ENABLED}"
    --inference.rtc.execution_horizon="${RTC_EXECUTION_HORIZON}"
    --inference.queue_threshold="${RTC_QUEUE_THRESHOLD}"
  )
fi

if [[ "${USE_CUDA_GRAPH}" == true ]]; then
  args+=(--cuda_graph=true)
fi

printf 'Policy: %s\nDevice: %s | Inference: %s | Duration: %s s\n' \
  "${POLICY_PATH}" "${DEVICE}" "${INFERENCE_TYPE}" "${DURATION}"
printf 'Sampling: %s | Flow steps: %s | Actions per chunk: %s\n' \
  "$([[ "${USE_CUDA_GRAPH}" == true ]] && echo 'CUDA graphs' || echo 'PyTorch eager')" "${NUM_STEPS}" "${NUM_ACTION_STEPS}"
printf 'Control target: %s Hz | Task: %s\n' "${FPS}" "${TASK}"
printf 'Cameras: %sx%s %s | wrist -> camera1, top -> camera2\n' "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}" "${CAMERA_FOURCC}"

if [[ "${DRY_RUN}" == true ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

exec "${args[@]}"
