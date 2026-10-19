#!/usr/bin/env bash
# Run the trained Pi-0.5 (pi05) policy on the SO-101 from the Jetson Orin.
#
# Twin of eval_pick_block_smolvla_orin.sh -- same hardware setup, different VLA. Pi-0.5 is a
# flow-matching VLA like SmolVLA (PaliGemma 2B + a 300M action expert, 9.4 GB in memory). It runs
# with Real-Time Chunking (RTC): inference runs in a background thread while the control loop
# executes the queued actions, so the loop never stalls on inference. --trt runs it on TensorRT
# engines (~0.25 s per chunk, actions closer to fp32 than the bf16 model); --cuda-graph replays
# PyTorch from CUDA graphs and encodes images in TF32 (~0.5 s), needing no engines; eager PyTorch
# takes ~1.15 s. See scripts/local/PI05_ORIN_OPTIMIZATION.md.
#
# FPS defaults to 30, the rate the training data was recorded at: each predicted action is one
# 1/30 s step, so running slower plays the plan back in slow motion. At 30 Hz only --trt and
# --cuda-graph keep up: eager inference (~1.15 s, 35 ticks) outlasts the actions left in the queue.
#
# RTC replace mode (RTC_ENABLED=true) swaps in each new chunk minus the actions executed during
# inference, guided to stay consistent with the leftover plan for RTC_EXECUTION_HORIZON actions.
# The horizon must exceed the inference delay (~8 ticks with --trt, ~16 with --cuda-graph at
# 30 Hz), or the
# actions executed during inference are never guided and the arm jumps at each new chunk.
# RTC_ENABLED=false appends each chunk behind the RTC_QUEUE_THRESHOLD actions still queued, so
# every executed chunk was predicted from an observation at least that many ticks old.
#
# Camera keys: pi-0.5's input features use base_0_rgb / left_wrist_0_rgb from its pretraining
# schema. The checkpoint was fine-tuned with top -> base_0_rgb and wrist -> left_wrist_0_rgb (its
# train_config.json says so); the third slot (right_wrist_0_rgb) is padded and masked out.
#
# Run with --help for overrides, or --dry-run to inspect the command.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

DRY_RUN=false
USE_CUDA_GRAPH=false
USE_TRT=false
for option in "$@"; do
  case "${option}" in
    --help|-h)
      cat <<'HELP'
Usage: bash scripts/local/eval_pick_block_pi05_orin.sh [--trt | --cuda-graph] [--dry-run]

Defaults: Pi-0.5 pick-color-block policy, CUDA, RTC inference, 120 seconds, no display.
Control: FPS=30 (the dataset's rate), INTERPOLATION_MULTIPLIER=1.
Sampling: NUM_ACTION_STEPS=50 actions per chunk (10 flow-matching steps, from the checkpoint).
RTC: RTC_ENABLED=true, RTC_EXECUTION_HORIZON=25, RTC_QUEUE_THRESHOLD=25.
Cameras: CAMERA_WIDTH=640, CAMERA_HEIGHT=480, CAMERA_FOURCC=MJPG, top -> base_0_rgb, wrist -> left_wrist_0_rgb.

--trt runs image encoding, KV fill and denoising steps on TensorRT fp16 engines, replayed from
CUDA graphs: ~0.25 s per chunk, actions about 2x closer to fp32 than the bf16 model. The engines
are built automatically on first use (about 4 min), in outputs/trt_engines/<checkpoint name>_orin_fp16/.
--cuda-graph replays PyTorch from CUDA graphs and encodes images in TF32 instead: ~0.5 s per
chunk, no engines to build. Eager PyTorch takes ~1.15 s, too slow for 30 Hz.

Environment overrides:
  POLICY_PATH   Hugging Face repository or local checkpoint directory
  TRT_ENGINE    Use this TensorRT engine directory instead (implies --trt; must already exist)
  ROBOT_PORT, ROBOT_ID, WRIST_CAMERA, TOP_CAMERA   Hardware configuration
  DEVICE, FPS, DURATION, TASK                       Rollout configuration
  NUM_ACTION_STEPS, INFERENCE_TYPE                   Chunk length and inference engine
  RTC_ENABLED, RTC_EXECUTION_HORIZON, RTC_QUEUE_THRESHOLD
  LATENCY_LOG_INTERVAL_S=5   Seconds between control-loop latency log lines (0: off)
  CAMERA_MAX_FRAME_AGE_MS=2000, CAMERA_RECONNECT_TIMEOUT_S=5.0

Examples:
  bash scripts/local/eval_pick_block_pi05_orin.sh --trt
  bash scripts/local/eval_pick_block_pi05_orin.sh --cuda-graph
  TASK="pick up the green block and place it into the cup" bash scripts/local/eval_pick_block_pi05_orin.sh --trt
  # The previous defaults (10 Hz, append mode):
  FPS=10 NUM_ACTION_STEPS=20 RTC_ENABLED=false RTC_EXECUTION_HORIZON=10 RTC_QUEUE_THRESHOLD=30 \
    bash scripts/local/eval_pick_block_pi05_orin.sh

Check a different checkpoint's camera rename map (train_config.json) before running it.
--dry-run prints the command without accessing hardware or loading the policy.
HELP
      exit 0
      ;;
    --cuda-graph) USE_CUDA_GRAPH=true ;;
    --trt) USE_TRT=true ;;
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown option: ${option} (use --help)" >&2; exit 2 ;;
  esac
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_PATH="${POLICY_PATH:-chasedreaminf/pi05_so101-pick-color-block-in-cup-2cam-run2-v1}"
# TensorRT: an explicit TRT_ENGINE must exist; with --trt alone the engine directory follows the
# checkpoint (same naming as build_pi05_trt_engine_orin.sh) and is built if missing.
BUILD_TRT_ENGINE=false
if [[ -z "${TRT_ENGINE:-}" && "${USE_TRT}" == true ]]; then
  TRT_ENGINE="outputs/trt_engines/$(basename -- "${POLICY_PATH}")_orin_fp16"
  for engine in vision kv step; do
    [[ -f "${TRT_ENGINE}/${engine}.engine" ]] || BUILD_TRT_ENGINE=true
  done
fi
if [[ -n "${TRT_ENGINE:-}" && "${USE_CUDA_GRAPH}" == true ]]; then
  echo "--trt already uses CUDA graphs; pass only one of --trt and --cuda-graph." >&2
  exit 2
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
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-25}"
RTC_ENABLED="${RTC_ENABLED:-true}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-25}"
DURATION="${DURATION:-120}"
TASK="${TASK:-pick up the blue block and place it into the cup}"
LATENCY_LOG_INTERVAL_S="${LATENCY_LOG_INTERVAL_S:-5}"
# Must match the checkpoint's training rename map (see the header).
RENAME_MAP='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'

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

  if [[ -n "${TRT_ENGINE:-}" && ! -d "${TRT_ENGINE}" && "${BUILD_TRT_ENGINE}" == false ]]; then
    echo "TensorRT engine directory not found: ${TRT_ENGINE}" >&2
    echo "Build it with: POLICY_PATH=${POLICY_PATH} bash scripts/local/build_pi05_trt_engine_orin.sh" >&2
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
    echo "No TensorRT engines for this checkpoint yet; building ${TRT_ENGINE} (about 4 min)..."
    POLICY_PATH="${POLICY_PATH}" ENGINE_DIR="${TRT_ENGINE}" NUM_CAMERAS=2 \
      bash scripts/local/build_pi05_trt_engine_orin.sh
  fi
fi

args=(
  uv run --no-sync lerobot-rollout
  --policy.path="${POLICY_PATH}"
  --robot.type=so101_follower
  --robot.port="${ROBOT_PORT}"
  --robot.id="${ROBOT_ID}"
  --robot.cameras="{
    wrist: {type: opencv, index_or_path: \"${WRIST_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${WRIST_CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}, max_frame_age_ms: ${CAMERA_MAX_FRAME_AGE_MS}, reconnect_timeout_s: ${CAMERA_RECONNECT_TIMEOUT_S}},
    top:   {type: opencv, index_or_path: \"${TOP_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${TOP_CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}, max_frame_age_ms: ${CAMERA_MAX_FRAME_AGE_MS}, reconnect_timeout_s: ${CAMERA_RECONNECT_TIMEOUT_S}}
  }"
  --strategy.type=base
  --inference.type="${INFERENCE_TYPE}"
  --policy.n_action_steps="${NUM_ACTION_STEPS}"
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

if [[ -n "${TRT_ENGINE:-}" ]]; then
  args+=(--tensorrt_engine="${TRT_ENGINE}")
fi

printf 'Policy: %s\nDevice: %s | Inference: %s | Duration: %s s\n' \
  "${POLICY_PATH}" "${DEVICE}" "${INFERENCE_TYPE}" "${DURATION}"
printf 'Sampling: %s | Actions per chunk: %s | RTC: %s (horizon %s, queue threshold %s)\n' \
  "$(if [[ -n "${TRT_ENGINE:-}" ]]; then echo "TensorRT ${TRT_ENGINE}"; elif [[ "${USE_CUDA_GRAPH}" == true ]]; then echo 'CUDA graphs'; else echo 'PyTorch eager'; fi)" \
  "${NUM_ACTION_STEPS}" "${RTC_ENABLED}" "${RTC_EXECUTION_HORIZON}" "${RTC_QUEUE_THRESHOLD}"
if [[ "${DRY_RUN}" == true && "${BUILD_TRT_ENGINE}" == true ]]; then
  echo "  (engines not built yet: a real run builds them first, about 4 min)"
fi
printf 'Control target: %s Hz | Task: %s\n' "${FPS}" "${TASK}"
printf 'Cameras: %sx%s %s | top -> base_0_rgb, wrist -> left_wrist_0_rgb\n' "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}" "${CAMERA_FOURCC}"
if [[ "${USE_CUDA_GRAPH}" == false && -z "${TRT_ENGINE:-}" && "${FPS}" -gt 10 ]]; then
  echo "Warning: eager Pi-0.5 (~1.15 s per chunk) cannot keep the queue filled at ${FPS} Hz; add --trt or --cuda-graph." >&2
fi

if [[ "${DRY_RUN}" == true ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

exec "${args[@]}"
