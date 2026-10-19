#!/usr/bin/env bash
# Run the trained SmolVLA policy on the SO-101 from the Jetson Orin.
#
# Twin of eval_pick_block_act_orin.sh — same hardware setup, different
# policy. SmolVLA is a vision-language-action model: heavier than ACT,
# but its predict_action_chunk takes the observation batch directly and
# is RTC-compatible (unlike diffusion). RTC runs inference in a background
# thread, so the main control loop ticks at FPS Hz with no inference
# stalls — same execution profile as ACT.
#
# Chunk size is 50 (vs ACT's typical 100), so RTC keeps a deeper queue
# fed automatically. At FPS=30 a chunk is consumed in 1.67s, giving
# generous headroom for VLA inference latency.
#
# Camera-key rename: SmolVLA's input_features use camera1/camera2/...
# from its pretraining schema. The fine-tuning baked in: top → camera1,
# wrist → camera2. We pass the same map at deploy time via --rename_map
# so the robot's wrist/top keys feed the policy's expected camera2/camera1
# slots. Any additional pretraining keys (camera3, empty_camera_0) get
# zero-padded by the SmolVLA model up to its config.empty_cameras count.
#
# Override hardware paths without editing the file, for example:
#   ROBOT_PORT=/dev/serial/by-id/... \
#     scripts/local/eval_pick_block_smolvla_orin.sh
#
# Performance tuning examples:
#   FPS=15 scripts/local/eval_pick_block_smolvla_orin.sh
#   RTC_ENABLED=true scripts/local/eval_pick_block_smolvla_orin.sh
set -euo pipefail

cd /home/nvidia/Workspace/lerobot

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

#POLICY_PATH="${POLICY_PATH:-chasedreaminf/smolvla_so101-pick-block-in-cup-2cam-run1-v2}"
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
FPS="${FPS:-20}"
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

args=(
  uv run --no-sync lerobot-rollout
  --policy.path="${POLICY_PATH}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{
    wrist: {type: opencv, index_or_path: \"${WRIST_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${WRIST_CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}, max_frame_age_ms: ${CAMERA_MAX_FRAME_AGE_MS}, reconnect_timeout_s: ${CAMERA_RECONNECT_TIMEOUT_S}},
    top:   {type: opencv, index_or_path: \"${TOP_CAMERA}\", width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${TOP_CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}, max_frame_age_ms: ${CAMERA_MAX_FRAME_AGE_MS}, reconnect_timeout_s: ${CAMERA_RECONNECT_TIMEOUT_S}}
  }" \
  --strategy.type=base \
  --inference.type="${INFERENCE_TYPE}" \
  --policy.n_action_steps="${NUM_ACTION_STEPS}" \
  --policy.num_steps="${NUM_STEPS}" \
  --task="${TASK}" \
  --fps="${FPS}" \
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}" \
  --duration="${DURATION}" \
  --device="${DEVICE}" \
  --display_data=false \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
)

if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  args+=(
    --inference.rtc.enabled="${RTC_ENABLED}"
    --inference.rtc.execution_horizon="${RTC_EXECUTION_HORIZON}"
    --inference.queue_threshold="${RTC_QUEUE_THRESHOLD}"
  )
fi

"${args[@]}"
