#!/usr/bin/env bash
# Put the Jetson Orin into a high-performance mode for policy rollout.
#
# Requires sudo. Use only when the Orin is plugged in and adequately cooled.
#
# Both settings are volatile: nvpmodel/jetson_clocks reset on reboot, and USB
# autosuspend resets on reboot *and* whenever a camera re-enumerates. Re-run this
# script after either.
set -euo pipefail

sudo nvpmodel -m 0
sudo jetson_clocks
nvpmodel -q

# Keep USB autosuspend off for the UVC cameras and every hub between them and the
# root port. The cameras ship with power/control=auto and a 2s idle timer, which is
# a known cause of mid-rollout UVC dropouts (the camera falls off the bus and
# re-enumerates, and lerobot's read thread dies with "exceeded maximum consecutive
# read failures"). A suspended hub takes its downstream cameras with it, so walk up
# the chain rather than touching the camera alone.
echo
echo "Disabling USB autosuspend for UVC cameras..."
declare -A seen=()
for iface in /sys/bus/usb/devices/*:*; do
  [[ -L "${iface}/driver" ]] || continue
  [[ "$(basename "$(readlink -f "${iface}/driver")")" == "uvcvideo" ]] || continue

  # /sys/bus/usb/devices/1-4.3:1.0 -> device 1-4.3, then hubs 1-4, 1
  node="$(basename "${iface%%:*}")"
  while :; do
    control="/sys/bus/usb/devices/${node}/power/control"
    if [[ -e "${control}" && -z "${seen[${node}]:-}" ]]; then
      seen["${node}"]=1
      echo on | sudo tee "${control}" >/dev/null
      product="$(cat "/sys/bus/usb/devices/${node}/product" 2>/dev/null || echo "USB device")"
      echo "  ${node} (${product}): autosuspend off"
    fi
    [[ "${node}" == *.* ]] || break
    node="${node%.*}"
  done
done

if [[ ${#seen[@]} -eq 0 ]]; then
  echo "  WARNING: no uvcvideo devices found — are the cameras plugged in?" >&2
fi
