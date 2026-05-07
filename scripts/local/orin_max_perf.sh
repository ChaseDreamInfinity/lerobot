#!/usr/bin/env bash
# Put the Jetson Orin into a high-performance mode for policy rollout.
#
# Requires sudo. Use only when the Orin is plugged in and adequately cooled.
set -euo pipefail

sudo nvpmodel -m 0
sudo jetson_clocks
nvpmodel -q
