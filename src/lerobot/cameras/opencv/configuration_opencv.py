# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation

__all__ = ["OpenCVCameraConfig", "ColorMode", "Cv2Rotation", "Cv2Backends"]


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """Configuration class for OpenCV-based camera devices or video files.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configurations:
    ```python
    # Basic configurations
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # Advanced configurations with FOURCC format
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90, fourcc="MJPG")     # With 90° rotation and MJPG format
    OpenCVCameraConfig(0, 30, 1280, 720, fourcc="YUYV")     # With YUYV format
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        fourcc: FOURCC code for video format (e.g., "MJPG", "YUYV", "I420"). Defaults to None (auto-detect).
        backend: OpenCV backend identifier (https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html). Defaults to ANY.
        max_read_failures: Consecutive failed reads tolerated before the background read
                      thread tries to reopen the device. Defaults to 10.
        reconnect_timeout_s: How long to keep trying to reopen a device that stopped
                      delivering frames, before giving up and killing the read thread.
                      This is a time budget rather than an attempt count on purpose: a
                      dead device fails reads instantly, so an attempt count is exhausted
                      in milliseconds, long before a USB camera finishes re-enumerating
                      (~1s). Set to 0 to restore the old fail-fast behaviour. Defaults to 5.0.
        reconnect_backoff_s: Delay between reopen attempts, in seconds. Defaults to 0.5.
        max_frame_age_ms: Default staleness tolerance for `read_latest()`. A USB camera that
                      drops off the bus typically needs ~1s to re-enumerate, so raise this
                      above that if you want rollouts to ride out a dropout on a stale frame
                      instead of raising. Defaults to 500.

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
        - FOURCC codes must be 4-character strings (e.g., "MJPG", "YUYV"). Some common FOUCC codes: https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs#fourcc-constants
        - Setting FOURCC can help achieve higher frame rates on some cameras.
    """

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY
    max_read_failures: int = 10
    reconnect_timeout_s: float = 5.0
    reconnect_backoff_s: float = 0.5
    max_frame_age_ms: int = 500

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.backend = Cv2Backends(self.backend)

        if self.max_read_failures < 0:
            raise ValueError(
                f"`max_read_failures` must be >= 0, but {self.max_read_failures} is provided."
            )
        for name in ("reconnect_timeout_s", "reconnect_backoff_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be >= 0, but {getattr(self, name)} is provided.")
        if self.max_frame_age_ms <= 0:
            raise ValueError(
                f"`max_frame_age_ms` must be > 0, but {self.max_frame_age_ms} is provided."
            )

        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )
