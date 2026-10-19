"""Regression checks for camera recovery without physical hardware."""

import unittest
from threading import Event
from unittest.mock import Mock, patch

import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


class TestOpenCVRecovery(unittest.TestCase):
    def make_camera(self):
        camera = OpenCVCamera(
            OpenCVCameraConfig(index_or_path=0, width=2, height=2, max_read_failures=0)
        )
        camera.stop_event = Event()
        camera.videocapture = Mock()
        camera.videocapture.read.return_value = (False, None)
        return camera

    def test_failed_reopen_retries_until_frame_arrives(self):
        camera = self.make_camera()
        missing = Mock()
        missing.isOpened.return_value = False
        recovered = Mock()

        def read_recovered():
            camera.stop_event.set()
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        recovered.read.side_effect = read_recovered
        connected_during_backoff = []

        def backoff(_timeout):
            connected_during_backoff.append(camera.is_connected)
            return False

        with (
            patch("lerobot.cameras.opencv.camera_opencv.cv2.VideoCapture", side_effect=[missing, recovered]) as open_capture,
            patch.object(camera, "_configure_capture_settings"),
            patch.object(camera.stop_event, "wait", side_effect=backoff),
        ):
            camera._read_loop()

        self.assertEqual(open_capture.call_count, 2)
        self.assertEqual(connected_during_backoff, [True, True])
        self.assertIsNotNone(camera.latest_frame)
        self.assertFalse(camera._reconnecting)

    def test_persistent_dropout_exhausts_budget(self):
        camera = self.make_camera()
        missing = Mock()
        missing.isOpened.return_value = False
        with (
            patch("lerobot.cameras.opencv.camera_opencv.cv2.VideoCapture", return_value=missing),
            patch("lerobot.cameras.opencv.camera_opencv.time.perf_counter", side_effect=[0.0, 6.0]),
            patch.object(camera.stop_event, "wait", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not recover within"):
                camera._read_loop()
        self.assertFalse(camera._reconnecting)
