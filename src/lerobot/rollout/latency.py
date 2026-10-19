# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Per-stage latency statistics for the rollout control loop, logged periodically."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _ordered(stages) -> list[str]:
    """Stages in first-seen order with ``loop`` last. A tick that sends no action (e.g. before the
    first chunk arrives) records ``loop`` without ``send``, which would otherwise reorder the log."""
    return [stage for stage in stages if stage != "loop"] + (["loop"] if "loop" in stages else [])


@dataclass
class _Stats:
    samples: list[float] = field(default_factory=list)

    def add(self, value_s: float) -> None:
        self.samples.append(value_s * 1e3)

    def summary(self) -> str:
        values = sorted(self.samples)
        mean = sum(values) / len(values)
        p99 = values[min(len(values) - 1, math.ceil(0.99 * len(values)) - 1)]
        return f"{mean:.1f}/{p99:.1f}/{values[-1]:.1f}"


class LatencyMonitor:
    """Collect stage timings per control tick and log a summary every ``interval_s``.

    Usage per tick: ``start_tick()``, then ``mark(stage)`` after each stage (it records the time
    since the previous mark), then ``end_tick()`` before sleeping. A tick abandoned before
    ``end_tick`` (e.g. during warmup) is not recorded.

    The whole-run summary logged by ``log_run_summary`` keeps mean/max only, so memory stays
    bounded in long (24/7) rollouts.
    """

    def __init__(self, interval_s: float, fps: float):
        self.interval_s = interval_s
        self.budget_s = 1.0 / fps
        self.fps = fps
        self._window: dict[str, _Stats] = {}
        self._run: dict[str, list[float]] = {}  # stage -> [count, sum_ms, max_ms]
        self._window_start = time.perf_counter()
        self._window_ticks = 0
        self._window_overruns = 0
        self._run_start = self._window_start
        self._run_ticks = 0
        self._run_overruns = 0
        self._tick_start: float | None = None
        self._last_mark = 0.0

    def start_tick(self) -> None:
        self._tick_start = self._last_mark = time.perf_counter()

    def mark(self, stage: str) -> None:
        if self._tick_start is None:
            return
        now = time.perf_counter()
        self._record(stage, now - self._last_mark)
        self._last_mark = now

    def end_tick(self) -> None:
        if self._tick_start is None:
            return
        now = time.perf_counter()
        loop_s = now - self._tick_start
        self._record("loop", loop_s)
        self._tick_start = None
        self._window_ticks += 1
        self._run_ticks += 1
        if loop_s > self.budget_s:
            self._window_overruns += 1
            self._run_overruns += 1
        if now - self._window_start >= self.interval_s:
            self._log_window(now)

    def _record(self, stage: str, value_s: float) -> None:
        self._window.setdefault(stage, _Stats()).add(value_s)
        run = self._run.setdefault(stage, [0, 0.0, 0.0])
        run[0] += 1
        run[1] += value_s * 1e3
        run[2] = max(run[2], value_s * 1e3)

    def _log_window(self, now: float) -> None:
        elapsed = now - self._window_start
        stages = " | ".join(f"{stage} {self._window[stage].summary()}" for stage in _ordered(self._window))
        logger.info(
            "Latency over %.0fs: %d ticks, %.1f Hz (target %.0f) | %s ms (mean/p99/max) | overruns %d (budget %.1f ms)",
            elapsed,
            self._window_ticks,
            self._window_ticks / elapsed,
            self.fps,
            stages,
            self._window_overruns,
            self.budget_s * 1e3,
        )
        self._window = {}
        self._window_start = now
        self._window_ticks = 0
        self._window_overruns = 0

    def log_run_summary(self) -> None:
        if not self._run_ticks:
            return
        elapsed = time.perf_counter() - self._run_start
        stages = " | ".join(
            f"{stage} {self._run[stage][1] / self._run[stage][0]:.1f}/{self._run[stage][2]:.1f}"
            for stage in _ordered(self._run)
        )
        logger.info(
            "Latency for the whole run (%.0fs): %d ticks, %.1f Hz (target %.0f) | %s ms (mean/max) | "
            "overruns %d (%.1f%%)",
            elapsed,
            self._run_ticks,
            self._run_ticks / elapsed,
            self.fps,
            stages,
            self._run_overruns,
            100 * self._run_overruns / self._run_ticks,
        )
