"""
Validation of bubble_detection.py on simulated camera frames.

Generates frames of a tube with bubbles flowing through it and runs
bubble_detection.py on them unchanged.

Usage: python3 validation.py

Stop with Ctrl+C (or q in the viewer window), a summary follows.

Why: with a real camera nobody knows how many bubbles actually passed, and
the available camera cannot resolve the bubbles in a 1.3 mm tube. The
simulation therefore tests the software on its own.

Not tested: whether the image model matches reality. Illumination, bubble
shape, refraction at the tube wall etc. can only be assessed on the real
setup.

Image scale, frame size, frame rate, finish line etc. are read directly
from bubble_detection.py, so exactly the configuration that runs on the
Jetson is tested. Bubbles are simulated as isolated, round and with high
contrast.
"""

import math
import pathlib
import re
import signal
import sys
import time

import cv2
import numpy as np


# Settings. Change these when the setup changes.

TARGET_SCRIPT = "bubble_detection.py"

# Read the upper-case constants from the target script. Importing it is not
# an option because it opens the camera as soon as it is loaded.
SOURCE = pathlib.Path(__file__).with_name(TARGET_SCRIPT).read_text(encoding="utf-8")
CFG = {"np": np}
for line in re.findall(r"^[A-Z0-9_]+ = .*$", SOURCE, re.M):
    exec(line, CFG)
FPS = CFG["FPS_RECORD"]

TUBE_OUTER_MM = 1.3
TUBE_INNER_MM = 1.0
BUBBLE_MIN_MM = 0.4
BUBBLE_MAX_MM = 0.9
FLOW_SPEED_MM_S = 0.2

# Gray levels for backlighting. 40 vs. 225 gives 185 levels of contrast,
# as expected with good illumination. Noise and blur are not simulated.
BACKGROUND_GRAY = 40
BUBBLE_GRAY = 225


# Simulation

class Simulation:
    """
    Generates the frames.

    The scene is described in mm and only converted to pixels when drawing,
    so bubble positions follow x = x0 + v * t without rounding drift.
    """

    def __init__(self):
        self.mm_per_px = CFG["MM_PER_PIXEL"]
        self.px_per_mm = 1.0 / self.mm_per_px
        self.finish_line_px = CFG["FINISH_LINE_X"]
        self.finish_line_mm = self.finish_line_px * self.mm_per_px

        self.in_flight = []  # tuples (start position in mm, radius in mm)
        self.passed = []  # radii of bubbles that crossed the finish line
        self.created = 0
        self._last_start = None
        self._t_prev = 0.0

    def _spawn(self, t_s):
        """
        Creates new bubbles left of the image as soon as there is room.
        """
        # Largest bubble is 90 % of the inner diameter, so it never touches the wall
        r_max = TUBE_INNER_MM / 2 * 0.9
        entry = -2.0 * r_max  # start two radii left of the image so a bubble enters completely
        while len(self.in_flight) < 100:  # safety limit, never reached in practice
            if self._last_start is not None:
                if self._last_start + FLOW_SPEED_MM_S * t_s < entry:
                    break  # no room yet
            r = min(np.random.uniform(BUBBLE_MIN_MM, BUBBLE_MAX_MM) / 2, r_max)
            start = entry - FLOW_SPEED_MM_S * t_s
            self.in_flight.append((start, r))
            self.created += 1
            # Gap: a minimum of 2.5 radii so that two bubbles never touch,
            # plus a random part for irregular spacing
            gap = 2.5 * r + np.random.uniform(0.5 * r_max, 3.0 * r_max)
            self._last_start = start - gap

    def _check_finish_line(self, t_s):
        """
        A bubble counts as passed once its center crosses the finish line
        between the previous and the current frame.
        """
        for start, r in self.in_flight:
            if (start + FLOW_SPEED_MM_S * self._t_prev < self.finish_line_mm <= start + FLOW_SPEED_MM_S * t_s):
                self.passed.append(r)
        self._t_prev = t_s

    def frame(self, index):
        """Returns the next camera frame."""
        width, height = CFG["FRAME_WIDTH"], CFG["FRAME_HEIGHT"]
        img = np.full((height, width), BACKGROUND_GRAY, np.uint8)
        center = height // 2
        outer = int(TUBE_OUTER_MM / 2 * self.px_per_mm)
        inner = int(TUBE_INNER_MM / 2 * self.px_per_mm)
        img[center - outer:center - inner] = 10
        img[center + inner:center + outer] = 10

        # The first frames stay empty, the target script builds its
        # reference image from them
        ref = CFG["REFERENCE_FRAME_COUNT"]
        if index >= ref:
            t_s = (index - ref) / FPS
            self._spawn(t_s)
            edge_mm = width * self.mm_per_px + 1.0
            for start, r in self.in_flight:
                x_mm = start + FLOW_SPEED_MM_S * t_s
                cx = int(round(x_mm * self.px_per_mm))
                rp = int(round(r * self.px_per_mm))
                if -rp < cx < width + rp and rp >= 1:
                    # Bubble: bright disc with a slightly darker core,
                    # both well above the background
                    cv2.circle(img, (cx, center), rp, BUBBLE_GRAY, -1, cv2.LINE_AA)
                    cv2.circle(img, (cx, center), rp - max(1, int(0.10 * rp)), BUBBLE_GRAY - 40, -1, cv2.LINE_AA)
            self._check_finish_line(t_s)
            self.in_flight = [(s, r) for s, r in self.in_flight if s + FLOW_SPEED_MM_S * t_s < edge_mm]

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# Fake camera, clock and stop handling

# Lists so they can be changed in place, also from viewer.py
_stop = [False]
_clock = [0.0]


def _on_ctrl_c(signum, frame):
    """
    The first Ctrl+C only sets a flag: the fake camera then reports
    "no frame", the target script leaves its loop normally and prints its
    result. A second Ctrl+C aborts immediately.
    """
    if _stop[0]:
        raise KeyboardInterrupt
    _stop[0] = True
    print("\n  [Ctrl+C, stopping capture, summary follows]\n", flush=True)


class FakeCamera:
    """Replaces cv2.VideoCapture and returns the synthetic frames."""

    def __init__(self, sim):
        self.sim = sim
        self.i = 0

    def read(self):
        if _stop[0]:
            return False, None
        frame = self.sim.frame(self.i)
        self.i += 1
        # Simulated clock in seconds: every frame advances time by exactly
        # 1 / FPS, independent of how fast this machine runs
        _clock[0] = self.i / FPS
        return True, frame

    def isOpened(self):
        return True

    def release(self):
        pass


class NullWriter:
    """
    Replaces cv2.VideoWriter and discards all frames.
    viewer.py swaps this class for a display window.
    """

    def write(self, frame):
        pass

    def release(self):
        pass


class OutputRecorder:
    """Passes console output through and keeps a copy for the summary."""

    def __init__(self, real):
        self.real = real
        self.text = []

    def write(self, s):
        self.real.write(s)
        self.text.append(s)
        return len(s)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


# Running the target script

def run_target(sim):
    """
    Runs the target script unchanged. Only the camera, the clock and the
    video writer are replaced, and restored afterwards.
    """
    camera = FakeCamera(sim)
    saved = (time.time, cv2.VideoCapture, cv2.VideoWriter)
    time.time = lambda: _clock[0]
    cv2.VideoCapture = lambda *a, **k: camera
    cv2.VideoWriter = lambda *a, **k: NullWriter()
    try:
        namespace = {"__name__": "__main__"}
        exec(compile(SOURCE, TARGET_SCRIPT, "exec"), namespace)
        return namespace
    finally:
        time.time, cv2.VideoCapture, cv2.VideoWriter = saved


# Summary

# Parses the per-bubble console lines to collect the measured speeds
COUNT_LINE = re.compile(r"Bubble #\s*\d+.*?v=(-?[\d.]+)")


def report(sim, result, output):
    expected_n = len(sim.passed)
    expected_v = sum(4 / 3 * math.pi * r ** 3 for r in sim.passed)  # sphere volume of every passed bubble
    measured_n = result.get("bubble_count", 0)
    measured_v = result.get("total_volume_ul", 0.0)
    speeds = [float(x) for x in COUNT_LINE.findall(output)]

    def deviation(measured, expected):
        return (measured - expected) / expected * 100 if expected else float("nan")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("expected measured deviation")
    print("-" * 70)
    print(f"Bubbles {expected_n} {measured_n} {deviation(measured_n, expected_n):.1f} %")
    print(f"Gas volume [uL] {expected_v:.4f} {measured_v:.4f} {deviation(measured_v, expected_v):.1f} %")
    if speeds:
        m = float(np.mean(speeds))
        print(f"Speed [mm/s] {FLOW_SPEED_MM_S:.3f} {m:.3f} {deviation(m, FLOW_SPEED_MM_S):.1f} %")
    print("-" * 70)
    print(f"  {sim.created} bubbles generated, {expected_n} of them crossed the finish line (the rest are still in flight)")
    print()

    if expected_n == 0:
        print("  Stopped too early, no bubble has reached the finish line yet.")
    elif measured_n < expected_n:
        print(f"{expected_n - measured_n} bubble(s) missed. Check: lost tracks, size filter, threshold.")
    elif measured_n > expected_n:
        print(f"{measured_n - expected_n} bubble(s) too many. Check: double counting at the finish line.")
    else:
        print("Count matches exactly.")
    print("=" * 70)


def main():
    sim = Simulation()
    print("=" * 70)
    print("BUBBLE DETECTION VALIDATION")
    print("=" * 70)
    print(f"Script: {TARGET_SCRIPT}")
    print(f"Scale: {sim.mm_per_px} mm/px, tube {TUBE_OUTER_MM} mm = {TUBE_OUTER_MM * sim.px_per_mm:.0f} px")
    print(f"Field of view: {CFG['FRAME_WIDTH'] * sim.mm_per_px:.2f} mm of tube length")
    print(f"Bubbles: {BUBBLE_MIN_MM}..{BUBBLE_MAX_MM} mm diameter, random size and spacing")
    print(f"Flow: {FLOW_SPEED_MM_S} mm/s -> {FLOW_SPEED_MM_S / FPS * sim.px_per_mm:.1f} px per frame")
    print(f"Finish line: x = {sim.finish_line_px} px")
    print("Running. Stop with Ctrl+C (or q in the viewer window) to see the summary.")
    print("=" * 70)

    # Record the target script's console output to extract the speeds
    signal.signal(signal.SIGINT, _on_ctrl_c)
    real_stdout = sys.stdout
    recorder = OutputRecorder(real_stdout)
    sys.stdout = recorder
    try:
        result = run_target(sim)
    finally:
        sys.stdout = real_stdout
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    report(sim, result, "".join(recorder.text))


if __name__ == "__main__":
    main()
