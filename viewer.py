"""
Live view of the validation run.

Runs validation.py and shows every frame produced by
bubble_detection.py in a window instead of discarding it.

Usage: python3 viewer.py

Stop with q in the window. The summary is printed afterwards.
"""

import os
import sys
import cv2
import validation as V

# Delay per frame in ms
DISPLAY_DELAY_MS = 60


class DisplayWriter:
    """Replaces the video writer: frames are displayed instead of saved."""

    def write(self, frame):
        # Scaled to 80 % so the window fits on a laptop screen
        cv2.imshow("Bubble detection", cv2.resize(frame, None, fx=0.8, fy=0.8))
        if cv2.waitKey(DISPLAY_DELAY_MS) == ord("q"):
            V._stop[0] = True

    def release(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # cv2.imshow crashes without a display (e.g. over SSH on the Jetson)
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        sys.exit("No display found. Use validation.py instead.")
    V.NullWriter = DisplayWriter
    V.main()
