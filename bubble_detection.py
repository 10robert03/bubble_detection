"""
Bubble detection in a capillary tube (outer diameter 1.3 mm).

Runs on an NVIDIA Jetson Nano with a CSI camera. For every frame:

    difference to a reference image of the empty tube
    -> binary mask -> bubble contours -> ellipse fit
    -> frame-to-frame tracking
    -> count, volume and speed once a bubble crosses the finish line

Usage:
    python3 bubble_detection.py      (stop with Ctrl+C)
"""

import cv2
import numpy as np
import time

# Capture resolution. 1280x720 is one of the sensor modes of the IMX219
# camera module (native resolution: 3264x2464).
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Image scale in mm per pixel, obtained by measuring an object of known size.
# Placeholder: assumes the 1.3 mm tube spans 80 px (1.3 / 80). Must be
# recalibrated on the final optical setup.
MM_PER_PIXEL = 0.01625

# Frame rate the Jetson actually processes, not the 60 fps camera rate.
# Measured at 7.6 fps (not under full load), rounded down to 7.
# Used as the playback rate of the recorded video.
FPS_RECORD = 7

# Contour area limits in px^2.
# The lower limit rejects noise that survives the morphological opening
# (estimated). The upper limit corresponds to a bubble of about 1 mm diameter.
MIN_BUBBLE_AREA_PX = 475  # ~0.2 mm radius
MAX_BUBBLE_AREA_PX = 3500  # ~0.54 mm radius

# Maximum distance a bubble may move between two frames and still be matched
# to its previous position. 30 px at 7 fps covers speeds up to ~3.4 mm/s.
# Adjust once the real flow speed is known.
MAX_MATCH_DIST_PX = 30

# Camera, tube and illumination are fixed, so one reference image taken at
# start-up is sufficient. The per-pixel median of 15 frames pushes sensor
# noise below one gray level and takes only about 2 s at 7 fps.
REFERENCE_FRAME_COUNT = 15

# Minimum gray-level difference to the reference image for a pixel to be
# treated as bubble. Assumes ~3 gray levels of sensor noise (to be measured
# on the real sensor), so 25 is about eight times the noise level.
DIFF_THRESHOLD_GRAYVALUE = 25


def px_to_mm(px):
    return px * MM_PER_PIXEL


def ellipsoid_volume_ul(radial_mm, axial_mm):
    """
    Bubble volume in µL (= mm^3), modelled as an ellipsoid of revolution
    around the tube axis. For radial == axial this is the sphere formula.

    The depth axis is invisible to the camera. It is assumed to equal the
    radial axis because the round tube has no preferred transverse direction.
    """
    return (4 / 3) * np.pi * (radial_mm ** 2) * axial_mm


def gstreamer_pipeline(w=FRAME_WIDTH, h=FRAME_HEIGHT, fps=60):
    """
    GStreamer pipeline for the Jetson CSI camera.
    """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        % (w, h, fps)
    )


def extract_bubble_contours(binary_frame):
    """
    Returns the outer contour of every bubble in the binary mask whose area
    lies within the bubble size limits.
    """
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if MIN_BUBBLE_AREA_PX <= cv2.contourArea(c) <= MAX_BUBBLE_AREA_PX]


def split_axes_by_flow_direction(width, height, angle_deg):
    """
    Assigns the two axes returned by cv2.fitEllipse to their physical
    direction and returns them as semi-axes (radial_px, axial_px):

        axial  = along the flow (horizontal in the image)
        radial = across the flow
    """
    theta = np.deg2rad(angle_deg)

    if abs(np.cos(theta)) >= abs(np.sin(theta)):
        axial_px = width / 2
        radial_px = height / 2
    else:
        axial_px = height / 2
        radial_px = width / 2

    return radial_px, axial_px


def fit_bubble_ellipse(contour):
    """
    Fits an ellipse to a contour and returns (cx, cy, radial_px, axial_px).

    cv2.fitEllipse needs at least 5 points, since an ellipse has five
    parameters. Fewer points only occur for shapes with straight edges such
    as rectangles. For those the minimum enclosing circle is used instead,
    which overestimates the size but keeps the script running.
    """
    if len(contour) >= 5:
        (cx, cy), (width, height), angle_deg = cv2.fitEllipse(contour)
        radial_px, axial_px = split_axes_by_flow_direction(width, height, angle_deg)
        return cx, cy, radial_px, axial_px
    else:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        return cx, cy, radius, radius  # circle: radial == axial


def capture_reference_background(cap, n_frames):
    """
    Builds a reference image of the empty tube as the per-pixel median of
    n_frames grayscale frames. The tube should be as empty as possible
    while this runs.
    """
    frames = []
    read_fails = 0  # give up after 100 failed reads instead of looping forever
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            read_fails += 1
            if read_fails > 100:
                raise RuntimeError("Camera delivers no frames (100 failed reads).")
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # np.median returns float64, convert back to an 8-bit image
    return np.median(frames, axis=0).astype(np.uint8)


def match_bubbles(current, tracked, max_dist=MAX_MATCH_DIST_PX):
    """
    Matches the bubbles of the current frame to those of the previous frame
    so that every bubble keeps its ID.
    current: list of (cx, cy, radial, axial)
    tracked: {id: (x, y, radial, axial)} from the previous frame
    returns: {index in current: id, or None for a new bubble}
    """

    if not tracked or not current:
        return {i: None for i in range(len(current))}

    ids = list(tracked.keys())

    # Start with infinity so an unfilled entry can never be the closest match
    dists = np.full((len(current), len(ids)), np.inf)
    for ci, (cx, cy, _, _) in enumerate(current):
        for ti, bid in enumerate(ids):
            lx, ly, _, _ = tracked[bid]
            dists[ci, ti] = np.hypot(cx - lx, cy - ly)

    # All (current, tracked) index pairs, sorted by ascending distance
    order = np.dstack(np.unravel_index(np.argsort(dists.ravel()), dists.shape))[0]
    assignment = {}
    used_tracks = set()

    for ci, ti in order:
        if ci in assignment:
            continue
        if ti in used_tracks:
            continue
        if dists[ci, ti] > max_dist:
            break  # all remaining pairs are even further apart
        assignment[ci] = ids[ti]
        used_tracks.add(ti)

    for ci in range(len(current)):
        if ci not in assignment:
            assignment[ci] = None

    return assignment


# Camera and video output
pipeline_string = gstreamer_pipeline()
cap = cv2.VideoCapture(pipeline_string, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the camera pipeline.")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
OUTPUT_VIDEO_PATH = "bubble_recording.avi"
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS_RECORD, (FRAME_WIDTH, FRAME_HEIGHT))

print(f"Capturing reference image ({REFERENCE_FRAME_COUNT} frames, the tube should be empty now)")
background_reference = capture_reference_background(cap, REFERENCE_FRAME_COUNT)
print("Reference image captured.\n")

# Bubbles flow from left to right, so the finish line is vertical.
# A bubble is counted once its center crosses this line.
FINISH_LINE_X = FRAME_WIDTH - 200

tracked_bubbles = {}  # {id: (x, y, radial_px, axial_px)}, latest state of every bubble
bubble_history = {}  # {id: [(x, y, radial_px, axial_px, t), ...]}, used for the speed
next_id = 0
bubble_count = 0
counted_bubble_ids = set()
total_volume_ul = 0.0

print("Bubble detection started. Stop with Ctrl+C.\n")
print(f"Calibration: {MM_PER_PIXEL} mm/px\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        # Segmentation: everything that differs enough from the empty tube
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background_reference)
        _, foreground_mask = cv2.threshold(diff, DIFF_THRESHOLD_GRAYVALUE, 255, cv2.THRESH_BINARY)

        # Morphological opening removes isolated noise pixels. 3x3 is the
        # smallest kernel that covers all eight neighbours.
        kernel = np.ones((3, 3), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

        # Detection
        contours = extract_bubble_contours(foreground_mask)

        current_frame_bubbles = []
        for c in contours:
            cx, cy, radial_px, axial_px = fit_bubble_ellipse(c)
            current_frame_bubbles.append((cx, cy, radial_px, axial_px))

        # Tracking and counting
        assignment = match_bubbles(current_frame_bubbles, tracked_bubbles)
        new_tracked = {}
        for ci, (cx, cy, radial_px, axial_px) in enumerate(current_frame_bubbles):
            bid = assignment[ci]
            if bid is not None:
                prev_x = tracked_bubbles[bid][0]
                if (prev_x < FINISH_LINE_X and cx >= FINISH_LINE_X
                        and bid not in counted_bubble_ids):
                    counted_bubble_ids.add(bid)
                    bubble_count += 1
                    radial_mm = px_to_mm(radial_px)
                    axial_mm = px_to_mm(axial_px)
                    vol_ul = ellipsoid_volume_ul(radial_mm, axial_mm)
                    total_volume_ul += vol_ul

                    # Speed over the whole track instead of between two
                    # frames: at mm/min a bubble often moves less than
                    # 1 px per frame.
                    x0, y0, _, _, t0 = bubble_history[bid][0]
                    dt = now - t0
                    speed_mm_s = px_to_mm(np.hypot(cx - x0, cy - y0)) / dt if dt > 0 else 0.0

                    print(
                        f"Bubble #{bid} counted | "
                        f"radial={radial_mm:.3f} mm axial={axial_mm:.3f} mm | "
                        f"V={vol_ul:.4f} µL | "
                        f"v={speed_mm_s:.2f} mm/s"
                    )

                new_tracked[bid] = (cx, cy, radial_px, axial_px)
                bubble_history.setdefault(bid, []).append((cx, cy, radial_px, axial_px, now))

            else:
                # New bubbles are only accepted left of the finish line
                if cx < FINISH_LINE_X:
                    new_tracked[next_id] = (cx, cy, radial_px, axial_px)
                    bubble_history[next_id] = [(cx, cy, radial_px, axial_px, now)]
                    next_id += 1

        tracked_bubbles = new_tracked

        # Drop the state of bubbles that are no longer tracked so memory does
        # not grow during long runs. IDs are never reused
        bubble_history = {bid: h for bid, h in bubble_history.items() if bid in tracked_bubbles}
        counted_bubble_ids &= tracked_bubbles.keys()

        # Annotate the frame
        for bid, (x, y, radial_px, axial_px) in tracked_bubbles.items():
            radial_mm = px_to_mm(radial_px)
            axial_mm = px_to_mm(axial_px)
            vol_ul = ellipsoid_volume_ul(radial_mm, axial_mm)
            cv2.ellipse(frame, (int(x), int(y)), (int(axial_px), int(radial_px)), 0, 0, 360, (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)
            cv2.putText(frame, f"ID:{bid} {vol_ul:.3f}uL", (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,)

        cv2.line(frame, (FINISH_LINE_X, 0), (FINISH_LINE_X, FRAME_HEIGHT), (255, 0, 0), 2)
        cv2.putText(frame, "Finish", (FINISH_LINE_X + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.putText(frame, f"Bubbles: {bubble_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Volume: {total_volume_ul:.4f} uL", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        video_writer.write(frame)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    cap.release()
    video_writer.release()
    print("\nResult:")
    print(f"Bubbles counted: {bubble_count}")
    print(f"Total volume   : {total_volume_ul:.6f} µL")
    print(f"Video saved to : {OUTPUT_VIDEO_PATH}")
    print("Camera released.")
