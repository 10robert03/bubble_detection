# Camera-Based Bubble Detection in a Capillary Tube

Real-time detection, counting, volume and speed measurement of gas bubbles
in a tube (1.3 mm outer diameter) on an NVIDIA Jetson Nano.

This is a university project from my bachelors degree. It is a
feasibility study: the software pipeline is complete and validated on
simulated data, while the optical setup needed for real measurements has
not been built yet.

## What it does

For every camera frame the script

1. compares the frame with a reference image of the empty tube,
2. turns the difference into a binary mask and removes noise,
3. finds the contour of every bubble and fits an ellipse to it,
4. tracks each bubble from frame to frame,
5. counts it once its center crosses a finish line and computes its
   volume and speed.

The annotated video is saved to `bubble_recording.avi` and every counted
bubble is printed to the console.

## Hardware and software

- NVIDIA Jetson Nano Developer Kit
- Raspberry Pi Camera Module v2.1, CSI connection
- JetPack with its preinstalled OpenCV 4.1.1 (with GStreamer support),
  Python 3.6.9, NumPy 1.13.3

## Usage

### On the Jetson Nano

```
python3 bubble_detection.py
```

Keep the tube free of bubbles for the first two seconds while the
reference image is captured.

The script needs an OpenCV build with GStreamer support to open the CSI camera.

### Validation on any computer

```
pip install opencv-python numpy
python3 validation.py   # console only
python3 viewer.py       # with a live window, stop with q
```

All three files must be in the same folder.

## How it works

**Reference image instead of frame differencing.** Bubbles move only a few
pixels per frame, so the difference between two consecutive frames would
only show thin crescents at the bubble edges. Comparing against a fixed
image of the empty tube shows the whole bubble.

**Median of 15 frames.** The reference image is the per-pixel median of 15
frames. The median suppresses sensor noise and ignores a bubble that
appears in only a few of those frames.

**Absolute difference.** In backlight a bubble has both brighter and darker
regions than the liquid. `cv2.absdiff` captures both.

**Volume model.** Each bubble is modelled as an ellipsoid of revolution
around the tube axis: V = 4/3 * pi * r_radial^2 * r_axial. The camera cannot
see the depth of a bubble, so the depth is assumed equal to the visible
radial axis.

**Speed.** Measured over the whole track of a bubble instead of between two
frames, because at flow speeds of mm/min a bubble often moves less than one
pixel per frame.

**No CUDA.** OpenCV offers CUDA versions of the preprocessing steps but not
of the contour and ellipse functions, so every frame would have to return
to the CPU anyway. At the target flow speeds 7.6 fps is already more than
sufficient, so CUDA would be an over kill.

## Validation

`validation.py` generates frames with bubbles of known size, position and
speed, then runs `bubble_detection.py` on them unchanged. Only the camera,
the system clock and the video writer are replaced.

All configuration values are read from `bubble_detection.py`, so 
the validation always tests the exact configuration that runs on the Jetson.

The validation checks the software, not the physics: illumination,
refraction at the tube wall and the real bubble shape can only be assessed
on the real setup.

## Author
Robert Deines
