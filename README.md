# Traffic Counter: Vehicle Detection, Tracking and Counting with YOLO11

A computer vision project that detects vehicles in recorded traffic video, tracks each one with a persistent ID, and counts how many cross a virtual line, separated by direction (in / out).

Built with **Ultralytics YOLO11**, **ByteTrack**, **Supervision**, and **OpenCV**.

![Demo](outputs/demo.gif)

> Replace the image above with your own GIF or screenshot of the output video. If the video is large, upload it to Google Drive or YouTube and link it here: [Watch the full demo](YOUR_VIDEO_LINK)

---

## How it works

Detection alone cannot count vehicles, because the same car appears in hundreds of frames. This project uses a three-stage pipeline:

1. **Detect:** YOLO11 finds vehicles in every frame (car, motorcycle, bus, truck).
2. **Track:** ByteTrack assigns each vehicle a stable ID across frames, so it is counted only once.
3. **Count:** A virtual line is drawn across the road. When a tracked vehicle's center crosses it, the counter increases. The crossing direction decides whether it counts as "in" or "out".

```
Video frame -> YOLO11 detection -> ByteTrack IDs -> Line-crossing check -> Counts + annotated video + CSV
```

## Features

- Vehicle detection for 4 COCO classes: car (2), motorcycle (3), bus (5), truck (7)
- Persistent object tracking with ByteTrack
- Directional line-crossing counter
- Annotated output video with boxes, IDs, the counting line, and live totals
- Per-frame count log exported to CSV
- Chart of cumulative vehicle count over time
- GPU acceleration on Google Colab (Tesla T4), with CPU fallback for local use

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Language |
| Ultralytics (YOLO11) | Detection and tracking |
| Supervision | Line zone counting and annotations |
| OpenCV | Video reading and writing |
| Pandas | CSV logging |
| Matplotlib | Charts |

## Project structure

```
traffic-counter/
├── traffic_counter.ipynb    # Main notebook (runs on Google Colab)
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
    ├── counts.csv           # Count log over time
    ├── chart.png            # Cumulative count chart
    └── demo.gif             # Preview of the output
```

## Getting started

### Option 1: Google Colab (recommended, free GPU)

1. Open `traffic_counter.ipynb` in [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Install the dependencies:
   ```python
   !pip install ultralytics supervision
   ```
4. Upload your video to the Files panel (or mount Google Drive) and set the `VIDEO` path in the notebook.
5. Run all cells.
6. Download `output_counted.mp4` and `counts.csv` from the Files panel.

### Option 2: Run locally

```bash
# Clone the repository
git clone https://github.com/eshitab19/traffic-counter.git
cd traffic-counter

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

Then set `VIDEO` to your video path and run the script or notebook.

**No NVIDIA GPU?** The project still runs on CPU. In the code, use `device="cpu"`, switch to the `yolo11n.pt` model, and set `imgsz=640`. Test on short clips, and use Colab for full-quality runs.

## Configuration

| Setting | Where | What it does |
|---|---|---|
| `yolo11m.pt` | `YOLO(...)` | Model size. Options: `n`, `s`, `m`, `l`, `x`. Larger is more accurate but slower |
| `imgsz=1280` | `model.track(...)` | Input resolution. Higher helps detect small, distant vehicles |
| `conf=0.35` | `model.track(...)` | Minimum confidence. Raise to reduce false detections, lower to catch more vehicles |
| `classes=[2, 3, 5, 7]` | `model.track(...)` | Vehicle classes to detect |
| `tracker="bytetrack.yaml"` | `model.track(...)` | Tracking algorithm |
| `0.6` in `LineZone` | Line definition | Vertical position of the counting line (fraction of frame height) |

Place the counting line where vehicles are large and clearly separated. Avoid the far distance, where boxes are tiny and merge. If "in" and "out" look swapped, swap the line's `start` and `end` points.

## Results

**Test video:** `sample_video.mp4` (736x1280, 601 frames)

| Vehicle type | Manual count | Detected count | Accuracy |
|---|---|---|---|
| Car | _fill in_ | _fill in_ | _fill in_ |
| Motorcycle | _fill in_ | _fill in_ | _fill in_ |
| Bus | _fill in_ | _fill in_ | _fill in_ |
| Truck | _fill in_ | _fill in_ | _fill in_ |
| **Total** | _fill in_ | _fill in_ | _fill in_ |

**Model comparison (optional):**

| Model | Image size | Total detected | Processing time |
|---|---|---|---|
| yolo11n | 640 | _fill in_ | _fill in_ |
| yolo11m | 1280 | _fill in_ | _fill in_ |

![Cumulative count chart](outputs/chart.png)

## Limitations

- **Occlusion:** when vehicles overlap, a tracker can assign a new ID to the same vehicle and count it twice.
- **Camera angle:** results are best with a fixed, elevated camera looking along the road.
- **Lighting and weather:** night footage, glare, and rain reduce detection accuracy.
- **Small objects:** vehicles far from the camera may be missed at lower resolution.
- **Single line:** the count depends on line placement, so it needs tuning for each video.

## Future improvements

- Polygon-zone counting for more robust results
- Vehicle speed estimation using pixel-to-meter calibration
- Per-minute traffic flow statistics
- Streamlit dashboard for video upload and live stats
- Custom-trained model for local vehicle types (for example auto-rickshaws)

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [Supervision by Roboflow](https://github.com/roboflow/supervision)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- Test video source: _add the source or credit here_

## License

This project is for educational purposes. Note that Ultralytics YOLO is released under the AGPL-3.0 license; check its terms if you plan to use this commercially.

## Author

**Your Name**
GitHub: [@eshitab19](https://github.com/eshitab19)
