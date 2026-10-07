# Computer Vision Internship Utilities & Deep Learning Pipeline

Enterprise-grade computer vision dataset engineering and YOLOv8 video analytics suite.
Engineered for fault tolerance, handling edge cases, and supporting high-throughput computer vision workflows.

---

## Quick Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

> **Requirements:**
> - Python 3.8+
> - `opencv-python`
> - `ultralytics` (includes PyTorch & YOLOv8)
> - `tqdm`
> - Optional: **FFmpeg** on `PATH` for video transcode acceleration (automatic OpenCV fallback is provided).

---

## Architecture & Module Overview

```
video-image-practise/
├── dataset_utils.py       # Comprehensive YOLO & Dataset Engineering CLI
├── extract_frames.py      # Video frame extraction (interval, FPS, scene change, count)
├── yolo_video.py          # YOLOv8 Video Analytics (intrusion, loitering, line-crossing, occupancy)
├── auto_annotate.py       # Batch auto-annotation using YOLO model inference
├── yolov8n.pt             # Default YOLOv8 Nano weights
└── requirements.txt       # Python package dependencies
```

---

## Module 1: Video Frame Extraction (`extract_frames.py`)

Extracts frames from a single video or batch processes entire video folders (subsuming `vdo2img.py`).

### Usage Modes

```powershell
# 1. Extract by time interval (1 frame every second)
python extract_frames.py input.mp4 --interval 1 --output frames

# 2. Extract at target FPS (e.g. 0.5 fps = 1 frame every 2 seconds)
python extract_frames.py input.mp4 --fps 0.5 --output frames

# 3. Extract exact number of uniformly distributed frames across video
python extract_frames.py input.mp4 --count 100 --output frames

# 4. Extract within a specific time window
python extract_frames.py input.mp4 --interval 2 --start 00:01:00 --end 00:03:30 --output frames

# 5. Motion / Scene-change filtering (skips redundant static frames to save storage)
python extract_frames.py input.mp4 --interval 1 --scene-threshold 15.0 --output frames

# 6. Batch directory mode (process all videos in a folder)
python extract_frames.py --video-dir path\to\videos --interval 1 --output all_frames
```

### Options & Edge Cases Handled

- `--pattern`: Custom naming pattern, e.g. `{video}_{sec:04d}.jpg` or `frame_{index:06d}.jpg`.
- `--format`: `jpg`, `png`, or `webp`.
- `--resize` / `--scale`: Downscale or resize frames on-the-fly (`--resize 1280x720` or `--scale 0.5`).
- **Resilience:** Handles variable frame rate (VFR) videos, corrupt frames, zero FPS metadata, and non-seekable streams gracefully.

---

## Module 2: Dataset Engineering & YOLO Label Management (`dataset_utils.py`)

Unified CLI supporting both **flat directories** (images and `.txt` in same directory) and **standard hierarchical YOLO structures** (`images/` and `labels/` subdirectories).

### 1. Sequential Renaming with Zero Collisions

Two-phase atomic UUID staging prevents file overwrite even when renaming files in-place:

```powershell
# Rename images and matching YOLO .txt labels simultaneously
python dataset_utils.py rename-pairs path\to\dataset --prefix my_dataset_ --padding 4 --start-idx 1

# Support separate images and labels folders
python dataset_utils.py rename-pairs path\to\dataset\images --labels-dir path\to\dataset\labels

# Dry-run preview and CSV audit manifest export
python dataset_utils.py rename-pairs path\to\dataset --dry-run --manifest rename_log.csv

# Rename video files
python dataset_utils.py rename-videos path\to\videos --prefix cam01_vid_ --padding 3
```

### 2. Dataset Distribution (`distribute`)

Synchronized distribution of images **and matching `.txt` labels** across $N$ folders (for assigning annotation tasks to interns):

```powershell
# Distribute into 10 folders (balanced count)
python dataset_utils.py distribute path\to\dataset --dest path\to\distributed --num-folders 10

# Distribute in fixed batch sizes of 500 pairs per folder
python dataset_utils.py distribute path\to\dataset --dest path\to\distributed --batch-size 500

# With distribution manifest log
python dataset_utils.py distribute path\to\dataset --dest path\to\distributed --num-folders 5 --manifest dist_manifest.csv
```

### 3. Background & Orphan Separation (`separate-bg`)

Non-destructively quarantines unannotated images (background / negative samples) and orphaned `.txt` files into designated folders:

```powershell
python dataset_utils.py separate-bg path\to\dataset --bg-dir path\to\bg_imgs --orphan-dir path\to\orphans
```

### 4. Deep YOLO Annotation Validation (`validate`)

Checks annotation integrity and flags out-of-bounds coordinates, non-integer classes, degenerate boxes, and duplicate detections:

```powershell
# Run validation check with classes.txt and export report
python dataset_utils.py validate path\to\labels --classes classes.txt --report val_issues.csv

# Auto-repair out-of-bounds coordinates and duplicate boxes
python dataset_utils.py validate path\to\labels --classes classes.txt --fix
```

### 5. Dataset Statistics & Insights (`stats`)

Outputs complete breakdown of instances, class balance, bounding box dimensions, and background image ratios:

```powershell
python dataset_utils.py stats path\to\dataset --classes classes.txt
python dataset_utils.py stats path\to\dataset --classes classes.txt --json dataset_summary.json
```

### 6. Train / Validation / Test Splitter (`split`)

Stratified split keeping image and label pairs synchronized, outputting standard YOLO directory layout and auto-generating `data.yaml`:

```powershell
python dataset_utils.py split path\to\dataset --output yolo_split --ratios 0.7 0.2 0.1 --classes classes.txt
```

Generated layout:
```
yolo_split/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### 7. Class Remapping & Filtering (`remap-classes`)

Remap class IDs or filter classes directly across `.txt` annotations:

```powershell
# Remap class 12 to 0, class 5 to 1, and drop all others
python dataset_utils.py remap-classes path\to\labels --mapping "12:0,5:1"
```

### 8. Annotation Visualizer (`visualize`)

Renders bounding boxes and class names onto images to verify annotation quality visually:

```powershell
python dataset_utils.py visualize path\to\dataset --classes classes.txt --samples 25 --output viz_out
```

### 9. Dataset Mismatch Detection (`mismatches`)

```powershell
python dataset_utils.py mismatches path\to\dataset --report mismatch_report.csv
```

### 10. Empty Label Cleaning & Negative Sample Generation

```powershell
# Quarantine or remove empty/whitespace-only txt files
python dataset_utils.py remove-empty-labels path\to\dataset --quarantine path\to\quarantine

# Create empty 0-byte txt files for unannotated images (for YOLO background learning)
python dataset_utils.py create-labels path\to\dataset
```

---

## Module 3: YOLOv8 Video Analytics & Surveillance (`yolo_video.py`)

Production-grade real-time video analytics with ByteTrack tracking, on-screen HUD, threat color coding, and automated audit logging.

### Detection Modes

1. **`all`**: Standard COCO object detection.
2. **`person`**: Filtered detection and tracking for person class (`ID 0`).
3. **`intrusion`**: Polygon ROI breach detection.
4. **`loitering`**: Polygon ROI dwell time monitoring with visual timers and configurable threshold.
5. **`line-crossing`**: Bidirectional virtual tripwire counting (IN / OUT).
6. **`occupancy`**: Real-time headcount and capacity tracking in ROI.

### Examples

```powershell
# 1. Intrusion Detection with Polygon ROI
python yolo_video.py input.mp4 --mode intrusion --roi 100,100 500,100 500,400 100,400

# 2. Loitering Detection with 15s threshold and on-screen active timer HUD
python yolo_video.py input.mp4 --mode loitering --loiter-seconds 15 --roi 100,100 500,100 500,400 100,400

# 3. Virtual Tripwire Line Crossing (IN / OUT counts)
python yolo_video.py input.mp4 --mode line-crossing --line 250,50 250,600

# 4. Save Annotated Video and Record Alert Audit Log to CSV
python yolo_video.py input.mp4 --mode loitering --loiter-seconds 20 \
  --roi 100,100 500,100 500,400 100,400 \
  --save output_annotated.mp4 \
  --alert-log alerts.csv \
  --save-snapshots alert_snapshots

# 5. Headless Server Mode (no display window)
python yolo_video.py input.mp4 --mode intrusion --roi 100,100 500,100 500,400 100,400 --no-display --alert-log alerts.json

# 6. Live Webcam or RTSP Stream
python yolo_video.py 0 --mode person
python yolo_video.py rtsp://user:pass@192.168.1.10:554/live --mode intrusion --roi 100,100 500,100 500,400 100,400
```

---

## Module 4: Automated Batch YOLO Annotation (`auto_annotate.py`)

Runs model inference on raw image directories to auto-generate YOLO `.txt` labels:

```powershell
python auto_annotate.py path\to\raw_images --output-labels path\to\labels --weights yolov8n.pt --confidence 0.5 --classes-file classes.txt
```

---

## Edge Case Summary

| Category | Edge Case Handled | How It Is Resolved |
|---|---|---|
| **File I/O** | Destination name collision | Two-phase UUID staging with atomic renames and automatic rollback on failure. |
| **Data Safety** | Accidental file deletion | Safe move/quarantine options (`separate-bg`, `--quarantine`) replacing destructive removal. |
| **Dataset Balance** | Annotation loss during distribution | Synchronized multi-folder distribution preserving image-label matching pairs and manifest log. |
| **YOLO Formatting** | Malformed coordinates & labels | Strict validator checking 5 columns, $[0.0, 1.0]$ bounds, NaN/Inf, zero-area, and auto-clamping via `--fix`. |
| **Duplicates** | Overlapping duplicate boxes | IoU calculation (> 0.95 threshold) to remove redundant duplicate annotations. |
| **Negative Samples** | False positive training errors | Automatic 0-byte `.txt` generation for background images to train YOLO negative samples. |
| **Video Streams** | Non-standard/VFR video codecs | OpenCV fallback when FFmpeg is not installed; timestamp-based frame sampling. |
| **Analytics HUD** | Missing threat awareness | Real-time on-screen dwell timers (`[ID: 1] 14.5s/30.0s`) with Green $\to$ Yellow $\to$ Red color changes. |
| **Audit Trails** | Untracked security events | Full event logging (CSV/JSON) with timestamp, track ID, coordinates, duration, and snapshot crops. |
