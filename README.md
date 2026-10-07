# Enterprise Media Processing & Computer Vision Pipeline

A production-grade, enterprise media-processing and computer vision platform engineered with **FastAPI / Flask + Vanilla CSS Dark Modern UI + FFmpeg + OpenCV + YOLOv8**.

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Primary Workflow: "How Many Images Do You Want?"](#3-primary-workflow-how-many-images-do-you-want)
- [4. Frame Interpolation & Two Cases](#4-frame-interpolation--two-cases)
- [5. System Architecture & Folder Structure](#5-system-architecture--folder-structure)
- [6. Requirements & Prerequisites](#6-requirements--prerequisites)
- [7. Installation & Setup](#7-installation--setup)
- [8. Configuration & Environment Variables](#8-configuration--environment-variables)
- [9. How to Run Locally](#9-how-to-run-locally)
- [10. API Specification](#10-api-specification)
- [11. CLI Usage](#11-cli-usage)
- [12. Memory & Performance Considerations](#12-memory--performance-considerations)
- [13. Security & Validation](#13-security--validation)
- [14. Known Limitations & Future Improvements](#14-known-limitations--future-improvements)
- [15. Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

The primary utility of this platform is **Task 1: Video &rarr; Images**. It enables both non-technical users and machine learning engineers to convert single or multiple videos into an exact number of image frames without understanding complex video-processing math or frame-rate conversions.

A non-technical user simply asks:
> **"How many images do you want?"** (e.g. `2500`)

The application automatically reads the video metadata, calculates the required effective frame rate, determines whether interpolation is required, synthesizes missing frames if necessary, and guarantees that the output contains **exactly** the requested number of images.

---

## 2. Key Features

- **Intuitive Target Image Count Interface:** Single input with smart preset chips (`30`, `100`, `500`, `1000`, `2500`, Custom positive integer).
- **Exact Count Guarantee:** Always outputs exactly the requested number of images (never under-produces or leaves trailing frames).
- **Intelligent Case Handling:**
  - **Case A (`Requested <= Native Frames`):** Uniform timeline sampling across the video (0 interpolated frames).
  - **Case B (`Requested > Native Frames`):** Hardware-accelerated FFmpeg `minterpolate` filter with linear frame blending fallback.
- **Transparent Frame Attribution:** Clear UI distinction between original frames and synthetic interpolated frames (e.g., `300 original + 2200 interpolated = 2500 images`) with artifact disclaimer.
- **Multi-Video Batch Processing:** Upload multiple videos simultaneously; each video is processed into isolated subdirectories without filename collisions.
- **Paginated Image Gallery:** High-throughput progressive thumbnail loading (50 per page) preventing browser memory crashes when browsing 2500+ images.
- **Streaming & Low Memory Footprint:** Chunked file uploads, disk-streamed ZIP archiving (never loads full videos or thousands of images into RAM).
- **Sanitized Validation & User-Friendly Errors:** OpenCV deep header and stream verification; raw Python exceptions/tracebacks are masked from the UI.
- **Customization Options:** Image format selection (`JPEG`, `PNG`), JPEG compression quality slider, and non-distorting aspect-ratio preserved downscaling.

---

## 3. Primary Workflow: "How Many Images Do You Want?"

```
+-------------------------------------------------------------+
|                HOW MANY IMAGES DO YOU WANT?                 |
|                            [ 2500 ]                         |
|   [ 30 ]    [ 100 ]    [ 500 ]    [ 1000 ]    [ 2500 ]      |
+-------------------------------------------------------------+
| Video Duration: 10.0 sec  | Original FPS: 30 FPS            |
| Original Frames: 300      | Effective FPS: 250 FPS          |
+-------------------------------------------------------------+
| [!] Interpolation Required                                  |
| 300 original frames + 2200 interpolated frames = 2500 total |
| "Interpolated frames are generated between original video   |
| frames and may contain visual artifacts."                   |
+-------------------------------------------------------------+
```

1. **Upload:** User drops one or more videos.
2. **Instant Metadata Extraction:** Video duration, native FPS, resolution, and total frame count are inspected via OpenCV.
3. **Target Selection:** User specifies the exact image count desired.
4. **Live Strategy Feedback:** The system immediately informs the user whether native sampling or synthetic interpolation will be utilized.
5. **Execution:** Background asynchronous processing with real-time percentage progress and stage description.
6. **Delivery:** Paginated interactive gallery preview and instant ZIP download (individual or batch archive).

---

## 4. Frame Interpolation & Two Cases

### Case A: Requested Images $\le$ Available Source Frames
* **Example:** 10-second video at 30 FPS = ~300 frames. User requests 100 images.
* **Strategy:** Native uniform frame sampling.
* **Execution:** Frames are extracted uniformly at time indices $t_i = i 	imes rac{	ext{Duration}}{N - 1}$.
* **Result:** Exactly 100 original frames generated, 0 interpolated frames.

### Case B: Requested Images $>$ Available Source Frames
* **Example:** 10-second video at 30 FPS = ~300 frames. User requests 2500 images.
* **Strategy:** Temporal Frame Interpolation.
* **Calculations:**
  $$	ext{Target Effective FPS} = rac{	ext{Requested Images}}{	ext{Duration (seconds)}} = rac{2500}{10} = 250	ext{ FPS}$$
* **Execution:**
  - FFmpeg filter: `minterpolate=fps=250:mi_mode=blend`
  - High-performance linear motion/blending filter generates intermediate in-between frames.
  - Python Linear Blend Fallback: If FFmpeg is unavailable, OpenCV weighted linear interpolation ($I = (1 - lpha) F_A + lpha F_B$) ensures seamless processing on any machine.
* **Result:** Exactly 2500 frames generated (300 original frames + 2200 synthetic interpolated frames).

---

## 5. System Architecture & Folder Structure

```
video-image-practise/
??? task1_video_to_images_app.py     # Main Web App (Routes & REST API)
??? task1_extract_frames.py          # Unified CLI extraction utility
??? task1_services/                  # Modular Enterprise Service Layer
?   ??? __init__.py
?   ??? config.py                    # Environment settings & dynamic paths
?   ??? validator.py                 # Deep OpenCV video & stream validation
?   ??? interpolator.py              # FFmpeg minterpolate & Python fallback
?   ??? extractor.py                 # Unified Case A / Case B orchestrator
?   ??? job_manager.py               # Thread-safe background job scheduler
?   ??? zip_service.py               # Disk-streamed ZIP archive generator
??? templates/
?   ??? task1_index.html             # Dark glassmorphism frontend application
??? storage/                         # Managed data directories
    ??? uploads/                     # Sanitized uploaded video streams
    ??? outputs/                     # Isolated per-job frame directories
    ??? zips/                        # Generated ZIP archives
```

---

## 6. Requirements & Prerequisites

- **Operating System:** Windows, macOS, or Linux.
- **Python:** 3.8+ (Tested on Python 3.10 - 3.14).
- **FFmpeg:** Recommended for accelerated interpolation (auto-detected on PATH or configured via `FFMPEG_PATH`). If absent, the built-in OpenCV fallback automatically activates.

---

## 7. Installation & Setup

1. **Clone or Navigate to the Repository:**
   ```bash
   cd C:\Users\Intern\Downloads\video-image-practise
   ```

2. **Install Python Dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Verify FFmpeg (Optional but Recommended):**
   ```bash
   ffmpeg -version
   ```
   *(If not on PATH, see Configuration below to set `FFMPEG_PATH`).*

---

## 8. Configuration & Environment Variables

All filesystem locations and thresholds are fully configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `UPLOAD_DIR` | `./storage/uploads` | Temporary upload staging directory |
| `OUTPUT_DIR` | `./storage/outputs` | Directory storing generated image sets |
| `ZIP_DIR` | `./storage/zips` | Destination for generated ZIP downloads |
| `TEMP_DIR` | `./storage/temp` | Temporary frame cache for interpolation |
| `MAX_FILE_SIZE_MB` | `500` | Maximum single video upload size in Megabytes |
| `MAX_IMAGE_COUNT` | `10000` | Safety ceiling for maximum requested images per video |
| `FFMPEG_PATH` | *(Auto-detected)* | Absolute path to the `ffmpeg` executable binary |

---

## 9. How to Run Locally

### Start the Web Application
```bash
python task1_video_to_images_app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 10. API Specification

| Endpoint | Method | Description |
|---|---|---|
| `POST /api/upload` | Multipart | Uploads one or more videos; returns metadata & validation results |
| `POST /api/calculate-strategy` | JSON | Calculates effective FPS, Case A/B determination, and frame breakdown |
| `POST /api/extract` | JSON | Initiates background batch extraction; returns `batch_id` |
| `GET /api/jobs/<job_id>` | GET | Polls background processing status, progress percentage, and stage info |
| `GET /api/jobs/<job_id>/images` | GET | Returns paginated list of generated image URLs (`?page=1&per_page=50`) |
| `GET /api/jobs/<job_id>/image/<path>`| GET | Serves specific extracted image file securely |
| `GET /api/jobs/<job_id>/download-zip`| GET | Downloads master ZIP archive containing all processed videos |
| `GET /api/jobs/<job_id>/download-zip/<video_id>` | GET | Downloads individual video ZIP archive |

---

## 11. CLI Usage

The command-line interface provides the exact same high-performance extraction and interpolation capabilities:

```bash
# Request 100 uniformly sampled frames (Case A)
python task1_extract_frames.py input.mp4 --count 100 --output out_frames

# Request 2500 images with automatic interpolation (Case B)
python task1_extract_frames.py input.mp4 --count 2500 --output out_frames --format jpg --quality 90

# Downscale to 1280x720 while preserving aspect ratio
python task1_extract_frames.py input.mp4 --count 500 --resize 1280x720 --output out_frames

# Batch extract all videos in a directory
python task1_extract_frames.py --video-dir ./raw_videos --output ./dataset_frames --count 200
```

---

## 12. Memory & Performance Considerations

- **No Full Video Loading into RAM:** Video streams are piped through OpenCV/FFmpeg frames without accumulating raw video bytes in memory.
- **Disk-Streamed ZIP Files:** ZIP archives are generated by streaming files from disk using `zipfile.ZIP_DEFLATED`, preventing high memory spikes.
- **Paginated Image Gallery:** Browsers rendering thousands of DOM `<img>` elements concurrently can crash. The UI utilizes lazy pagination (50 images per page) for instant interaction.
- **Capping & Trimming Guards:** Video decoders may drop or misalign 1 frame at boundary timestamps. Both native and interpolated pipelines enforce strict length validation so `generated == requested` is always true.

---

## 13. Security & Validation

- **Path Traversal Protection:** User-supplied filenames are sanitized with UUID-prefixed directory isolation.
- **Deep Stream Inspection:** Files renamed to `.mp4` that contain corrupt or non-media bytes are rejected immediately via stream decoding checks.
- **Masked Technical Errors:** Internal exceptions are logged to stdout/stderr while returning clean, actionable messages to the UI.

---

## 14. Known Limitations & Future Improvements

- **FFmpeg Frame Blending:** The `minterpolate` filter with `mi_mode=blend` generates smooth transitions between frames. On fast-moving objects, ghosting or transparency artifacts may occur.
- **Deep Learning Enhancements (Future Work):**
  - Optical-flow deep neural networks such as **RIFE** (Real-Time Intermediate Flow Estimation) or **FILM** (Frame Interpolation for Large Motion) can be incorporated as optional GPU-accelerated backend plugins.
  - Due to standard local PC hardware constraints and to ensure universal reliability without heavy PyTorch model downloads, the FFmpeg filter is prioritized.

---

## 15. Troubleshooting

- **"OpenCV could not open video"**: Ensure the video file is not corrupted and uses standard codecs (H.264, HEVC, VP9, AV1).
- **FFmpeg not detected**: Set the environment variable `FFMPEG_PATH="C:\path\to\ffmpeg.exe"` or install FFmpeg and add it to your system PATH.
- **Port 5000 in use**: Start the app on a custom port: `PORT=5050 python task1_video_to_images_app.py`.
