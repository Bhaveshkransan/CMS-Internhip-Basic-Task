"""
Video Validation & Metadata Inspection Module.
Ensures media readability, detects corruption, and extracts accurate stream parameters.
"""

import math
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2

from .config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES

class VideoValidationError(Exception):
    """User-friendly video validation error."""
    pass

def validate_video_file(file_path: Path) -> None:
    """Validate existence, extension, and file size limits."""
    if not file_path.exists():
        raise VideoValidationError("File does not exist on disk.")

    size = file_path.stat().st_size
    if size == 0:
        raise VideoValidationError("Uploaded file is empty (0 bytes).")

    if size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise VideoValidationError(f"File size exceeds the {max_mb} MB limit.")

    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise VideoValidationError(f"Unsupported format '{ext}'. Supported formats: {allowed_list}")

def inspect_video(file_path: Path) -> Dict[str, Any]:
    """
    Deep media inspection using OpenCV.
    Verifies actual container and stream readability, returns normalized metadata.
    """
    validate_video_file(file_path)

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise VideoValidationError("Unable to read this video. The file may be corrupted or use an unsupported codec.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    # Test reading the first frame to confirm stream integrity
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise VideoValidationError("Unable to decode video frames. The video stream appears to be corrupted.")

    if fps <= 0 or math.isnan(fps):
        fps = 30.0  # Fallback to standard framerate

    duration_seconds = total_frames / fps if (fps > 0 and total_frames > 0) else 0.0
    if duration_seconds <= 0:
        duration_seconds = 0.1

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    hours = minutes // 60
    minutes = minutes % 60
    duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return {
        "filename": file_path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration_seconds, 2),
        "duration_formatted": duration_formatted,
        "size_bytes": file_path.stat().st_size,
    }
