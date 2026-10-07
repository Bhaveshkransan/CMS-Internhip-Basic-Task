"""
Frame Interpolation Engine.
Uses FFmpeg minterpolate for high-performance intermediate frame synthesis,
with pure-Python blended fallback.
"""

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import cv2
import numpy as np

from .config import FFMPEG_BIN

def interpolate_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    target_count: int,
    effective_fps: float,
    img_fmt: str = "jpg",
    quality: int = 92,
    resize_dim: Optional[Tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Path]:
    """
    Interpolate and extract frames using FFmpeg minterpolate filter with exact frame count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / f"frame_%06d.{img_fmt}")

    # Build filter chain
    filters = []
    # minterpolate to generate smooth intermediate frames at target effective FPS
    filters.append(f"minterpolate=fps={effective_fps:.4f}:mi_mode=blend")

    if resize_dim and resize_dim[0] > 0 and resize_dim[1] > 0:
        filters.append(f"scale={resize_dim[0]}:{resize_dim[1]}:flags=lanczos")

    filter_str = ",".join(filters)

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", filter_str,
        "-vframes", str(target_count),
    ]

    if img_fmt in ["jpg", "jpeg"]:
        # FFmpeg q:v scale: 2 is high quality (approx 95%), 31 is lowest
        qscale = max(2, min(31, int(round((100 - quality) * 0.3 + 2))))
        cmd.extend(["-q:v", str(qscale)])
    elif img_fmt == "png":
        cmd.extend(["-compression_level", "3"])
    elif img_fmt == "webp":
        cmd.extend(["-quality", str(quality)])

    cmd.append(pattern)

    # Execute FFmpeg process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    _, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg interpolation failed: {stderr[-400:]}")

    generated_files = sorted(output_dir.glob(f"frame_*.{img_fmt}"))

    # Ensure EXACT target count
    if len(generated_files) > target_count:
        # Trim excess frames from the end
        for extra in generated_files[target_count:]:
            try:
                extra.unlink()
            except Exception:
                pass
        generated_files = generated_files[:target_count]
    elif len(generated_files) < target_count and len(generated_files) > 0:
        # If short by a couple of boundary frames, replicate the last frame
        last_file = generated_files[-1]
        needed = target_count - len(generated_files)
        last_data = last_file.read_bytes()
        start_idx = len(generated_files) + 1
        for i in range(needed):
            new_file = output_dir / f"frame_{start_idx + i:06d}.{img_fmt}"
            new_file.write_bytes(last_data)
            generated_files.append(new_file)

    if progress_callback:
        progress_callback(len(generated_files), target_count)

    return generated_files

def interpolate_frames_python_fallback(
    video_path: Path,
    output_dir: Path,
    target_count: int,
    total_native_frames: int,
    img_fmt: str = "jpg",
    quality: int = 92,
    resize_dim: Optional[Tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Path]:
    """
    Pure Python blended frame generator fallback if FFmpeg is unavailable.
    Blends adjacent native frames according to fractional position.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video for fallback interpolation.")

    native_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_dim and resize_dim[0] > 0 and resize_dim[1] > 0:
            frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
        native_frames.append(frame)
    cap.release()

    if not native_frames:
        raise RuntimeError("No frames could be read from video.")

    num_native = len(native_frames)
    generated_files = []
    
    # Calculate fractional indices across duration
    fractional_indices = np.linspace(0, num_native - 1, target_count)

    for i, fidx in enumerate(fractional_indices):
        idx_low = int(math.floor(fidx))
        idx_high = min(idx_low + 1, num_native - 1)
        alpha = fidx - idx_low

        if alpha < 1e-4 or idx_low == idx_high:
            out_frame = native_frames[idx_low]
        else:
            # Linear blend of adjacent frames
            out_frame = cv2.addWeighted(
                native_frames[idx_low], 1.0 - alpha,
                native_frames[idx_high], alpha,
                0
            )

        out_path = output_dir / f"frame_{i + 1:06d}.{img_fmt}"
        if img_fmt in ["jpg", "jpeg"]:
            cv2.imwrite(str(out_path), out_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif img_fmt == "png":
            cv2.imwrite(str(out_path), out_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        elif img_fmt == "webp":
            cv2.imwrite(str(out_path), out_frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
        else:
            cv2.imwrite(str(out_path), out_frame)

        generated_files.append(out_path)
        if progress_callback and (i % 10 == 0 or i == target_count - 1):
            progress_callback(i + 1, target_count)

    return generated_files
