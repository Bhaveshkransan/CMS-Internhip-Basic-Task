"""
Unified Frame Extraction Service.
Intelligently handles:
- Case A: Requested count <= available native frames (uniform timeline sampling)
- Case B: Requested count > available native frames (frame interpolation)
Guarantees EXACT output frame counts, aspect-ratio resizing, and live progress reporting.
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import cv2
import numpy as np

from .config import FFMPEG_BIN, DEFAULT_IMAGE_QUALITY
from .validator import inspect_video
from .interpolator import interpolate_frames_ffmpeg, interpolate_frames_python_fallback

def calculate_extraction_strategy(metadata: Dict[str, Any], requested_count: int) -> Dict[str, Any]:
    """
    Calculate required effective FPS, whether interpolation is needed,
    and frame distribution breakdown.
    """
    total_frames = max(1, metadata.get("total_frames", 1))
    duration = max(0.01, metadata.get("duration_seconds", 1.0))
    native_fps = metadata.get("fps", 30.0)

    effective_fps = requested_count / duration
    needs_interpolation = requested_count > total_frames

    if needs_interpolation:
        original_frames_used = total_frames
        interpolated_frames = requested_count - total_frames
        message = (
            f"Your video contains approximately {total_frames} original frames. "
            f"You requested {requested_count} images, so {interpolated_frames} additional "
            f"frames will be generated through frame interpolation."
        )
    else:
        original_frames_used = requested_count
        interpolated_frames = 0
        message = (
            f"Your video contains approximately {total_frames} original frames. "
            f"You requested {requested_count} images, which will be sampled directly and "
            f"evenly across the video timeline."
        )

    return {
        "requested_count": requested_count,
        "total_source_frames": total_frames,
        "duration_seconds": duration,
        "native_fps": native_fps,
        "effective_fps": round(effective_fps, 2),
        "needs_interpolation": needs_interpolation,
        "original_frames_used": original_frames_used,
        "interpolated_frames": interpolated_frames,
        "message": message,
        "disclaimer": "Interpolated frames are generated between original video frames and may contain visual artifacts." if needs_interpolation else None
    }

def extract_native_frames_exact(
    video_path: Path,
    output_dir: Path,
    target_count: int,
    total_frames: int,
    fps: float,
    img_fmt: str = "jpg",
    quality: int = 92,
    resize_dim: Optional[Tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    Extract exactly target_count frames from original video with uniform distribution.
    Optimized for streaming sequential reads without memory ballooning.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file for frame extraction.")

    # Select exact frame indices
    if target_count >= total_frames:
        target_indices = list(range(total_frames))
    else:
        target_indices = list(np.linspace(0, total_frames - 1, target_count, dtype=int))

    target_set = set(target_indices)
    extracted_items = []
    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_set:
            if resize_dim and resize_dim[0] > 0 and resize_dim[1] > 0:
                frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)

            extracted_count += 1
            file_name = f"frame_{extracted_count:06d}.{img_fmt}"
            out_path = output_dir / file_name

            if img_fmt in ["jpg", "jpeg"]:
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif img_fmt == "png":
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            elif img_fmt == "webp":
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
            else:
                cv2.imwrite(str(out_path), frame)

            timestamp = round(frame_idx / fps, 2) if fps > 0 else 0.0
            extracted_items.append({
                "filename": file_name,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "is_interpolated": False,
            })

            if progress_callback:
                progress_callback(extracted_count, target_count, file_name)

            if extracted_count >= target_count:
                break

        frame_idx += 1

    cap.release()

    # In rare cases where video ends early, ensure EXACT count
    if len(extracted_items) < target_count and len(extracted_items) > 0:
        needed = target_count - len(extracted_items)
        last_item = extracted_items[-1]
        last_path = output_dir / last_item["filename"]
        last_bytes = last_path.read_bytes()
        start_count = len(extracted_items) + 1
        for i in range(needed):
            fname = f"frame_{start_count + i:06d}.{img_fmt}"
            (output_dir / fname).write_bytes(last_bytes)
            extracted_items.append({
                "filename": fname,
                "frame_idx": last_item["frame_idx"],
                "timestamp": last_item["timestamp"],
                "is_interpolated": True,
            })
            if progress_callback:
                progress_callback(len(extracted_items), target_count, fname)

    return extracted_items

def extract_video_to_images(
    video_path: Path,
    output_dir: Path,
    requested_count: int,
    img_fmt: str = "jpg",
    quality: int = 92,
    resize_dim: Optional[Tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None
) -> Dict[str, Any]:
    """
    Main extraction pipeline:
    1. Inspects video
    2. Calculates strategy
    3. Runs Case A (sampled native frames) or Case B (interpolated frames)
    4. Guarantees exact count and returns comprehensive job summary.
    """
    meta = inspect_video(video_path)
    strategy = calculate_extraction_strategy(meta, requested_count)
    total_frames = meta["total_frames"]
    fps = meta["fps"]
    duration = meta["duration_seconds"]

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_items = []
    if strategy["needs_interpolation"]:
        # Case B: Interpolation
        def prog_cb(current, total):
            if progress_callback:
                progress_callback(current, total, f"frame_{current:06d}.{img_fmt}", "Interpolating frames")

        if FFMPEG_BIN:
            try:
                paths = interpolate_frames_ffmpeg(
                    video_path=video_path,
                    output_dir=output_dir,
                    target_count=requested_count,
                    effective_fps=strategy["effective_fps"],
                    img_fmt=img_fmt,
                    quality=quality,
                    resize_dim=resize_dim,
                    progress_callback=prog_cb
                )
            except Exception:
                paths = interpolate_frames_python_fallback(
                    video_path=video_path,
                    output_dir=output_dir,
                    target_count=requested_count,
                    total_native_frames=total_frames,
                    img_fmt=img_fmt,
                    quality=quality,
                    resize_dim=resize_dim,
                    progress_callback=prog_cb
                )
        else:
            paths = interpolate_frames_python_fallback(
                video_path=video_path,
                output_dir=output_dir,
                target_count=requested_count,
                total_native_frames=total_frames,
                img_fmt=img_fmt,
                quality=quality,
                resize_dim=resize_dim,
                progress_callback=prog_cb
            )

        for i, p in enumerate(paths):
            extracted_items.append({
                "filename": p.name,
                "frame_idx": i + 1,
                "timestamp": round((i / requested_count) * duration, 2) if duration > 0 else 0.0,
                "is_interpolated": i >= total_frames,
            })
    else:
        # Case A: Native Sampling
        def prog_cb(current, total, fname):
            if progress_callback:
                progress_callback(current, total, fname, "Extracting original frames")

        extracted_items = extract_native_frames_exact(
            video_path=video_path,
            output_dir=output_dir,
            target_count=requested_count,
            total_frames=total_frames,
            fps=fps,
            img_fmt=img_fmt,
            quality=quality,
            resize_dim=resize_dim,
            progress_callback=prog_cb
        )

    return {
        "video_info": meta,
        "strategy": strategy,
        "requested": requested_count,
        "generated": len(extracted_items),
        "original_frames": strategy["original_frames_used"],
        "interpolated_frames": strategy["interpolated_frames"],
        "items": extracted_items,
        "first_frame": extracted_items[0]["filename"] if extracted_items else None,
        "format": img_fmt,
        "quality": quality,
        "resolution": f"{resize_dim[0]}x{resize_dim[1]}" if resize_dim else f"{meta['width']}x{meta['height']}",
    }
