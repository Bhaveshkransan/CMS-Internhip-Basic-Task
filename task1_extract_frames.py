"""High-Performance Video Frame Extraction Utility for Computer Vision & Deep Learning.

Extract frames from single videos or entire directories with flexible sampling:
  - Time interval (e.g. every N seconds)
  - Target FPS (e.g. 1 frame/sec, 0.5 fps, 2 fps)
  - Uniform frame count (e.g. exactly 100 frames across duration)
  - Frame stride (e.g. every Nth frame)
  - Scene / motion change detection (skipping duplicate static frames)
  - Time window clipping (--start / --end timestamps)
  - Custom output naming patterns, resizing, and image formats.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXTENSIONS = {
    ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg",
    ".wmv", ".webm",
}


def parse_timestamp(value: Optional[str]) -> Optional[float]:
    """Parse timestamp string (e.g. '12.5', '01:30', '01:15:30') into seconds."""
    if value is None:
        return None
    val = value.strip()
    if not val:
        return None
    if ":" in val:
        parts = [float(p) for p in val.split(":")]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        else:
            raise ValueError(f"Invalid timestamp format: {val}")
    return float(val)


def open_video(video_path: Path) -> Tuple[cv2.VideoCapture, float, int, int, int]:
    """Open video file and retrieve metadata (cap, fps, total_frames, width, height)."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0  # sensible fallback for variable streams
    if total_frames <= 0:
        total_frames = 1

    return capture, fps, total_frames, width, height


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    interval_seconds: Optional[float] = None,
    fps_target: Optional[float] = None,
    target_count: Optional[int] = None,
    stride: Optional[int] = None,
    scene_threshold: Optional[float] = None,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    naming_pattern: str = "{video}_{sec:04d}.jpg",
    img_format: str = "jpg",
    quality: int = 95,
    resize: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
) -> int:
    """Extract frames from a single video using specified sampling criteria."""
    capture, fps, total_frames, orig_w, orig_h = open_video(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_sec = start_seconds if start_seconds is not None else 0.0
    end_sec = end_seconds if end_seconds is not None else (total_frames / fps)
    start_sec = max(0.0, start_sec)
    end_sec = max(start_sec, end_sec)

    start_frame = int(round(start_sec * fps))
    end_frame = min(total_frames, int(round(end_sec * fps)))
    total_window_frames = max(1, end_frame - start_frame)

    # Determine frame sampling strategy
    saved_count = 0
    prev_frame_gray: Optional[np.ndarray] = None

    encode_params: List[int] = []
    if img_format.lower() in {"jpg", "jpeg"}:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_format.lower() == "png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    try:
        # Strategy A: Target Count (Uniform distribution or Interpolation)
        if target_count is not None:
            if target_count <= 0:
                raise ValueError("target_count must be > 0.")

            # If user requests more frames than available in the video window, use frame interpolation
            if target_count > total_window_frames:
                capture.release()
                from task1_services.extractor import extract_video_to_images
                res = extract_video_to_images(
                    video_path=video_path,
                    output_dir=output_dir,
                    requested_count=target_count,
                    img_fmt=img_format,
                    quality=quality,
                    resize_dim=resize,
                )
                print(f"[INFO] Interpolation applied: {res['original_frames']} original frames, {res['interpolated_frames']} synthetic interpolated frames generated.")
                return res["generated"]

            if target_count == 1:
                indices = [start_frame + total_window_frames // 2]
            else:
                indices = [
                    start_frame + (i * (total_window_frames - 1)) // (target_count - 1)
                    for i in range(min(target_count, total_window_frames))
                ]

            for idx in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = capture.read()
                if not ret or frame is None:
                    continue

                if scale is not None and scale > 0:
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                elif resize is not None:
                    frame = cv2.resize(frame, resize)

                sec = int(idx / fps)
                filename = naming_pattern.format(
                    video=video_path.stem,
                    index=saved_count + 1,
                    sec=sec,
                    frame_idx=idx,
                )
                if not filename.lower().endswith(f".{img_format.lower()}"):
                    filename = f"{filename}.{img_format.lower()}"

                out_file = output_dir / filename
                cv2.imwrite(str(out_file), frame, encode_params)
                saved_count += 1
            return saved_count

        # Strategy B: Streaming sequential reading (Interval, FPS, Stride, Scene Diff)
        frame_step = 1
        if interval_seconds is not None and interval_seconds > 0:
            frame_step = max(1, int(round(interval_seconds * fps)))
        elif fps_target is not None and fps_target > 0:
            frame_step = max(1, int(round(fps / fps_target)))
        elif stride is not None and stride > 0:
            frame_step = stride

        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame_idx = start_frame
        while current_frame_idx < end_frame:
            ret, frame = capture.read()
            if not ret or frame is None:
                break

            should_save = False

            # Check if frame matches step cadence
            if (current_frame_idx - start_frame) % frame_step == 0:
                should_save = True

            # Check scene change filter if enabled
            if should_save and scene_threshold is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame_gray is not None:
                    # Calculate mean absolute difference
                    diff = cv2.absdiff(gray, prev_frame_gray)
                    mean_diff = float(np.mean(diff))
                    if mean_diff < scene_threshold:
                        should_save = False
                if should_save:
                    prev_frame_gray = gray

            if should_save:
                if scale is not None and scale > 0:
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                elif resize is not None:
                    frame = cv2.resize(frame, resize)

                sec = int(current_frame_idx / fps)
                filename = naming_pattern.format(
                    video=video_path.stem,
                    index=saved_count + 1,
                    sec=sec,
                    frame_idx=current_frame_idx,
                )
                if not filename.lower().endswith(f".{img_format.lower()}"):
                    filename = f"{filename}.{img_format.lower()}"

                out_file = output_dir / filename
                cv2.imwrite(str(out_file), frame, encode_params)
                saved_count += 1

            current_frame_idx += 1

    finally:
        capture.release()

    return saved_count


def extract_batch(
    video_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    **extract_kwargs,
) -> int:
    """Batch extract frames across all videos in a directory."""
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {video_dir}")

    iterator = video_dir.rglob("*") if recursive else video_dir.iterdir()
    video_files = sorted([
        p for p in iterator
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        print(f"No video files found in {video_dir}")
        return 0

    total_extracted = 0
    print(f"Found {len(video_files)} video(s) to process in {video_dir}...")

    for idx, vid_path in enumerate(video_files, start=1):
        try:
            count = extract_frames_from_video(
                vid_path,
                output_dir,
                **extract_kwargs,
            )
            print(f"[{idx}/{len(video_files)}] {vid_path.name}: Extracted {count} frame(s)")
            total_extracted += count
        except Exception as exc:
            print(f"[{idx}/{len(video_files)}] Error processing {vid_path.name}: {exc}")

    return total_extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract video frames at intervals, uniform count, stride, or motion thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("video", type=Path, nargs="?", help="Input video path")
    input_group.add_argument("--video-dir", type=Path, help="Directory containing multiple videos to process")

    parser.add_argument(
        "--output", type=Path, default=Path("output_images"),
        help="Directory where extracted frames are saved",
    )

    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--interval", type=float, metavar="SECONDS", help="Extract frame every N seconds")
    sampling_group.add_argument("--fps", type=float, metavar="FPS", help="Extract at target frames per second (e.g. 1.0)")
    sampling_group.add_argument("--count", type=int, metavar="N", help="Extract exactly N uniformly distributed frames")
    sampling_group.add_argument("--stride", type=int, metavar="N", help="Extract every N-th frame")

    parser.add_argument("--scene-threshold", type=float, default=None, help="Skip static frames below this pixel difference threshold (e.g. 15.0)")
    parser.add_argument("--start", type=str, default=None, help="Start time in seconds or HH:MM:SS")
    parser.add_argument("--end", type=str, default=None, help="End time in seconds or HH:MM:SS")

    parser.add_argument("--pattern", type=str, default="{video}_{sec:04d}.jpg", help="Naming template ({video}, {sec}, {index}, {frame_idx})")
    parser.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg", help="Image output format")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--resize", type=str, default=None, help="Resize WxH (e.g. 1280x720)")
    parser.add_argument("--scale", type=float, default=None, help="Scale factor (e.g. 0.5)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for videos when using --video-dir")

    args = parser.parse_args()

    # Default to 1 second interval if no sampling method is provided
    interval = args.interval
    if interval is None and args.fps is None and args.count is None and args.stride is None:
        interval = 1.0

    parsed_start = parse_timestamp(args.start)
    parsed_end = parse_timestamp(args.end)

    resize_tuple = None
    if args.resize:
        w_str, h_str = args.resize.lower().split("x")
        resize_tuple = (int(w_str), int(h_str))

    kwargs = {
        "interval_seconds": interval,
        "fps_target": args.fps,
        "target_count": args.count,
        "stride": args.stride,
        "scene_threshold": args.scene_threshold,
        "start_seconds": parsed_start,
        "end_seconds": parsed_end,
        "naming_pattern": args.pattern,
        "img_format": args.format,
        "quality": args.quality,
        "resize": resize_tuple,
        "scale": args.scale,
    }

    try:
        if args.video_dir is not None:
            total = extract_batch(args.video_dir, args.output, recursive=args.recursive, **kwargs)
            print(f"Total extracted: {total} frames saved to {args.output}")
        else:
            total = extract_frames_from_video(args.video, args.output, **kwargs)
            print(f"Saved {total} frame(s) to {args.output}")
    except Exception as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
