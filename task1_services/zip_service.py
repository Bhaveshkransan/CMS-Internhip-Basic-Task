"""
Disk-Streaming ZIP Creation Service.
Streams images directly into compressed archives without memory buffering.
Supports single video and multi-video batches.
"""

import zipfile
from pathlib import Path
from typing import Dict, Any, List

def create_single_video_zip(image_dir: Path, zip_dest: Path, prefix: str = "frame") -> Path:
    """Create a zip for a single video folder."""
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img_path in sorted(image_dir.glob("*.*")):
            if img_path.is_file() and not img_path.name.endswith(".zip"):
                zf.write(img_path, arcname=img_path.name)
    return zip_dest

def create_multi_video_batch_zip(batch_output_dir: Path, zip_dest: Path, video_subdirs: List[Path]) -> Path:
    """
    Create a master zip containing separate subdirectories for each video in the batch.
    Example:
      master.zip/
        video1/frame_000001.jpg
        video2/frame_000001.jpg
    """
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for vdir in video_subdirs:
            folder_name = vdir.name
            for img_path in sorted(vdir.glob("*.*")):
                if img_path.is_file() and not img_path.name.endswith(".zip"):
                    zf.write(img_path, arcname=f"{folder_name}/{img_path.name}")
    return zip_dest
