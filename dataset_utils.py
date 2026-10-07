"""Comprehensive Command-Line and Programmatic Suite for YOLO & Computer Vision Datasets.

This module provides enterprise-grade dataset management, validation, cleaning,
splitting, renaming, conversion, and audit reporting for YOLO workflows.
Supports both flat directories and standard hierarchical YOLO layouts (images/ & labels/).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Supported extensions
VIDEO_EXTENSIONS = {
    ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg",
    ".wmv", ".webm",
}
IMAGE_EXTENSIONS = {
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
}
CONVERTIBLE_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS - {".jpg"}


# --------------------------------------------------------------------------- #
# Helper Utilities
# --------------------------------------------------------------------------- #

def get_files(
    directory: Path,
    extensions: Set[str],
    recursive: bool = False,
) -> List[Path]:
    """Retrieve sorted list of files matching extensions case-insensitively."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    ext_lower = {ext.lower() for ext in extensions}
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = [
        path for path in iterator
        if path.is_file() and path.suffix.lower() in ext_lower
    ]
    return sorted(files, key=lambda p: (p.parent, p.name.casefold()))


def find_image_label_pairs(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
    recursive: bool = False,
) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Path]]:
    """Pair image files with corresponding YOLO .txt files.

    If labels_dir is None, looks for .txt files in the same directory as images.
    Returns:
        pairs: List of (image_path, label_path_or_None)
        orphaned_labels: List of label_path without matching images
    """
    images = get_files(images_dir, IMAGE_EXTENSIONS, recursive=recursive)

    if labels_dir is not None:
        if not labels_dir.is_dir():
            raise NotADirectoryError(f"Labels directory not found: {labels_dir}")
        labels = get_files(labels_dir, {".txt"}, recursive=recursive)
    else:
        labels = get_files(images_dir, {".txt"}, recursive=recursive)

    # Map stem -> list of label paths to handle potential case variations
    label_map: Dict[str, Path] = {}
    for lbl in labels:
        label_map.setdefault(lbl.stem.casefold(), lbl)

    matched_label_stems: Set[str] = set()
    pairs: List[Tuple[Path, Optional[Path]]] = []

    for img in images:
        stem_key = img.stem.casefold()
        lbl = label_map.get(stem_key)
        if lbl is not None:
            matched_label_stems.add(stem_key)
            pairs.append((img, lbl))
        else:
            pairs.append((img, None))

    orphaned_labels = [
        lbl for lbl in labels
        if lbl.stem.casefold() not in matched_label_stems
    ]
    return pairs, orphaned_labels


def _check_output_paths(destinations: List[Path]) -> None:
    """Ensure no duplicate destination paths or collisions with existing files."""
    resolved = [p.resolve() for p in destinations]
    if len(set(resolved)) != len(destinations):
        raise ValueError("Multiple source files would produce identical destination paths.")
    existing = next((p for p in destinations if p.exists()), None)
    if existing is not None:
        raise FileExistsError(f"Operation would overwrite existing file: {existing}")


def _rename_batch_atomic(rename_pairs: List[Tuple[Path, Path]]) -> None:
    """Safely rename a batch of files via UUID staging to guarantee zero data loss.

    Handles in-place swaps, cyclical renames, and rolls back fully if an error occurs.
    """
    if not rename_pairs:
        return

    sources = [src for src, _ in rename_pairs]
    destinations = [dst for _, dst in rename_pairs]

    source_resolved = {s.resolve() for s in sources}
    dest_resolved = {d.resolve() for d in destinations}

    if len(source_resolved) != len(sources):
        raise ValueError("Duplicate source files selected for renaming.")
    if len(dest_resolved) != len(destinations):
        raise ValueError("Requested destination names collide with each other.")

    # Check destinations that are not part of the rename sources
    for _, dst in rename_pairs:
        if dst.exists() and dst.resolve() not in source_resolved:
            raise FileExistsError(f"Destination already exists and is not a source: {dst}")

    staged: List[Tuple[Path, Path, Path]] = []
    completed: List[Tuple[Path, Path, Path]] = []

    try:
        # Phase 1: Rename all sources to unique temporary files
        for src, dst in rename_pairs:
            temp_name = src.parent / f".tmp_{uuid.uuid4().hex}_{src.name}"
            src.rename(temp_name)
            staged.append((src, temp_name, dst))

        # Phase 2: Rename all temporary files to final destinations
        for src, temp_name, dst in staged:
            dst.parent.mkdir(parents=True, exist_ok=True)
            temp_name.rename(dst)
            completed.append((src, temp_name, dst))

    except Exception as exc:
        # Rollback completed
        for _, temp_name, dst in reversed(completed):
            if dst.exists():
                try:
                    dst.rename(temp_name)
                except Exception:
                    pass
        # Rollback staged
        for src, temp_name, _ in reversed(staged):
            if temp_name.exists():
                try:
                    temp_name.rename(src)
                except Exception:
                    pass
        raise OSError(f"Batch rename failed and was rolled back: {exc}") from exc


# --------------------------------------------------------------------------- #
# Module 1: Renaming Operations
# --------------------------------------------------------------------------- #

def rename_videos(
    directory: Path,
    prefix: str = "video_",
    start_index: int = 1,
    padding: int = 3,
    dry_run: bool = False,
) -> int:
    """Sequentially rename video files with configurable prefix and padding."""
    videos = get_files(directory, VIDEO_EXTENSIONS)
    if not videos:
        return 0

    rename_pairs: List[Tuple[Path, Path]] = []
    for idx, vid in enumerate(videos, start=start_index):
        new_name = f"{prefix}{idx:0{padding}d}{vid.suffix.lower()}"
        rename_pairs.append((vid, vid.with_name(new_name)))

    if dry_run:
        print(f"[DRY-RUN] Would rename {len(rename_pairs)} video(s):")
        for src, dst in rename_pairs[:10]:
            print(f"  {src.name} -> {dst.name}")
        if len(rename_pairs) > 10:
            print(f"  ... and {len(rename_pairs) - 10} more.")
        return len(rename_pairs)

    _rename_batch_atomic(rename_pairs)
    return len(rename_pairs)


def rename_images_and_annotations(
    directory: Path,
    labels_dir: Optional[Path] = None,
    prefix: str = "image_",
    start_index: int = 1,
    padding: int = 4,
    dry_run: bool = False,
    manifest_path: Optional[Path] = None,
) -> int:
    """Rename images and their matching YOLO annotations simultaneously.

    Supports flat directories (both in same folder) and separate labels_dir.
    Safe two-phase atomic staging prevents file loss or collisions.
    """
    pairs, orphans = find_image_label_pairs(directory, labels_dir)
    if not pairs:
        return 0

    # Check for duplicate stems among images (e.g. img1.jpg and img1.png)
    image_stems = [img.stem.casefold() for img, _ in pairs]
    if len(image_stems) != len(set(image_stems)):
        raise ValueError(
            "Images with identical stems but different extensions detected. "
            "Please remove or resolve ambiguous duplicate filenames before renaming."
        )

    rename_batch: List[Tuple[Path, Path]] = []
    manifest_rows: List[Dict[str, str]] = []

    for idx, (img, lbl) in enumerate(pairs, start=start_index):
        new_stem = f"{prefix}{idx:0{padding}d}"
        new_img = img.with_name(f"{new_stem}{img.suffix.lower()}")
        rename_batch.append((img, new_img))

        new_lbl_str = ""
        if lbl is not None:
            new_lbl = lbl.with_name(f"{new_stem}.txt")
            rename_batch.append((lbl, new_lbl))
            new_lbl_str = str(new_lbl.name)

        manifest_rows.append({
            "original_image": img.name,
            "new_image": new_img.name,
            "original_label": lbl.name if lbl else "",
            "new_label": new_lbl_str,
            "has_label": "true" if lbl else "false",
        })

    if dry_run:
        print(f"[DRY-RUN] Would rename {len(pairs)} image-label pair(s):")
        for row in manifest_rows[:10]:
            print(f"  {row['original_image']} -> {row['new_image']} (Label: {row['original_label']} -> {row['new_label']})")
        if len(manifest_rows) > 10:
            print(f"  ... and {len(manifest_rows) - 10} more.")
        return len(pairs)

    _rename_batch_atomic(rename_batch)

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["original_image", "new_image", "original_label", "new_label", "has_label"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    return len(pairs)


# --------------------------------------------------------------------------- #
# Module 2: Dataset Distribution
# --------------------------------------------------------------------------- #

def distribute_dataset(
    source_images: Path,
    destination_root: Path,
    source_labels: Optional[Path] = None,
    num_folders: Optional[int] = None,
    batch_size: Optional[int] = None,
    mode: str = "copy",
    manifest_path: Optional[Path] = None,
) -> int:
    """Distribute image-annotation pairs across N folders or batches of size K.

    Preserves exact matching pairs between images and annotations.
    """
    pairs, orphans = find_image_label_pairs(source_images, source_labels)
    if not pairs:
        return 0

    total_pairs = len(pairs)
    if num_folders is None and batch_size is None:
        raise ValueError("Must specify either num_folders (--num-folders) or batch_size (--batch-size).")

    if num_folders is not None:
        if num_folders <= 0:
            raise ValueError("num_folders must be greater than zero.")
        num_folders = min(num_folders, total_pairs)
        base = total_pairs // num_folders
        extra = total_pairs % num_folders
        folder_counts = [base + (1 if i < extra else 0) for i in range(num_folders)]
    else:
        assert batch_size is not None
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        num_folders = math.ceil(total_pairs / batch_size)
        folder_counts = []
        remaining = total_pairs
        for _ in range(num_folders):
            count = min(batch_size, remaining)
            folder_counts.append(count)
            remaining -= count

    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, str]] = []
    pair_index = 0

    for folder_idx, count in enumerate(folder_counts, start=1):
        folder_name = f"folder_{folder_idx:02d}"
        dest_folder = destination_root / folder_name

        # Determine subfolder structure: if source had separate labels, mirror it; otherwise flat
        if source_labels is not None:
            dest_img_dir = dest_folder / "images"
            dest_lbl_dir = dest_folder / "labels"
        else:
            dest_img_dir = dest_folder
            dest_lbl_dir = dest_folder

        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(count):
            img, lbl = pairs[pair_index]
            pair_index += 1

            target_img = dest_img_dir / img.name
            if mode == "move":
                shutil.move(str(img), str(target_img))
            else:
                shutil.copy2(str(img), str(target_img))

            target_lbl_name = ""
            if lbl is not None:
                target_lbl = dest_lbl_dir / lbl.name
                if mode == "move":
                    shutil.move(str(lbl), str(target_lbl))
                else:
                    shutil.copy2(str(lbl), str(target_lbl))
                target_lbl_name = target_lbl.name

            manifest_rows.append({
                "assigned_folder": folder_name,
                "image": img.name,
                "label": target_lbl_name,
                "mode": mode,
            })

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["assigned_folder", "image", "label", "mode"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    return total_pairs


# --------------------------------------------------------------------------- #
# Module 3: Background & Orphan Separation
# --------------------------------------------------------------------------- #

def separate_background_and_orphans(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
    dest_bg: Optional[Path] = None,
    dest_orphan: Optional[Path] = None,
    mode: str = "move",
) -> Tuple[int, int]:
    """Safely separate unannotated/background images and orphaned labels into dedicated folders."""
    pairs, orphans = find_image_label_pairs(images_dir, labels_dir)

    unannotated_images = [img for img, lbl in pairs if lbl is None]

    if dest_bg is None:
        dest_bg = images_dir / "background_images"
    if dest_orphan is None:
        dest_orphan = (labels_dir or images_dir) / "orphaned_labels"

    dest_bg.mkdir(parents=True, exist_ok=True)
    dest_orphan.mkdir(parents=True, exist_ok=True)

    action: Callable[[str, str], Any] = shutil.move if mode == "move" else shutil.copy2

    for img in unannotated_images:
        action(str(img), str(dest_bg / img.name))

    for lbl in orphans:
        action(str(lbl), str(dest_orphan / lbl.name))

    return len(unannotated_images), len(orphans)


# --------------------------------------------------------------------------- #
# Module 4: Label Cleaning & Verification
# --------------------------------------------------------------------------- #

def remove_empty_annotations(
    directory: Path,
    treat_whitespace_as_empty: bool = True,
    quarantine_dir: Optional[Path] = None,
) -> List[Path]:
    """Find and remove or quarantine empty or whitespace-only YOLO annotations."""
    removed: List[Path] = []
    if quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    for annotation in get_files(directory, {".txt"}):
        is_empty = False
        if annotation.stat().st_size == 0:
            is_empty = True
        elif treat_whitespace_as_empty:
            content = annotation.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                is_empty = True

        if is_empty:
            if quarantine_dir is not None:
                shutil.move(str(annotation), str(quarantine_dir / annotation.name))
            else:
                annotation.unlink()
            removed.append(annotation)
    return removed


def remove_unannotated_images(
    directory: Path,
    labels_dir: Optional[Path] = None,
) -> List[Path]:
    """Permanently delete images that lack a matching annotation file."""
    pairs, _ = find_image_label_pairs(directory, labels_dir)
    removed: List[Path] = []
    for img, lbl in pairs:
        if lbl is None:
            img.unlink()
            removed.append(img)
    return removed


def find_mismatches(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
) -> Tuple[List[Path], List[Path]]:
    """Identify images missing annotations and annotations missing images."""
    pairs, orphans = find_image_label_pairs(images_dir, labels_dir)
    missing_annotations = [img for img, lbl in pairs if lbl is None]
    return missing_annotations, orphans


def write_mismatch_report(
    images_dir: Path,
    report_path: Path,
    labels_dir: Optional[Path] = None,
) -> Tuple[int, int]:
    """Generate detailed audit report of dataset mismatches in TXT, CSV, or JSON format."""
    missing_annotations, missing_images = find_mismatches(images_dir, labels_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.suffix.lower() == ".json":
        data = {
            "summary": {
                "images_without_annotations": len(missing_annotations),
                "annotations_without_images": len(missing_images),
            },
            "images_without_annotations": [str(p) for p in missing_annotations],
            "annotations_without_images": [str(p) for p in missing_images],
        }
        report_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif report_path.suffix.lower() == ".csv":
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "FilePath"])
            for p in missing_annotations:
                writer.writerow(["ImageMissingAnnotation", str(p)])
            for p in missing_images:
                writer.writerow(["AnnotationMissingImage", str(p)])
    else:
        lines = [
            "# Dataset Mismatch Audit Report",
            f"Images Directory: {images_dir}",
            f"Labels Directory: {labels_dir or images_dir}",
            "",
            f"Images without annotations ({len(missing_annotations)}):",
            *(f"  {p.name}" for p in missing_annotations),
            "",
            f"Annotations without images ({len(missing_images)}):",
            *(f"  {p.name}" for p in missing_images),
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")

    return len(missing_annotations), len(missing_images)


def create_missing_annotations(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
) -> List[Path]:
    """Create empty (0-byte) YOLO .txt annotations for unannotated images.

    Essential for negative / background images in YOLO training pipelines.
    """
    pairs, _ = find_image_label_pairs(images_dir, labels_dir)
    target_dir = labels_dir if labels_dir is not None else images_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    for img, lbl in pairs:
        if lbl is None:
            new_lbl = target_dir / f"{img.stem}.txt"
            if not new_lbl.exists():
                with new_lbl.open("w", encoding="utf-8"):
                    pass
                created.append(new_lbl)
    return created


# --------------------------------------------------------------------------- #
# Module 5: Deep YOLO Annotation Validator & Auto-Repair
# --------------------------------------------------------------------------- #

@dataclass
class BBoxIssue:
    file_path: str
    line_number: int
    issue_type: str
    details: str
    raw_line: str


def validate_yolo_annotations(
    directory: Path,
    classes_file: Optional[Path] = None,
    num_classes: Optional[int] = None,
    fix: bool = False,
    backup: bool = True,
) -> Tuple[List[BBoxIssue], int]:
    """Perform rigorous validation on YOLO format .txt files.

    Checks:
      - Line structure: Exactly 5 values (class_id x_center y_center width height)
      - Integer class ID >= 0 and within known classes range (if provided)
      - Normalized bbox bounds: 0.0 <= coord <= 1.0
      - Non-zero width and height
      - Coordinates not NaN or Inf
      - Duplicate bounding boxes (IoU > 0.95)
    If fix=True, clamps coordinates, removes duplicates, and rewrites clean files.
    """
    known_classes: Optional[List[str]] = None
    if classes_file is not None and classes_file.is_file():
        known_classes = [
            line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        num_classes = len(known_classes)

    txt_files = get_files(directory, {".txt"})
    issues: List[BBoxIssue] = []
    files_fixed = 0

    def compute_iou(box1: List[float], box2: List[float]) -> float:
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    for txt in txt_files:
        try:
            content = txt.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            issues.append(BBoxIssue(str(txt), 0, "UNREADABLE_FILE", str(e), ""))
            continue

        lines = content.splitlines()
        valid_boxes: List[Tuple[int, List[float]]] = []
        file_has_error = False

        for line_no, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 5:
                issues.append(BBoxIssue(
                    str(txt), line_no, "INVALID_COLUMN_COUNT",
                    f"Expected 5 values, found {len(parts)}", raw
                ))
                file_has_error = True
                continue

            # Check class ID
            try:
                cls_id = int(parts[0])
                if cls_id < 0:
                    raise ValueError("Class ID cannot be negative.")
                if num_classes is not None and cls_id >= num_classes:
                    issues.append(BBoxIssue(
                        str(txt), line_no, "OUT_OF_RANGE_CLASS_ID",
                        f"Class ID {cls_id} >= num_classes ({num_classes})", raw
                    ))
                    file_has_error = True
            except ValueError:
                issues.append(BBoxIssue(
                    str(txt), line_no, "INVALID_CLASS_ID",
                    f"Class ID '{parts[0]}' is not a valid integer", raw
                ))
                file_has_error = True
                continue

            # Check coordinates
            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                issues.append(BBoxIssue(
                    str(txt), line_no, "NON_NUMERIC_COORDINATE",
                    "Non-numeric values in coordinates", raw
                ))
                file_has_error = True
                continue

            xc, yc, w, h = coords
            for val, name in zip(coords, ["x_center", "y_center", "width", "height"]):
                if math.isnan(val) or math.isinf(val):
                    issues.append(BBoxIssue(
                        str(txt), line_no, "NAN_OR_INF_VALUE",
                        f"{name} is NaN or Inf", raw
                    ))
                    file_has_error = True

            if w <= 0 or h <= 0:
                issues.append(BBoxIssue(
                    str(txt), line_no, "ZERO_OR_NEGATIVE_AREA",
                    f"Invalid dimensions: width={w}, height={h}", raw
                ))
                file_has_error = True

            out_of_bounds = False
            for val, name in zip(coords, ["x_center", "y_center", "width", "height"]):
                if val < 0.0 or val > 1.0:
                    out_of_bounds = True
            if out_of_bounds:
                issues.append(BBoxIssue(
                    str(txt), line_no, "COORDINATE_OUT_OF_BOUNDS",
                    f"Coordinates outside [0, 1]: {coords}", raw
                ))
                file_has_error = True

            # Clamped version for fix
            c_xc = max(0.0, min(1.0, xc))
            c_yc = max(0.0, min(1.0, yc))
            c_w = max(0.0001, min(1.0, w))
            c_h = max(0.0001, min(1.0, h))

            # Duplicate check
            is_dup = False
            for prev_cls, prev_box in valid_boxes:
                if prev_cls == cls_id and compute_iou([c_xc, c_yc, c_w, c_h], prev_box) > 0.95:
                    issues.append(BBoxIssue(
                        str(txt), line_no, "DUPLICATE_BOX",
                        f"IoU > 0.95 with prior box in same file", raw
                    ))
                    file_has_error = True
                    is_dup = True
                    break

            if not is_dup:
                valid_boxes.append((cls_id, [c_xc, c_yc, c_w, c_h]))

        if fix and file_has_error:
            if backup:
                backup_path = txt.with_suffix(".txt.bak")
                shutil.copy2(str(txt), str(backup_path))
            # Write corrected lines
            out_lines = [
                f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                for cls_id, box in valid_boxes
            ]
            txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            files_fixed += 1

    return issues, files_fixed


# --------------------------------------------------------------------------- #
# Module 6: Dataset Statistics
# --------------------------------------------------------------------------- #

def calculate_dataset_stats(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
    classes_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics, class balance, and bbox metrics."""
    class_names: Dict[int, str] = {}
    if classes_file is not None and classes_file.is_file():
        lines = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for idx, name in enumerate(lines):
            class_names[idx] = name

    pairs, orphans = find_image_label_pairs(images_dir, labels_dir)

    total_images = len(pairs)
    annotated_images = 0
    background_images = 0
    total_boxes = 0
    class_counts: Dict[int, int] = {}
    class_image_counts: Dict[int, int] = {}

    small_boxes = 0    # relative area < 0.001 (~32x32 in 1000x1000)
    medium_boxes = 0   # 0.001 <= area < 0.01
    large_boxes = 0    # area >= 0.01

    for img, lbl in pairs:
        if lbl is None or lbl.stat().st_size == 0:
            background_images += 1
            continue

        content = lbl.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            background_images += 1
            continue

        file_boxes = 0
        seen_classes_in_image: Set[int] = set()

        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    area = w * h

                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    seen_classes_in_image.add(cls_id)
                    file_boxes += 1
                    total_boxes += 1

                    if area < 0.001:
                        small_boxes += 1
                    elif area < 0.01:
                        medium_boxes += 1
                    else:
                        large_boxes += 1
                except ValueError:
                    pass

        if file_boxes > 0:
            annotated_images += 1
            for cls_id in seen_classes_in_image:
                class_image_counts[cls_id] = class_image_counts.get(cls_id, 0) + 1
        else:
            background_images += 1

    per_class_stats = []
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0.0
        name = class_names.get(cls_id, f"class_{cls_id}")
        per_class_stats.append({
            "class_id": cls_id,
            "class_name": name,
            "instances": count,
            "percentage": round(pct, 2),
            "images_count": class_image_counts.get(cls_id, 0),
        })

    return {
        "summary": {
            "total_images": total_images,
            "annotated_images": annotated_images,
            "background_images": background_images,
            "total_bounding_boxes": total_boxes,
            "avg_boxes_per_image": round(total_boxes / total_images, 2) if total_images else 0.0,
            "orphaned_labels": len(orphans),
        },
        "bbox_size_distribution": {
            "small_boxes": small_boxes,
            "medium_boxes": medium_boxes,
            "large_boxes": large_boxes,
        },
        "classes": per_class_stats,
    }


# --------------------------------------------------------------------------- #
# Module 7: Dataset Splitter (Train / Val / Test)
# --------------------------------------------------------------------------- #

def split_dataset(
    images_dir: Path,
    output_dir: Path,
    labels_dir: Optional[Path] = None,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    classes_file: Optional[Path] = None,
    seed: int = 42,
    mode: str = "copy",
) -> Dict[str, int]:
    """Split dataset into Train, Val, and Test subsets with standard YOLO directory structure.

    Automatically synchronizes labels and generates data.yaml.
    """
    pairs, _ = find_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        return {}

    train_ratio, val_ratio, test_ratio = ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-3):
        raise ValueError(f"Ratios must sum to 1.0. Given: {ratios}")

    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    action = shutil.move if mode == "move" else shutil.copy2
    results: Dict[str, int] = {}

    for split_name, split_pairs in splits.items():
        if not split_pairs:
            continue
        dest_img_dir = output_dir / "images" / split_name
        dest_lbl_dir = output_dir / "labels" / split_name
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, lbl in split_pairs:
            action(str(img), str(dest_img_dir / img.name))
            if lbl is not None and lbl.exists():
                action(str(lbl), str(dest_lbl_dir / lbl.name))
            else:
                # Create empty file so negative samples are preserved
                (dest_lbl_dir / f"{img.stem}.txt").touch()

        results[split_name] = len(split_pairs)

    # Generate data.yaml
    classes_list = []
    if classes_file is not None and classes_file.is_file():
        classes_list = [
            line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    yaml_lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if splits.get("test"):
        yaml_lines.append("test: images/test")
    yaml_lines.append("")
    yaml_lines.append(f"nc: {len(classes_list)}")
    if classes_list:
        yaml_lines.append("names:")
        for idx, name in enumerate(classes_list):
            yaml_lines.append(f"  {idx}: {name}")

    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return results


# --------------------------------------------------------------------------- #
# Module 8: Class Remapping & Filtering
# --------------------------------------------------------------------------- #

def remap_classes(
    directory: Path,
    mapping: Dict[int, Optional[int]],
    drop_unmapped: bool = True,
    backup: bool = True,
) -> int:
    """Remap class IDs or filter classes across all YOLO .txt files.

    If mapping[cls_id] is None, the box is removed.
    """
    txt_files = get_files(directory, {".txt"})
    modified_count = 0

    for txt in txt_files:
        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        new_lines: List[str] = []
        file_changed = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    if cls_id in mapping:
                        target = mapping[cls_id]
                        if target is not None:
                            new_lines.append(f"{target} {' '.join(parts[1:])}")
                        file_changed = True
                    elif not drop_unmapped:
                        new_lines.append(line)
                    else:
                        file_changed = True
                except ValueError:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if file_changed:
            if backup:
                shutil.copy2(str(txt), str(txt.with_suffix(".txt.bak")))
            txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
            modified_count += 1

    return modified_count


# --------------------------------------------------------------------------- #
# Module 9: Visualization
# --------------------------------------------------------------------------- #

def visualize_annotations(
    images_dir: Path,
    output_dir: Path,
    labels_dir: Optional[Path] = None,
    classes_file: Optional[Path] = None,
    num_samples: Optional[int] = 20,
) -> int:
    """Render bounding boxes and class labels onto images for visual quality inspection."""
    class_names: Dict[int, str] = {}
    if classes_file is not None and classes_file.is_file():
        lines = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for idx, name in enumerate(lines):
            class_names[idx] = name

    pairs, _ = find_image_label_pairs(images_dir, labels_dir)
    annotated_pairs = [(img, lbl) for img, lbl in pairs if lbl is not None and lbl.stat().st_size > 0]

    if not annotated_pairs:
        return 0

    if num_samples is not None and num_samples < len(annotated_pairs):
        sample_pairs = random.sample(annotated_pairs, num_samples)
    else:
        sample_pairs = annotated_pairs

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0

    # Palette of colors (BGR)
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    ]

    for img_path, lbl_path in sample_pairs:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h_img, w_img = image.shape[:2]

        lines = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    x1 = int((xc - w / 2) * w_img)
                    y1 = int((yc - h / 2) * h_img)
                    x2 = int((xc + w / 2) * w_img)
                    y2 = int((yc + h / 2) * h_img)

                    color = colors[cls_id % len(colors)]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    label_text = class_names.get(cls_id, f"ID:{cls_id}")
                    cv2.putText(
                        image, label_text, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )
                except ValueError:
                    pass

        out_path = output_dir / f"viz_{img_path.name}"
        cv2.imwrite(str(out_path), image)
        rendered += 1

    return rendered


# --------------------------------------------------------------------------- #
# Module 10: Format Conversions
# --------------------------------------------------------------------------- #

def convert_videos_to_mp4(directory: Path, output_dir: Path) -> int:
    """Convert non-MP4 videos to MP4 using ffmpeg with fallback to OpenCV."""
    videos = [
        path for path in get_files(directory, VIDEO_EXTENSIONS)
        if path.suffix.lower() != ".mp4"
    ]
    destinations = [output_dir / f"{source.stem}.mp4" for source in videos]
    _check_output_paths(destinations)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    converted = 0

    for source, destination in zip(videos, destinations):
        if ffmpeg is not None:
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-n",
                "-i", str(source), "-c:v", "libx264", "-c:a", "aac",
                "-movflags", "+faststart", str(destination),
            ]
            subprocess.run(command, check=True)
        else:
            # OpenCV Fallback
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            cap.release()
            writer.release()
        converted += 1

    return converted


def convert_images_to_jpg(directory: Path, output_dir: Path) -> int:
    """Convert non-JPG images to JPG with alpha-blending and depth normalization."""
    images = get_files(directory, CONVERTIBLE_IMAGE_EXTENSIONS)
    destinations = [output_dir / f"{source.stem}.jpg" for source in images]
    _check_output_paths(destinations)
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for source, destination in zip(images, destinations):
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {source}")
        if image.dtype == np.uint16:
            image = (image / 257).astype("uint8")
        elif image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            color = image[:, :, :3].astype("float32")
            alpha = image[:, :, 3:4].astype("float32") / 255.0
            image = (color * alpha + 255 * (1 - alpha)).astype("uint8")
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write JPG image: {destination}")
        converted += 1

    return converted


# --------------------------------------------------------------------------- #
# CLI Parser & Main Entrypoint
# --------------------------------------------------------------------------- #

def _confirm_or_exit(message: str, assume_yes: bool = False) -> None:
    if assume_yes:
        return
    answer = input(f"{message} Continue? [y/N] ")
    if answer.strip().casefold() not in {"y", "yes"}:
        raise SystemExit("Cancelled; no files were modified or deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive Computer Vision Dataset Engineering & YOLO Utility Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. rename-videos
    p_ren_vid = subparsers.add_parser("rename-videos", help="Sequentially rename video files")
    p_ren_vid.add_argument("directory", type=Path, help="Directory containing videos")
    p_ren_vid.add_argument("--prefix", type=str, default="video_", help="Naming prefix")
    p_ren_vid.add_argument("--start-idx", type=int, default=1, help="Starting counter index")
    p_ren_vid.add_argument("--padding", type=int, default=3, help="Digit padding (e.g. 3 -> 001)")
    p_ren_vid.add_argument("--dry-run", action="store_true", help="Preview renaming without changing files")

    # 2. rename-pairs
    p_ren_pair = subparsers.add_parser("rename-pairs", help="Rename images and matching YOLO annotations")
    p_ren_pair.add_argument("directory", type=Path, help="Images directory")
    p_ren_pair.add_argument("--labels-dir", type=Path, default=None, help="Separate labels directory (if any)")
    p_ren_pair.add_argument("--prefix", type=str, default="image_", help="Naming prefix")
    p_ren_pair.add_argument("--start-idx", type=int, default=1, help="Starting counter index")
    p_ren_pair.add_argument("--padding", type=int, default=4, help="Digit padding (e.g. 4 -> 0001)")
    p_ren_pair.add_argument("--manifest", type=Path, default=None, help="Path to save mapping CSV log")
    p_ren_pair.add_argument("--dry-run", action="store_true", help="Preview renaming without changing files")

    # 3. distribute
    p_dist = subparsers.add_parser("distribute", help="Distribute dataset pairs across N folders or batches")
    p_dist.add_argument("directory", type=Path, help="Source images directory")
    p_dist.add_argument("--dest", type=Path, required=True, help="Destination root directory")
    p_dist.add_argument("--labels-dir", type=Path, default=None, help="Source labels directory (if separate)")
    p_dist.add_argument("--num-folders", type=int, default=None, help="Number of destination folders")
    p_dist.add_argument("--batch-size", type=int, default=None, help="Number of pairs per destination folder")
    p_dist.add_argument("--mode", choices=["copy", "move"], default="copy", help="File action")
    p_dist.add_argument("--manifest", type=Path, default=None, help="Path to save distribution CSV manifest")

    # 4. separate-bg
    p_sep = subparsers.add_parser("separate-bg", help="Separate background images and orphaned annotations")
    p_sep.add_argument("directory", type=Path, help="Images directory")
    p_sep.add_argument("--labels-dir", type=Path, default=None, help="Labels directory")
    p_sep.add_argument("--bg-dir", type=Path, default=None, help="Destination directory for background images")
    p_sep.add_argument("--orphan-dir", type=Path, default=None, help="Destination directory for orphaned labels")
    p_sep.add_argument("--mode", choices=["copy", "move"], default="move", help="File action")

    # 5. clean-empty-labels
    p_clean_empty = subparsers.add_parser("remove-empty-labels", help="Clean or quarantine empty/whitespace annotations")
    p_clean_empty.add_argument("directory", type=Path, help="Labels directory")
    p_clean_empty.add_argument("--quarantine", type=Path, default=None, help="Move empty files to quarantine instead of deleting")
    p_clean_empty.add_argument("--yes", action="store_true", help="Confirm deletion non-interactively")

    # 6. remove-unannotated-images
    p_clean_img = subparsers.add_parser("remove-unannotated-images", help="Delete images missing annotations")
    p_clean_img.add_argument("directory", type=Path, help="Images directory")
    p_clean_img.add_argument("--labels-dir", type=Path, default=None, help="Labels directory")
    p_clean_img.add_argument("--yes", action="store_true", help="Confirm deletion non-interactively")

    # 7. mismatches
    p_mismatch = subparsers.add_parser("mismatches", help="Generate dataset mismatch report")
    p_mismatch.add_argument("directory", type=Path, help="Images directory")
    p_mismatch.add_argument("--labels-dir", type=Path, default=None, help="Labels directory")
    p_mismatch.add_argument("--report", type=Path, default=None, help="Report output path (.txt, .csv, or .json)")

    # 8. create-labels
    p_create_lbl = subparsers.add_parser("create-labels", help="Create empty YOLO labels for background images")
    p_create_lbl.add_argument("directory", type=Path, help="Images directory")
    p_create_lbl.add_argument("--labels-dir", type=Path, default=None, help="Labels directory")

    # 9. validate
    p_val = subparsers.add_parser("validate", help="Deep YOLO annotation syntax, coordinate, and duplicate validator")
    p_val.add_argument("directory", type=Path, help="Labels directory")
    p_val.add_argument("--classes", type=Path, default=None, help="Path to classes.txt for class ID checking")
    p_val.add_argument("--num-classes", type=int, default=None, help="Total number of valid classes")
    p_val.add_argument("--fix", action="store_true", help="Auto-repair out-of-bounds coordinates and duplicates")
    p_val.add_argument("--report", type=Path, default=None, help="Save validation report CSV")

    # 10. stats
    p_stats = subparsers.add_parser("stats", help="Calculate comprehensive dataset and class statistics")
    p_stats.add_argument("directory", type=Path, help="Images directory")
    p_stats.add_argument("--labels-dir", type=Path, default=None, help="Labels directory")
    p_stats.add_argument("--classes", type=Path, default=None, help="Path to classes.txt")
    p_stats.add_argument("--json", type=Path, default=None, help="Save stats as JSON")

    # 11. split
    p_split = subparsers.add_parser("split", help="Split dataset into Train/Val/Test with standard YOLO layout")
    p_split.add_argument("directory", type=Path, help="Source images directory")
    p_split.add_argument("--output", type=Path, required=True, help="Destination dataset directory")
    p_split.add_argument("--labels-dir", type=Path, default=None, help="Source labels directory")
    p_split.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], help="Train Val Test ratios")
    p_split.add_argument("--classes", type=Path, default=None, help="Path to classes.txt to write to data.yaml")
    p_split.add_argument("--mode", choices=["copy", "move"], default="copy", help="File action")

    # 12. remap-classes
    p_remap = subparsers.add_parser("remap-classes", help="Remap or filter class IDs in YOLO annotations")
    p_remap.add_argument("directory", type=Path, help="Labels directory")
    p_remap.add_argument("--mapping", type=str, required=True, help="Mapping spec, e.g. '0:0,1:1,12:2' or file path")
    p_remap.add_argument("--keep-unmapped", action="store_true", help="Keep unmapped classes instead of dropping")

    # 13. visualize
    p_viz = subparsers.add_parser("visualize", help="Render YOLO bounding boxes and labels onto images")
    p_viz.add_argument("directory", type=Path, help="Images directory")
    p_viz.add_argument("--output", type=Path, default=Path("visualized_samples"), help="Output directory")
    p_viz.add_argument("--labels-dir", type=Path, default=None, help="Labels directory")
    p_viz.add_argument("--classes", type=Path, default=None, help="Path to classes.txt")
    p_viz.add_argument("--samples", type=int, default=20, help="Number of sample images to render")

    # 14. videos-to-mp4
    p_v2mp4 = subparsers.add_parser("videos-to-mp4", help="Convert videos to MP4 format")
    p_v2mp4.add_argument("directory", type=Path, help="Videos directory")
    p_v2mp4.add_argument("--output", type=Path, default=Path("mp4"), help="Output directory")

    # 15. images-to-jpg
    p_i2jpg = subparsers.add_parser("images-to-jpg", help="Convert images to JPG format")
    p_i2jpg.add_argument("directory", type=Path, help="Images directory")
    p_i2jpg.add_argument("--output", type=Path, default=Path("jpg"), help="Output directory")

    args = parser.parse_args()

    try:
        if args.command == "rename-videos":
            count = rename_videos(args.directory, prefix=args.prefix, start_index=args.start_idx, padding=args.padding, dry_run=args.dry_run)
            print(f"Successfully processed {count} video(s).")

        elif args.command == "rename-pairs":
            count = rename_images_and_annotations(
                args.directory, labels_dir=args.labels_dir, prefix=args.prefix,
                start_index=args.start_idx, padding=args.padding, dry_run=args.dry_run,
                manifest_path=args.manifest,
            )
            print(f"Successfully processed {count} image-annotation pair(s).")

        elif args.command == "distribute":
            count = distribute_dataset(
                args.directory, args.dest, source_labels=args.labels_dir,
                num_folders=args.num_folders, batch_size=args.batch_size,
                mode=args.mode, manifest_path=args.manifest,
            )
            print(f"Successfully distributed {count} pair(s) into {args.dest}.")

        elif args.command == "separate-bg":
            n_bg, n_orph = separate_background_and_orphans(
                args.directory, labels_dir=args.labels_dir,
                dest_bg=args.bg_dir, dest_orphan=args.orphan_dir, mode=args.mode,
            )
            print(f"Separated {n_bg} background/unannotated image(s) and {n_orph} orphaned annotation(s).")

        elif args.command == "remove-empty-labels":
            if args.quarantine is None:
                _confirm_or_exit(f"This will permanently delete empty labels from {args.directory}.", args.yes)
            removed = remove_empty_annotations(args.directory, quarantine_dir=args.quarantine)
            action_desc = "Quarantined" if args.quarantine else "Removed"
            print(f"{action_desc} {len(removed)} empty/whitespace annotation file(s).")

        elif args.command == "remove-unannotated-images":
            _confirm_or_exit(f"This will permanently delete images from {args.directory}.", args.yes)
            removed = remove_unannotated_images(args.directory, labels_dir=args.labels_dir)
            print(f"Removed {len(removed)} image(s) without annotations.")

        elif args.command == "mismatches":
            report = args.report or (args.directory.parent / f"{args.directory.name}_mismatch_report.txt")
            n_img, n_lbl = write_mismatch_report(args.directory, report, labels_dir=args.labels_dir)
            print(f"Report saved to {report}: {n_img} image(s) missing annotations, {n_lbl} annotation(s) missing images.")

        elif args.command == "create-labels":
            created = create_missing_annotations(args.directory, labels_dir=args.labels_dir)
            print(f"Created {len(created)} missing annotation file(s).")

        elif args.command == "validate":
            issues, fixed = validate_yolo_annotations(
                args.directory, classes_file=args.classes, num_classes=args.num_classes, fix=args.fix
            )
            print(f"Validation completed: Found {len(issues)} issue(s) across annotations.")
            if args.fix:
                print(f"Auto-repair applied: {fixed} file(s) corrected.")
            if args.report and issues:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                with args.report.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["FilePath", "LineNumber", "IssueType", "Details", "RawLine"])
                    for issue in issues:
                        writer.writerow([issue.file_path, issue.line_number, issue.issue_type, issue.details, issue.raw_line])
                print(f"Detailed issues report saved to {args.report}")

        elif args.command == "stats":
            stats = calculate_dataset_stats(args.directory, labels_dir=args.labels_dir, classes_file=args.classes)
            summary = stats["summary"]
            print("=" * 60)
            print(f"Dataset Summary: {args.directory}")
            print(f"Total Images: {summary['total_images']} | Annotated: {summary['annotated_images']} | Background: {summary['background_images']}")
            print(f"Total Boxes: {summary['total_bounding_boxes']} | Avg/Image: {summary['avg_boxes_per_image']}")
            print("-" * 60)
            print(f"{'Class ID':<10}{'Class Name':<20}{'Count':<10}{'Percent':<10}{'Images':<10}")
            print("-" * 60)
            for cls_info in stats["classes"]:
                print(f"{cls_info['class_id']:<10}{cls_info['class_name']:<20}{cls_info['instances']:<10}{cls_info['percentage']:<10}%{cls_info['images_count']:<10}")
            print("=" * 60)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
                print(f"Full statistics exported to {args.json}")

        elif args.command == "split":
            res = split_dataset(
                args.directory, args.output, labels_dir=args.labels_dir,
                ratios=tuple(args.ratios), classes_file=args.classes, mode=args.mode,
            )
            print(f"Dataset successfully split into {args.output}: {res}")

        elif args.command == "remap-classes":
            # Parse mapping from JSON or comma pairs
            mapping: Dict[int, Optional[int]] = {}
            if Path(args.mapping).is_file():
                raw_map = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
                mapping = {int(k): (int(v) if v is not None else None) for k, v in raw_map.items()}
            else:
                for item in args.mapping.split(","):
                    k, v = item.strip().split(":")
                    mapping[int(k)] = int(v) if v.lower() != "none" else None
            modified = remap_classes(args.directory, mapping, drop_unmapped=not args.keep_unmapped)
            print(f"Remapped annotations across {modified} file(s).")

        elif args.command == "visualize":
            count = visualize_annotations(
                args.directory, args.output, labels_dir=args.labels_dir,
                classes_file=args.classes, num_samples=args.samples,
            )
            print(f"Rendered {count} visualized sample(s) to {args.output}")

        elif args.command == "videos-to-mp4":
            count = convert_videos_to_mp4(args.directory, args.output)
            print(f"Converted {count} video(s) to {args.output}.")

        elif args.command == "images-to-jpg":
            count = convert_images_to_jpg(args.directory, args.output)
            print(f"Converted {count} image(s) to {args.output}.")

    except Exception as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
