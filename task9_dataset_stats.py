"""Standalone Script: Compute YOLO Dataset Statistics & Class Distribution.

Calculates:
  - Total images, annotated images, background images
  - Total bounding boxes, average boxes per image
  - Class-wise instance counts and percentage breakdown
  - Bounding box scale breakdown (small, medium, large)
Outputs clean terminal tables and optional JSON export.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def compute_stats(images_dir: Path, labels_dir: Path | None = None, classes_file: Path | None = None, json_out: Path | None = None) -> None:
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    lbl_dir = labels_dir or images_dir
    labels = {p.stem.casefold(): p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

    class_names = {}
    if classes_file and classes_file.is_file():
        lines = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        class_names = {idx: name for idx, name in enumerate(lines)}

    total_images = len(images)
    annotated_images = 0
    background_images = 0
    total_boxes = 0
    class_counts: Dict[int, int] = {}
    small_boxes = 0
    medium_boxes = 0
    large_boxes = 0

    for img in images:
        lbl = labels.get(img.stem.casefold())
        if lbl is None or lbl.stat().st_size == 0:
            background_images += 1
            continue

        content = lbl.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            background_images += 1
            continue

        lines = content.splitlines()
        img_boxes = 0
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                try:
                    c = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    class_counts[c] = class_counts.get(c, 0) + 1
                    img_boxes += 1
                    total_boxes += 1
                    area = w * h
                    if area < 0.001:
                        small_boxes += 1
                    elif area < 0.01:
                        medium_boxes += 1
                    else:
                        large_boxes += 1
                except ValueError:
                    pass

        if img_boxes > 0:
            annotated_images += 1
        else:
            background_images += 1

    print("=" * 65)
    print(f"YOLO Dataset Statistics: {images_dir}")
    print(f"  Total Images: {total_images} | Annotated: {annotated_images} | Background: {background_images}")
    print(f"  Total Boxes:  {total_boxes} | Avg Boxes/Image: {round(total_boxes / total_images, 2) if total_images else 0}")
    print("-" * 65)
    print(f"  Box Scales: Small (<32²): {small_boxes} | Medium: {medium_boxes} | Large: {large_boxes}")
    print("-" * 65)
    print(f"{'Class ID':<10}{'Class Name':<25}{'Count':<15}{'Percentage':<15}")
    print("-" * 65)
    for c_id in sorted(class_counts.keys()):
        cnt = class_counts[c_id]
        pct = round(cnt / total_boxes * 100, 2) if total_boxes else 0
        c_name = class_names.get(c_id, f"class_{c_id}")
        print(f"{c_id:<10}{c_name:<25}{cnt:<15}{pct}%")
    print("=" * 65)

    if json_out:
        data = {
            "total_images": total_images,
            "annotated_images": annotated_images,
            "background_images": background_images,
            "total_boxes": total_boxes,
            "classes": [
                {"id": cid, "name": class_names.get(cid, f"class_{cid}"), "count": cnt}
                for cid, cnt in sorted(class_counts.items())
            ],
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nStats exported to: {json_out}")


def main():
    parser = argparse.ArgumentParser(description="Calculate dataset class statistics and bounding box distribution.")
    parser.add_argument("images_folder", type=Path, help="Folder containing images")
    parser.add_argument("--labels-folder", type=Path, default=None, help="Folder containing labels (if separate)")
    parser.add_argument("--classes-file", type=Path, default=None, help="Path to classes.txt")
    parser.add_argument("--json", type=Path, default=None, help="Save stats as JSON file")
    args = parser.parse_args()

    compute_stats(args.images_folder, args.labels_folder, args.classes_file, args.json)


if __name__ == "__main__":
    main()
