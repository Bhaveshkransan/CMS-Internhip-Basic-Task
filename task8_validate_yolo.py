"""Standalone Script: Validate & Auto-Repair YOLO Annotations.

Inspects all .txt files in a folder for YOLO formatting rules:
  1. Exactly 5 values per line (class_id x_center y_center width height)
  2. Class IDs must be non-negative integers
  3. Coordinates must be normalized between 0.0 and 1.0
  4. Width and height must be strictly positive
  5. Flags duplicate/overlapping bounding boxes (IoU > 0.95)
  6. Auto-repair flag (--fix) clamps coordinates and removes duplicate boxes safely.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import List, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def validate_folder(
    labels_dir: Path,
    classes_file: Path | None = None,
    fix: bool = False,
    report_path: Path | None = None,
) -> None:
    num_classes = None
    if classes_file and classes_file.is_file():
        num_classes = len([l for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()])

    txt_files = [
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt" and (classes_file is None or p.resolve() != classes_file.resolve())
    ]
    issues = []
    files_fixed = 0

    for txt in txt_files:
        lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        valid_boxes: List[Tuple[int, List[float]]] = []
        has_error = False

        for l_num, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 5:
                issues.append((txt.name, l_num, "COLUMNS_ERROR", f"Expected 5 values, got {len(parts)}"))
                has_error = True
                continue

            try:
                cls_id = int(parts[0])
                if cls_id < 0 or (num_classes and cls_id >= num_classes):
                    issues.append((txt.name, l_num, "INVALID_CLASS", f"Class ID {cls_id} invalid"))
                    has_error = True
            except ValueError:
                issues.append((txt.name, l_num, "NON_INT_CLASS", f"Class '{parts[0]}' not integer"))
                has_error = True
                continue

            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                issues.append((txt.name, l_num, "NON_NUMERIC_COORD", "Coordinates contain non-numbers"))
                has_error = True
                continue

            xc, yc, w, h = coords
            if w <= 0 or h <= 0:
                issues.append((txt.name, l_num, "ZERO_OR_NEGATIVE_AREA", f"w={w}, h={h}"))
                has_error = True

            if any(val < 0.0 or val > 1.0 for val in coords):
                issues.append((txt.name, l_num, "OUT_OF_BOUNDS", f"Coords not in [0, 1]: {coords}"))
                has_error = True

            # Clamped version for fix
            c_box = [max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)), max(0.0001, min(1.0, w)), max(0.0001, min(1.0, h))]

            is_dup = False
            for p_cls, p_box in valid_boxes:
                if p_cls == cls_id and compute_iou(c_box, p_box) > 0.95:
                    issues.append((txt.name, l_num, "DUPLICATE_BOX", "IoU > 0.95 with prior box"))
                    has_error = True
                    is_dup = True
                    break

            if not is_dup:
                valid_boxes.append((cls_id, c_box))

        if fix and has_error:
            # Backup original
            shutil.copy2(str(txt), str(txt.with_suffix(".txt.bak")))
            out_lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in valid_boxes]
            txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            files_fixed += 1

    print("=" * 60)
    print(f"Validation Report for {labels_dir}:")
    print(f"  Total Files Scanned: {len(txt_files)}")
    print(f"  Total Issues Found:  {len(issues)}")
    if fix:
        print(f"  Files Auto-Repaired: {files_fixed} (originals backed up as .bak)")
    print("=" * 60)

    if issues:
        print("\nTop Issues Sample:")
        for iss in issues[:10]:
            print(f"  [{iss[2]}] {iss[0]} (Line {iss[1]}): {iss[3]}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more.")

    if report_path and issues:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["FileName", "LineNumber", "IssueType", "Details"])
            writer.writerows(issues)
        print(f"\nFull issues log written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate and repair YOLO annotation files.")
    parser.add_argument("labels_folder", type=Path, help="Folder containing .txt labels")
    parser.add_argument("--classes-file", type=Path, default=None, help="Path to classes.txt")
    parser.add_argument("--fix", action="store_true", help="Auto-repair out-of-bounds coordinates & duplicate boxes")
    parser.add_argument("--report", type=Path, default=None, help="Save issues report to CSV")
    args = parser.parse_args()

    validate_folder(args.labels_folder, args.classes_file, args.fix, args.report)


if __name__ == "__main__":
    main()
