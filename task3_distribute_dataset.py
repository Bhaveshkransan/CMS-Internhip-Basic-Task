"""Standalone Script: Distribute Dataset Pairs into Multiple Folders.

Distributes images AND their corresponding YOLO .txt annotations across N folders
(e.g., folder_01, folder_02) or by batch size (e.g., 500 images per batch).
Ensures matching annotations are never lost when dividing work among annotators.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def distribute(
    source_dir: Path,
    dest_dir: Path,
    labels_dir: Optional[Path] = None,
    num_folders: Optional[int] = None,
    batch_size: Optional[int] = None,
    mode: str = "copy",
    manifest_path: Optional[Path] = None,
) -> int:
    images = sorted(
        [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.casefold(),
    )
    lbl_dir = labels_dir if labels_dir is not None else source_dir
    labels = sorted(
        [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.name.casefold(),
    )
    label_map = {lbl.stem.casefold(): lbl for lbl in labels}

    pairs: List[Tuple[Path, Optional[Path]]] = [
        (img, label_map.get(img.stem.casefold())) for img in images
    ]

    total_pairs = len(pairs)
    if total_pairs == 0:
        print(f"No images found in {source_dir}")
        return 0

    if num_folders is None and batch_size is None:
        raise ValueError("Specify either --num-folders N or --batch-size K.")

    if num_folders is not None:
        num_folders = min(num_folders, total_pairs)
        base = total_pairs // num_folders
        extra = total_pairs % num_folders
        counts = [base + (1 if i < extra else 0) for i in range(num_folders)]
    else:
        assert batch_size is not None
        num_folders = math.ceil(total_pairs / batch_size)
        counts = []
        rem = total_pairs
        for _ in range(num_folders):
            c = min(batch_size, rem)
            counts.append(c)
            rem -= c

    dest_dir.mkdir(parents=True, exist_ok=True)
    action = shutil.move if mode == "move" else shutil.copy2
    manifest_rows: List[Dict[str, str]] = []
    idx = 0

    for f_idx, count in enumerate(counts, start=1):
        f_name = f"folder_{f_idx:02d}"
        folder_path = dest_dir / f_name
        folder_path.mkdir(parents=True, exist_ok=True)

        for _ in range(count):
            img, lbl = pairs[idx]
            idx += 1

            action(str(img), str(folder_path / img.name))
            lbl_name = ""
            if lbl is not None:
                action(str(lbl), str(folder_path / lbl.name))
                lbl_name = lbl.name

            manifest_rows.append({
                "assigned_folder": f_name,
                "image": img.name,
                "label": lbl_name,
            })

    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["assigned_folder", "image", "label"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Distribution manifest saved to: {manifest_path}")

    print(f"Successfully distributed {total_pairs} pairs into {len(counts)} folders in {dest_dir}.")
    return total_pairs


def main():
    parser = argparse.ArgumentParser(description="Distribute images and matching annotations into multiple folders.")
    parser.add_argument("source_folder", type=Path, help="Folder containing images and labels")
    parser.add_argument("--dest", type=Path, required=True, help="Destination folder where subfolders will be created")
    parser.add_argument("--labels-folder", type=Path, default=None, help="Separate labels folder (if not in source)")
    parser.add_argument("--num-folders", type=int, default=None, help="Number of destination folders (e.g. 10)")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of images per folder (e.g. 500)")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="File action (copy or move)")
    parser.add_argument("--manifest", type=Path, default=None, help="Save distribution log to CSV")
    args = parser.parse_args()

    distribute(
        args.source_folder,
        args.dest,
        labels_dir=args.labels_folder,
        num_folders=args.num_folders,
        batch_size=args.batch_size,
        mode=args.mode,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
