"""Standalone Script: Rename Images and Matching YOLO Annotations.

Renames image files and corresponding .txt label files simultaneously.
Uses safe two-phase atomic staging to prevent overwriting or data loss.
Supports custom prefix, digit padding, starting index, and dry-run preview.
"""

import argparse
import csv
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_images_and_labels(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Path]]:
    """Pair image files with matching .txt files by stem name."""
    images = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.casefold(),
    )
    lbl_dir = labels_dir if labels_dir is not None else images_dir
    labels = sorted(
        [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt"],
        key=lambda p: p.name.casefold(),
    )

    label_map = {lbl.stem.casefold(): lbl for lbl in labels}
    matched_stems: Set[str] = set()
    pairs: List[Tuple[Path, Optional[Path]]] = []

    for img in images:
        stem = img.stem.casefold()
        lbl = label_map.get(stem)
        if lbl:
            matched_stems.add(stem)
            pairs.append((img, lbl))
        else:
            pairs.append((img, None))

    orphans = [lbl for lbl in labels if lbl.stem.casefold() not in matched_stems]
    return pairs, orphans


def rename_pairs(
    images_dir: Path,
    labels_dir: Optional[Path] = None,
    prefix: str = "image_",
    start_index: int = 1,
    padding: int = 4,
    dry_run: bool = False,
    manifest: Optional[Path] = None,
) -> int:
    pairs, orphans = get_images_and_labels(images_dir, labels_dir)
    if not pairs:
        print(f"No images found in {images_dir}")
        return 0

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
            new_lbl_str = new_lbl.name

        manifest_rows.append({
            "original_image": img.name,
            "new_image": new_img.name,
            "original_label": lbl.name if lbl else "",
            "new_label": new_lbl_str,
        })

    if dry_run:
        print(f"\n[DRY-RUN] Preview of {len(pairs)} pairs to be renamed:")
        for r in manifest_rows[:10]:
            print(f"  {r['original_image']} -> {r['new_image']}  |  Label: {r['original_label']} -> {r['new_label']}")
        if len(manifest_rows) > 10:
            print(f"  ... and {len(manifest_rows) - 10} more.")
        return len(pairs)

    # Two-phase atomic rename using temporary UUIDs to prevent file overwrites
    staged: List[Tuple[Path, Path, Path]] = []
    completed: List[Tuple[Path, Path, Path]] = []
    try:
        for src, dst in rename_batch:
            tmp = src.parent / f".tmp_{uuid.uuid4().hex}_{src.name}"
            src.rename(tmp)
            staged.append((src, tmp, dst))

        for src, tmp, dst in staged:
            tmp.rename(dst)
            completed.append((src, tmp, dst))
    except Exception as exc:
        for _, tmp, dst in reversed(completed):
            if dst.exists():
                dst.rename(tmp)
        for src, tmp, _ in reversed(staged):
            if tmp.exists():
                tmp.rename(src)
        raise OSError(f"Rename failed and rolled back safely: {exc}")

    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["original_image", "new_image", "original_label", "new_label"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Rename log manifest saved to: {manifest}")

    print(f"Successfully renamed {len(pairs)} image-annotation pair(s).")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Rename images and matching YOLO .txt annotations together.")
    parser.add_argument("input_folder", type=Path, help="Folder containing images (and optionally labels)")
    parser.add_argument("--labels-folder", type=Path, default=None, help="Separate labels folder (if not in same folder)")
    parser.add_argument("--prefix", type=str, default="image_", help="Naming prefix (e.g. 'bhavesh_ubi_')")
    parser.add_argument("--start-idx", type=int, default=1, help="Starting number")
    parser.add_argument("--padding", type=int, default=4, help="Digit padding (e.g. 4 -> 0001)")
    parser.add_argument("--manifest", type=Path, default=None, help="Save rename mapping log to CSV")
    parser.add_argument("--dry-run", action="store_true", help="Preview renaming without changing files")
    args = parser.parse_args()

    rename_pairs(
        args.input_folder,
        labels_dir=args.labels_folder,
        prefix=args.prefix,
        start_index=args.start_idx,
        padding=args.padding,
        dry_run=args.dry_run,
        manifest=args.manifest,
    )


if __name__ == "__main__":
    main()
