"""Standalone Script: Separate Background Images & Orphaned Labels.

Safely isolates:
  1. Images without matching .txt files -> moved to background_images folder
  2. .txt files without matching images -> moved to orphaned_labels folder
Replaces destructive deletion with non-destructive quarantine.
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional, Set

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def separate_background(
    dataset_dir: Path,
    bg_dest: Optional[Path] = None,
    orphan_dest: Optional[Path] = None,
    mode: str = "move",
) -> None:
    images = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt"]

    img_stems = {img.stem.casefold(): img for img in images}
    lbl_stems = {lbl.stem.casefold(): lbl for lbl in labels}

    # Find unannotated images and orphaned labels
    unannotated_imgs = [img for stem, img in img_stems.items() if stem not in lbl_stems]
    orphaned_lbls = [lbl for stem, lbl in lbl_stems.items() if stem not in img_stems]

    target_bg = bg_dest or (dataset_dir / "background_images")
    target_orphan = orphan_dest or (dataset_dir / "orphaned_labels")

    target_bg.mkdir(parents=True, exist_ok=True)
    target_orphan.mkdir(parents=True, exist_ok=True)

    action = shutil.move if mode == "move" else shutil.copy2

    for img in unannotated_imgs:
        action(str(img), str(target_bg / img.name))

    for lbl in orphaned_lbls:
        action(str(lbl), str(target_orphan / lbl.name))

    print("=" * 60)
    print(f"Separation Complete [{mode.upper()} Mode]:")
    print(f"  Moved {len(unannotated_imgs)} unannotated image(s) to: {target_bg}")
    print(f"  Moved {len(orphaned_lbls)} orphaned label(s) to:    {target_orphan}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Separate unannotated/background images and orphaned labels.")
    parser.add_argument("folder", type=Path, help="Folder containing images and labels")
    parser.add_argument("--bg-dest", type=Path, default=None, help="Destination folder for background images")
    parser.add_argument("--orphan-dest", type=Path, default=None, help="Destination folder for orphaned labels")
    parser.add_argument("--mode", choices=["move", "copy"], default="move", help="File action (move or copy)")
    args = parser.parse_args()

    separate_background(args.folder, args.bg_dest, args.orphan_dest, args.mode)


if __name__ == "__main__":
    main()
