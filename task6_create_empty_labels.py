"""Standalone Script: Create Empty YOLO Labels for Background Images.

Batch-creates 0-byte .txt files for every image that lacks an annotation file.
Essential in YOLO training so background/negative images prevent false positives.
"""

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def create_empty_labels(images_dir: Path, labels_dir: Path | None = None) -> None:
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    target_dir = labels_dir or images_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    created_count = 0
    for img in images:
        txt_path = target_dir / f"{img.stem}.txt"
        if not txt_path.exists():
            txt_path.touch()
            created_count += 1

    print(f"Created {created_count} empty (0-byte) YOLO label file(s) in {target_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Create empty .txt files for unannotated/background images.")
    parser.add_argument("images_folder", type=Path, help="Folder containing images")
    parser.add_argument("--labels-folder", type=Path, default=None, help="Folder where .txt files should be created")
    args = parser.parse_args()

    create_empty_labels(args.images_folder, args.labels_folder)


if __name__ == "__main__":
    main()
