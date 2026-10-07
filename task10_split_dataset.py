"""Standalone Script: Split Dataset into Train / Validation / Test with data.yaml.

Splits image-label pairs while keeping them strictly synchronized.
Outputs the standard YOLO directory hierarchy:
  output_dir/
    images/train, images/val, images/test
    labels/train, labels/val, labels/test
    data.yaml
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def split(
    images_dir: Path,
    output_dir: Path,
    labels_dir: Optional[Path] = None,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    classes_file: Optional[Path] = None,
    seed: int = 42,
    mode: str = "copy",
) -> None:
    images = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.casefold(),
    )
    lbl_dir = labels_dir or images_dir
    labels = {p.stem.casefold(): p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

    pairs = [(img, labels.get(img.stem.casefold())) for img in images]
    if not pairs:
        print(f"No images found in {images_dir}")
        return

    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    r_tr, r_val, r_ts = ratios
    n_tr = int(n_total * r_tr)
    n_val = int(n_total * r_val)

    partitions = {
        "train": shuffled[:n_tr],
        "val": shuffled[n_tr:n_tr + n_val],
        "test": shuffled[n_tr + n_val:],
    }

    action = shutil.move if mode == "move" else shutil.copy2
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_pairs in partitions.items():
        if not split_pairs:
            continue
        dest_img = output_dir / "images" / split_name
        dest_lbl = output_dir / "labels" / split_name
        dest_img.mkdir(parents=True, exist_ok=True)
        dest_lbl.mkdir(parents=True, exist_ok=True)

        for img, lbl in split_pairs:
            action(str(img), str(dest_img / img.name))
            if lbl and lbl.exists():
                action(str(lbl), str(dest_lbl / lbl.name))
            else:
                (dest_lbl / f"{img.stem}.txt").touch()

    # Create data.yaml
    class_names = []
    if classes_file and classes_file.is_file():
        class_names = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    yaml_lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if partitions.get("test"):
        yaml_lines.append("test: images/test")
    yaml_lines.append("")
    yaml_lines.append(f"nc: {len(class_names)}")
    if class_names:
        yaml_lines.append("names:")
        for idx, name in enumerate(class_names):
            yaml_lines.append(f"  {idx}: {name}")

    yaml_file = output_dir / "data.yaml"
    yaml_file.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("=" * 60)
    print(f"Dataset Successfully Split into {output_dir}:")
    print(f"  Train: {len(partitions['train'])} pairs")
    print(f"  Val:   {len(partitions['val'])} pairs")
    print(f"  Test:  {len(partitions['test'])} pairs")
    print(f"  Generated YOLO config: {yaml_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test with synchronized YOLO pairs.")
    parser.add_argument("images_folder", type=Path, help="Folder containing images")
    parser.add_argument("--output", type=Path, required=True, help="Destination folder for split dataset")
    parser.add_argument("--labels-folder", type=Path, default=None, help="Folder containing labels (if separate)")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], help="Train Val Test split ratios")
    parser.add_argument("--classes-file", type=Path, default=None, help="Path to classes.txt")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="File action (copy or move)")
    args = parser.parse_args()

    split(
        args.images_folder,
        args.output,
        labels_dir=args.labels_folder,
        ratios=tuple(args.ratios),
        classes_file=args.classes_file,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
