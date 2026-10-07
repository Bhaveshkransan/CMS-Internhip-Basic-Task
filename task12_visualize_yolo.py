"""Standalone Script: Visualize YOLO Bounding Boxes on Images.

Draws color-coded bounding boxes and class names directly on images
and saves rendered copies to a folder for quick visual quality checks.
"""

import argparse
import random
from pathlib import Path
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]


def visualize(images_dir: Path, output_dir: Path, labels_dir: Path | None = None, classes_file: Path | None = None, samples: int = 20) -> None:
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    lbl_dir = labels_dir or images_dir
    labels = {p.stem.casefold(): p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

    annotated = [(img, labels[img.stem.casefold()]) for img in images if img.stem.casefold() in labels and labels[img.stem.casefold()].stat().st_size > 0]
    if not annotated:
        print("No annotated images found to visualize.")
        return

    class_names = {}
    if classes_file and classes_file.is_file():
        lines = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        class_names = {idx: name for idx, name in enumerate(lines)}

    sample_items = random.sample(annotated, min(samples, len(annotated)))
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in sample_items:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    c = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    x1 = int((xc - w / 2) * w_img)
                    y1 = int((yc - h / 2) * h_img)
                    x2 = int((xc + w / 2) * w_img)
                    y2 = int((yc + h / 2) * h_img)

                    color = COLORS[c % len(COLORS)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label_str = class_names.get(c, f"ID:{c}")
                    cv2.putText(img, label_str, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except ValueError:
                    pass

        cv2.imwrite(str(output_dir / f"viz_{img_path.name}"), img)

    print(f"Rendered {len(sample_items)} visual inspection sample(s) to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render YOLO bounding boxes onto images.")
    parser.add_argument("images_folder", type=Path, help="Folder containing images")
    parser.add_argument("--output", type=Path, default=Path("visualized_samples"), help="Output folder")
    parser.add_argument("--labels-folder", type=Path, default=None, help="Folder containing labels (if separate)")
    parser.add_argument("--classes-file", type=Path, default=None, help="Path to classes.txt")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to render")
    args = parser.parse_args()

    visualize(args.images_folder, args.output, args.labels_folder, args.classes_file, args.samples)


if __name__ == "__main__":
    main()
