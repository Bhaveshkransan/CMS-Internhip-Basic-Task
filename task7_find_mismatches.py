"""Standalone Script: Find Dataset Mismatches (Unpaired Images & Labels).

Checks for:
  - Images missing .txt annotation files
  - .txt annotation files missing corresponding images
Exports detailed reports in TXT or CSV format.
"""

import argparse
import csv
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_mismatches(images_dir: Path, labels_dir: Path | None = None, report_path: Path | None = None) -> None:
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    lbl_dir = labels_dir or images_dir
    labels = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt"]

    img_stems = {img.stem.casefold(): img for img in images}
    lbl_stems = {lbl.stem.casefold(): lbl for lbl in labels}

    missing_labels = [img for stem, img in img_stems.items() if stem not in lbl_stems]
    missing_images = [lbl for stem, lbl in lbl_stems.items() if stem not in img_stems]

    print("=" * 60)
    print(f"Dataset Audit Results for {images_dir}:")
    print(f"  Total Images: {len(images)} | Total Labels: {len(labels)}")
    print(f"  Images without labels: {len(missing_labels)}")
    print(f"  Labels without images: {len(missing_images)}")
    print("=" * 60)

    if missing_labels:
        print("\nImages missing labels (first 10):")
        for img in missing_labels[:10]:
            print(f"  {img.name}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more.")

    if missing_images:
        print("\nLabels missing images (first 10):")
        for lbl in missing_images[:10]:
            print(f"  {lbl.name}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more.")

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.suffix.lower() == ".csv":
            with report_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Type", "FileName", "FullPath"])
                for p in missing_labels:
                    writer.writerow(["ImageMissingLabel", p.name, str(p)])
                for p in missing_images:
                    writer.writerow(["LabelMissingImage", p.name, str(p)])
        else:
            lines = [
                f"# Dataset Mismatch Audit Report",
                f"Images Folder: {images_dir}",
                f"Labels Folder: {lbl_dir}",
                "",
                f"Images Missing Labels ({len(missing_labels)}):",
                *(f"  {p.name}" for p in missing_labels),
                "",
                f"Labels Missing Images ({len(missing_images)}):",
                *(f"  {p.name}" for p in missing_images),
            ]
            report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nReport written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Find mismatched image and YOLO label files.")
    parser.add_argument("images_folder", type=Path, help="Folder containing images")
    parser.add_argument("--labels-folder", type=Path, default=None, help="Folder containing labels (if separate)")
    parser.add_argument("--report", type=Path, default=None, help="Save report to .txt or .csv")
    args = parser.parse_args()

    find_mismatches(args.images_folder, args.labels_folder, args.report)


if __name__ == "__main__":
    main()
