"""Automated YOLOv8 Model Inference and Dataset Annotation Generator.

Recursively or flatly annotates raw image folders using pre-trained or custom YOLO models.
Outputs standard normalized YOLO annotations (class_id x_center y_center width height)
with options for confidence filtering, class remapping, classes.txt generation,
and comprehensive annotation summary reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def run_auto_annotate(
    input_dir: Path,
    output_labels_dir: Path,
    weights: Path,
    confidence: float = 0.5,
    iou_thresh: float = 0.45,
    filter_classes: Optional[List[int]] = None,
    class_mapping: Optional[Dict[int, int]] = None,
    classes_file: Optional[Path] = None,
    recursive: bool = False,
    overwrite: bool = True,
    create_empty_for_background: bool = True,
) -> Dict[str, int]:
    """Execute model prediction and format bounding boxes as YOLO .txt files."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input image directory does not exist: {input_dir}")
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights file not found: {weights}")

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model weights: {weights}...")
    model = YOLO(str(weights))
    model_names: Dict[int, str] = model.names or {}

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    image_paths = sorted([
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        print(f"No images found in {input_dir}")
        return {"total_images": 0, "annotated_images": 0, "total_boxes": 0}

    print(f"Found {len(image_paths)} image(s) to process. Running inference...")

    total_images = len(image_paths)
    annotated_images = 0
    total_boxes = 0
    class_counts: Dict[int, int] = {}

    for idx, img_path in enumerate(image_paths, start=1):
        txt_path = output_labels_dir / f"{img_path.stem}.txt"
        if txt_path.exists() and not overwrite:
            continue

        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=confidence,
            iou=iou_thresh,
            classes=filter_classes,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        lines: List[str] = []
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for cls_id, box in zip(classes, xywhn):
                final_cls_id = cls_id
                if class_mapping is not None:
                    if cls_id not in class_mapping:
                        continue
                    final_cls_id = class_mapping[cls_id]

                xc, yc, w, h = box
                lines.append(f"{final_cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_counts[final_cls_id] = class_counts.get(final_cls_id, 0) + 1
                total_boxes += 1

        if lines:
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            annotated_images += 1
        elif create_empty_for_background:
            # 0-byte file represents background/negative image in YOLO
            txt_path.touch()

        if idx % 100 == 0 or idx == total_images:
            print(f"Progress: [{idx}/{total_images}] images processed.")

    # Generate classes.txt if requested
    if classes_file is not None:
        classes_file.parent.mkdir(parents=True, exist_ok=True)
        max_id = max(class_counts.keys()) if class_counts else len(model_names) - 1
        class_lines = [model_names.get(i, f"class_{i}") for i in range(max_id + 1)]
        classes_file.write_text("\n".join(class_lines) + "\n", encoding="utf-8")
        print(f"Generated classes file at: {classes_file}")

    print("=" * 60)
    print("Auto-Annotation Summary:")
    print(f"Total Images: {total_images} | Images with Detections: {annotated_images} | Total Boxes: {total_boxes}")
    for cid, cnt in sorted(class_counts.items()):
        cname = model_names.get(cid, f"class_{cid}")
        print(f"  Class {cid} ({cname}): {cnt} instances")
    print("=" * 60)

    return {
        "total_images": total_images,
        "annotated_images": annotated_images,
        "total_boxes": total_boxes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate YOLO annotations using YOLOv8 model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Directory containing raw images to annotate")
    parser.add_argument(
        "--output-labels", type=Path, default=None,
        help="Directory to save .txt annotations (default: input_dir/labels)",
    )
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"), help="YOLOv8 model weights (.pt)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Filter specific class IDs to detect")
    parser.add_argument("--remap", type=str, default=None, help="Remap classes, e.g. '0:0,5:1' or JSON mapping")
    parser.add_argument("--classes-file", type=Path, default=None, help="Path to write classes.txt")
    parser.add_argument("--recursive", action="store_true", help="Recursively traverse input subdirectories")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing .txt annotations")

    args = parser.parse_args()

    out_labels = args.output_labels or (args.input / "labels")

    mapping = None
    if args.remap:
        if Path(args.remap).is_file():
            raw_map = json.loads(Path(args.remap).read_text(encoding="utf-8"))
            mapping = {int(k): int(v) for k, v in raw_map.items()}
        else:
            mapping = {}
            for item in args.remap.split(","):
                k, v = item.strip().split(":")
                mapping[int(k)] = int(v)

    try:
        run_auto_annotate(
            args.input,
            out_labels,
            args.weights,
            confidence=args.confidence,
            iou_thresh=args.iou,
            filter_classes=args.classes,
            class_mapping=mapping,
            classes_file=args.classes_file,
            recursive=args.recursive,
            overwrite=not args.no_overwrite,
        )
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
