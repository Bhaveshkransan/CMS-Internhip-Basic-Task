"""High-Performance Video Analytics, Intrusion, Loitering, and Line-Crossing Suite using YOLOv8.

Modes supported:
  - all: Detect all COCO or custom trained classes.
  - person: Track person class (COCO 0).
  - intrusion: Detect person breaching a restricted polygon ROI.
  - loitering: Detect person remaining inside ROI exceeding customizable threshold.
  - line-crossing: Bidirectional virtual tripwire counting (IN / OUT).
  - occupancy: Real-time headcount and capacity monitoring within ROI zone.

Features:
  - Real-time on-screen HUD with active loiter dwell timers ([ID: 2] 14.5s / 30.0s).
  - Visual threat color coding (Green -> Yellow -> Red).
  - Automatic event logging to CSV / JSON audit files.
  - Alert snapshot image export for security investigations.
  - Headless server mode (--no-display).
  - Webcams (0), video files, and RTSP stream sources.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

PERSON_CLASS_ID = 0
DEFAULT_LOITER_SECONDS = 30.0


# --------------------------------------------------------------------------- #
# Geometry & Spatial Helpers
# --------------------------------------------------------------------------- #

def parse_points(values: Optional[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """Parse coordinate arguments in format ['100,100', '500,100', '500,400']."""
    if not values:
        return None
    try:
        points = [tuple(map(int, v.strip().split(","))) for v in values]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Points must be formatted as X,Y (e.g. 100,200). Error: {exc}")
    return points


def interactive_select_polygon(first_frame: np.ndarray, window_title: str = "Select ROI Polygon") -> List[Tuple[int, int]]:
    """Interactive GUI callback to draw a polygon ROI on the first frame."""
    points: List[Tuple[int, int]] = []
    window = f"{window_title} (Click points | Enter: finish | R: reset | Esc: cancel)"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            display = first_frame.copy()
            for p in points:
                cv2.circle(display, p, 5, (0, 255, 255), -1)
            if len(points) > 1:
                cv2.polylines(display, [np.array(points, dtype="int32")], len(points) >= 3, (0, 255, 255), 2)

            cv2.imshow(window, display)
            key = cv2.waitKey(25) & 0xFF
            if key in (10, 13) and len(points) >= 3:
                return points.copy()
            if key == ord("r") or key == ord("R"):
                points.clear()
            if key == 27:  # Esc
                raise ValueError("ROI selection was cancelled by user.")
    finally:
        cv2.destroyWindow(window)


def interactive_select_line(first_frame: np.ndarray) -> List[Tuple[int, int]]:
    """Interactive GUI callback to draw a virtual tripwire line (2 points)."""
    points: List[Tuple[int, int]] = []
    window = "Select Tripwire Line (Click 2 points: Start -> End | Enter: finish | R: reset | Esc: cancel)"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            display = first_frame.copy()
            for p in points:
                cv2.circle(display, p, 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 0, 255), 3)

            cv2.imshow(window, display)
            key = cv2.waitKey(25) & 0xFF
            if key in (10, 13) and len(points) == 2:
                return points.copy()
            if key == ord("r") or key == ord("R"):
                points.clear()
            if key == 27:
                raise ValueError("Tripwire selection was cancelled.")
    finally:
        cv2.destroyWindow(window)


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Check if point (x, y) is inside or on the edge of the polygon."""
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def line_side(line_p1: Tuple[int, int], line_p2: Tuple[int, int], pt: Tuple[float, float]) -> float:
    """Calculate which side of a directed line segment a point lies on (signed determinant)."""
    return (line_p2[0] - line_p1[0]) * (pt[1] - line_p1[1]) - (line_p2[1] - line_p1[1]) * (pt[0] - line_p1[0])


# --------------------------------------------------------------------------- #
# Event Logging & Snapshot Capture
# --------------------------------------------------------------------------- #

class AlertLogger:
    def __init__(self, log_path: Optional[Path], snapshot_dir: Optional[Path]):
        self.log_path = log_path
        self.snapshot_dir = snapshot_dir
        self.records: List[Dict[str, Any]] = []

        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.suffix.lower() == ".csv" and not self.log_path.exists():
                with self.log_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp", "VideoTimeSec", "FrameIndex", "TrackID",
                        "EventType", "ZoneName", "DurationSec", "BBox", "SnapshotPath"
                    ])

    def log_event(
        self,
        event_type: str,
        track_id: int,
        frame_idx: int,
        video_time_sec: float,
        frame_img: np.ndarray,
        bbox: List[float],
        zone_name: str = "Zone_1",
        duration: float = 0.0,
    ) -> None:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot_filename = ""

        if self.snapshot_dir is not None:
            snap_name = f"alert_{event_type.lower()}_id{track_id}_f{frame_idx}_{int(video_time_sec)}s.jpg"
            snap_path = self.snapshot_dir / snap_name
            cv2.imwrite(str(snap_path), frame_img)
            snapshot_filename = str(snap_path.name)

        record = {
            "Timestamp": now_str,
            "VideoTimeSec": round(video_time_sec, 2),
            "FrameIndex": frame_idx,
            "TrackID": track_id,
            "EventType": event_type,
            "ZoneName": zone_name,
            "DurationSec": round(duration, 2),
            "BBox": [round(x, 1) for x in bbox],
            "SnapshotPath": snapshot_filename,
        }
        self.records.append(record)

        if self.log_path is not None:
            if self.log_path.suffix.lower() == ".csv":
                with self.log_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        record["Timestamp"], record["VideoTimeSec"], record["FrameIndex"],
                        record["TrackID"], record["EventType"], record["ZoneName"],
                        record["DurationSec"], str(record["BBox"]), record["SnapshotPath"]
                    ])
            elif self.log_path.suffix.lower() == ".json":
                self.log_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Core Pipeline Execution
# --------------------------------------------------------------------------- #

def run_analytics(args: argparse.Namespace) -> None:
    # 1. Open Source (File, Webcam, or RTSP)
    source_str = str(args.video)
    if source_str.isdigit():
        source_input: Any = int(source_str)
    else:
        source_input = source_str

    capture = cv2.VideoCapture(source_input)
    if not capture.isOpened():
        raise ValueError(f"Could not open video source: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = capture.read()
    if not ret or first_frame is None:
        capture.release()
        raise ValueError(f"Unable to read initial frame from {args.video}")

    # 2. Setup Spatial Geometries (ROI Polygon or Tripwire Line)
    polygon = None
    tripwire_line = None

    if args.mode in {"intrusion", "loitering", "occupancy"}:
        roi_pts = parse_points(args.roi)
        if roi_pts is None:
            if args.no_display:
                capture.release()
                raise ValueError("Headless mode (--no-display) requires --roi coordinates.")
            roi_pts = interactive_select_polygon(first_frame, f"Select {args.mode.upper()} ROI")
        polygon = np.array(roi_pts, dtype="int32")

    elif args.mode == "line-crossing":
        line_pts = parse_points(args.line)
        if line_pts is None:
            if args.no_display:
                capture.release()
                raise ValueError("Headless mode (--no-display) requires --line coordinates.")
            line_pts = interactive_select_line(first_frame)
        if len(line_pts) != 2:
            raise ValueError("Tripwire requires exactly 2 points: --line X1,Y1 X2,Y2")
        tripwire_line = (line_pts[0], line_pts[1])

    # 3. Load YOLO Model
    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    # 4. Video Writer Setup
    writer = None
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise OSError(f"Could not create output video writer: {args.save}")

    # 5. Alert & State Trackers
    alert_logger = AlertLogger(args.alert_log, args.save_snapshots)
    loiter_threshold = args.loiter_seconds if args.loiter_seconds is not None else DEFAULT_LOITER_SECONDS

    # State tracking:
    # active_inside: track_id -> entry_video_time_sec
    active_inside: Dict[int, float] = {}
    loiter_alerted: Set[int] = set()

    # Line crossing state: track_id -> previous_side (+1 or -1)
    prev_line_sides: Dict[int, float] = {}
    in_count = 0
    out_count = 0

    frame_index = 0
    curr_frame = first_frame
    conf = args.confidence or (0.5 if args.mode != "all" else 0.25)

    print(f"Starting YOLOv8 Video Analytics [{args.mode.upper()} Mode] on {args.video}...")

    try:
        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if frame_index > 0:
                ret, curr_frame = capture.read()
                if not ret or curr_frame is None:
                    break

            frame_time = frame_index / fps
            annotated = curr_frame.copy()

            # Run YOLO Tracking (ByteTrack)
            if args.mode in {"intrusion", "loitering", "line-crossing", "occupancy", "person"}:
                results = model.track(
                    curr_frame,
                    persist=True,
                    classes=[PERSON_CLASS_ID] if args.mode != "all" else None,
                    conf=conf,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            else:
                results = model.predict(curr_frame, conf=conf, verbose=False)

            result = results[0]
            boxes = result.boxes

            current_frame_inside_ids: Set[int] = set()
            active_threat_intrusion = False
            active_threat_loiter = False

            if boxes is not None and len(boxes) > 0:
                coords = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for box_xyxy, cls_id, track_id in zip(coords, cls_ids, track_ids):
                    x1, y1, x2, y2 = box_xyxy
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    bottom_center = ((x1 + x2) / 2.0, float(y2))

                    is_inside = False
                    if polygon is not None:
                        is_inside = point_in_polygon(bottom_center, polygon)

                    # --- Mode: Intrusion & Loitering & Occupancy ---
                    if is_inside and track_id is not None:
                        current_frame_inside_ids.add(track_id)

                        if args.mode == "intrusion":
                            active_threat_intrusion = True
                            if track_id not in active_inside:
                                print(f"⚠️ [INTRUSION ALERT] Person Track ID {track_id} entered zone at {frame_time:.1f}s!")
                                alert_logger.log_event("INTRUSION", track_id, frame_index, frame_time, curr_frame, [x1, y1, x2, y2])
                            active_inside[track_id] = frame_time

                        elif args.mode == "loitering":
                            entry_time = active_inside.get(track_id, frame_time)
                            active_inside[track_id] = entry_time
                            dwell_time = frame_time - entry_time

                            # Threat color progression
                            ratio = min(1.0, dwell_time / loiter_threshold)
                            if ratio < 0.5:
                                box_color = (0, 255, 0)      # Green
                            elif ratio < 1.0:
                                box_color = (0, 215, 255)    # Yellow / Orange Warning
                            else:
                                box_color = (0, 0, 255)      # Red Alert
                                active_threat_loiter = True

                                if track_id not in loiter_alerted:
                                    print(f"⚠️ [LOITERING ALERT] Track ID {track_id} loitering for {dwell_time:.1f}s (Threshold: {loiter_threshold}s)!")
                                    alert_logger.log_event("LOITERING", track_id, frame_index, frame_time, curr_frame, [x1, y1, x2, y2], duration=dwell_time)
                                    loiter_alerted.add(track_id)

                            # Visual dwell HUD over person
                            hud_text = f"ID:{track_id} | {dwell_time:.1f}s/{loiter_threshold:.0f}s"
                            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                            cv2.putText(
                                annotated, hud_text, (int(x1), max(20, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2
                            )

                    # --- Mode: Line Crossing (Tripwire) ---
                    if args.mode == "line-crossing" and tripwire_line is not None and track_id is not None:
                        current_side = line_side(tripwire_line[0], tripwire_line[1], bottom_center)
                        prev_side = prev_line_sides.get(track_id)

                        if prev_side is not None:
                            # Crossed from negative side to positive side -> IN
                            if prev_side < 0 and current_side > 0:
                                in_count += 1
                                print(f"➡️ [LINE CROSSING: IN] Track ID {track_id} crossed line. Total IN: {in_count}")
                                alert_logger.log_event("LINE_IN", track_id, frame_index, frame_time, curr_frame, [x1, y1, x2, y2])
                            # Crossed from positive side to negative side -> OUT
                            elif prev_side > 0 and current_side < 0:
                                out_count += 1
                                print(f"⬅️ [LINE CROSSING: OUT] Track ID {track_id} crossed line. Total OUT: {out_count}")
                                alert_logger.log_event("LINE_OUT", track_id, frame_index, frame_time, curr_frame, [x1, y1, x2, y2])

                        prev_line_sides[track_id] = current_side

                    # Standard detection box rendering for other modes
                    if args.mode not in {"loitering"}:
                        color = (0, 0, 255) if (is_inside and args.mode == "intrusion") else (255, 200, 0)
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        tag = f"ID:{track_id}" if track_id is not None else f"cls:{cls_id}"
                        cv2.putText(annotated, tag, (int(x1), max(20, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Cleanup exited track IDs
            if args.mode in {"intrusion", "loitering"}:
                active_inside = {
                    tid: t for tid, t in active_inside.items()
                    if tid in current_frame_inside_ids
                }
                loiter_alerted = {
                    tid for tid in loiter_alerted
                    if tid in current_frame_inside_ids
                }

            # --- HUD Overlay & Zone Rendering ---
            if polygon is not None:
                poly_color = (0, 0, 255) if (active_threat_intrusion or active_threat_loiter) else (0, 255, 255)
                cv2.polylines(annotated, [polygon], True, poly_color, 2)

            if tripwire_line is not None:
                cv2.line(annotated, tripwire_line[0], tripwire_line[1], (0, 0, 255), 3)

            # Top Banner HUD
            hud_bg_color = (20, 20, 20)
            cv2.rectangle(annotated, (0, 0), (width, 50), hud_bg_color, -1)
            time_str = f"Time: {frame_time:.1f}s | Frame: {frame_index}"

            if args.mode == "intrusion":
                status_text = "STATUS: BREACH / INTRUSION!" if active_threat_intrusion else "STATUS: ZONE SECURE"
                status_color = (0, 0, 255) if active_threat_intrusion else (0, 255, 0)
                cv2.putText(annotated, f"INTRUSION DETECTOR | {status_text}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            elif args.mode == "loitering":
                loiter_count = len(loiter_alerted)
                status_color = (0, 0, 255) if loiter_count > 0 else (0, 255, 0)
                cv2.putText(annotated, f"LOITERING MONITOR | Inside: {len(current_frame_inside_ids)} | Alerted: {loiter_count}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            elif args.mode == "line-crossing":
                cv2.putText(annotated, f"TRIPWIRE COUNT | IN: {in_count} | OUT: {out_count} | Net: {in_count - out_count}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif args.mode == "occupancy":
                count = len(current_frame_inside_ids)
                max_occ = args.max_occupancy or 50
                occ_color = (0, 0, 255) if count > max_occ else (0, 255, 0)
                cv2.putText(annotated, f"ZONE OCCUPANCY | Count: {count} (Capacity: {max_occ})", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, occ_color, 2)

            else:
                cv2.putText(annotated, f"YOLOv8 DETECTIONS | {time_str}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Record frame if writer is active
            if writer is not None:
                writer.write(annotated)

            # Live preview window
            if not args.no_display:
                cv2.imshow("YOLOv8 Edge Video Analytics", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Processing aborted by user (pressed 'q').")
                    break

            frame_index += 1

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Analysis complete: Processed {frame_index} frame(s).")
    if args.alert_log:
        print(f"Alert audit logs written to: {args.alert_log}")
    if args.save_snapshots:
        print(f"Alert snapshots captured in: {args.save_snapshots}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise-grade YOLOv8 Video Analytics, Intrusion, Loitering, and Line Crossing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, help="Path to video file, RTSP URL, or webcam index (0)")
    parser.add_argument(
        "--mode",
        choices=["all", "person", "intrusion", "loitering", "line-crossing", "occupancy"],
        default="all",
        help="Detection analysis mode",
    )
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"), help="YOLOv8 model weights")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold (0.0 - 1.0)")
    parser.add_argument("--loiter-seconds", type=float, default=30.0, help="Loitering alert dwell threshold in seconds")
    parser.add_argument("--max-occupancy", type=int, default=10, help="Capacity limit for occupancy mode alerts")

    parser.add_argument("--roi", nargs="+", metavar="X,Y", help="Polygon ROI points (e.g. --roi 100,100 500,100 500,400 100,400)")
    parser.add_argument("--line", nargs="+", metavar="X,Y", help="Tripwire line coordinates (e.g. --line 200,100 200,500)")

    parser.add_argument("--save", type=Path, default=None, help="Output path for annotated video (.mp4)")
    parser.add_argument("--alert-log", type=Path, default=None, help="Save alert audit log to CSV or JSON file")
    parser.add_argument("--save-snapshots", type=Path, default=None, help="Directory to save image snapshots of triggered alerts")
    parser.add_argument("--max-frames", type=int, default=None, help="Process up to N frames (useful for testing or batch chunks)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without opening an OpenCV GUI window")

    args = parser.parse_args()

    if args.confidence is not None and not (0.0 < args.confidence <= 1.0):
        parser.error("--confidence must be strictly between 0 and 1.")

    try:
        run_analytics(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
