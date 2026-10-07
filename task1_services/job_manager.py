"""
Thread-Safe Job & Batch Processing Manager.
Manages single and multi-video background processing, progress state, and pagination.
"""

import threading
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import UPLOAD_DIR, OUTPUT_DIR, ZIP_DIR
from .extractor import extract_video_to_images
from .zip_service import create_single_video_zip, create_multi_video_batch_zip

class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_batch_job(self, video_files: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """Initialize a new batch job tracking one or more videos."""
        batch_id = str(uuid.uuid4())
        job_dir = OUTPUT_DIR / batch_id
        job_dir.mkdir(parents=True, exist_ok=True)

        videos_state = []
        for v in video_files:
            video_id = str(uuid.uuid4())[:8]
            videos_state.append({
                "video_id": video_id,
                "video_path": v["saved_path"],
                "original_filename": v["filename"],
                "clean_name": Path(v["filename"]).stem,
                "info": v["info"],
                "status": "queued",
                "progress": 0.0,
                "current": 0,
                "total": config.get("target_count", 60),
                "stage": "Waiting in queue",
                "summary": None,
                "first_frame_url": None,
                "images": [],
                "zip_url": None,
                "error": None
            })

        job_state = {
            "batch_id": batch_id,
            "status": "queued",
            "created_at": None,
            "total_videos": len(videos_state),
            "completed_videos": 0,
            "config": config,
            "videos": videos_state,
            "master_zip_url": None,
            "error": None
        }

        with self._lock:
            self._jobs[batch_id] = job_state

        return batch_id

    def get_job(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(batch_id)

    def start_processing(self, batch_id: str):
        thread = threading.Thread(target=self._process_worker, args=(batch_id,), daemon=True)
        thread.start()

    def _process_worker(self, batch_id: str):
        with self._lock:
            job = self._jobs.get(batch_id)
            if not job:
                return
            job["status"] = "processing"

        config = job["config"]
        target_count = int(config.get("target_count", 60))
        img_fmt = config.get("format", "jpg").lower()
        quality = int(config.get("quality", 92))
        resize_dim = tuple(config["resize"]) if config.get("resize") else None

        batch_output_dir = OUTPUT_DIR / batch_id
        completed_video_dirs = []

        try:
            for v_entry in job["videos"]:
                video_id = v_entry["video_id"]
                v_entry["status"] = "processing"
                v_entry["stage"] = "Starting extraction"

                v_output_dir = batch_output_dir / f"{v_entry['clean_name']}_{video_id}"
                v_output_dir.mkdir(parents=True, exist_ok=True)

                def make_progress_callback(ve):
                    def cb(current, total, fname, stage):
                        with self._lock:
                            ve["current"] = current
                            ve["total"] = total
                            ve["progress"] = min(100.0, round((current / max(1, total)) * 100, 1))
                            ve["stage"] = f"{stage} ({current}/{total})"
                            if not ve["first_frame_url"] and current >= 1:
                                ve["first_frame_url"] = f"/api/jobs/{batch_id}/image/{ve['clean_name']}_{video_id}/{fname}"
                    return cb

                try:
                    summary = extract_video_to_images(
                        video_path=Path(v_entry["video_path"]),
                        output_dir=v_output_dir,
                        requested_count=target_count,
                        img_fmt=img_fmt,
                        quality=quality,
                        resize_dim=resize_dim,
                        progress_callback=make_progress_callback(v_entry)
                    )

                    # Build individual video zip
                    v_zip_name = f"extracted_{v_entry['clean_name']}_{video_id}.zip"
                    v_zip_path = ZIP_DIR / v_zip_name
                    create_single_video_zip(v_output_dir, v_zip_path)

                    with self._lock:
                        v_entry["status"] = "completed"
                        v_entry["stage"] = "Complete"
                        v_entry["progress"] = 100.0
                        v_entry["summary"] = summary
                        v_entry["images"] = [
                            {
                                "filename": itm["filename"],
                                "url": f"/api/jobs/{batch_id}/image/{v_entry['clean_name']}_{video_id}/{itm['filename']}",
                                "frame_idx": itm["frame_idx"],
                                "timestamp": itm["timestamp"],
                                "is_interpolated": itm["is_interpolated"],
                            }
                            for itm in summary["items"]
                        ]
                        v_entry["first_frame_url"] = v_entry["images"][0]["url"] if v_entry["images"] else None
                        v_entry["zip_url"] = f"/api/jobs/{batch_id}/download-zip/{video_id}"
                        job["completed_videos"] += 1
                        completed_video_dirs.append(v_output_dir)

                except Exception as ex:
                    with self._lock:
                        v_entry["status"] = "failed"
                        v_entry["stage"] = "Failed"
                        v_entry["error"] = str(ex)

            # Master ZIP for multi-video batch
            if len(completed_video_dirs) > 1:
                master_zip_name = f"batch_{batch_id[:8]}_all_videos.zip"
                master_zip_path = ZIP_DIR / master_zip_name
                create_multi_video_batch_zip(batch_output_dir, master_zip_path, completed_video_dirs)
                with self._lock:
                    job["master_zip_url"] = f"/api/jobs/{batch_id}/download-zip"
            elif len(completed_video_dirs) == 1:
                with self._lock:
                    job["master_zip_url"] = job["videos"][0]["zip_url"]

            with self._lock:
                job["status"] = "completed"

        except Exception as batch_ex:
            with self._lock:
                job["status"] = "failed"
                job["error"] = str(batch_ex)

job_manager = JobManager()
