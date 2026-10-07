"""
Task 1: Video to Images Web Application (Enterprise Edition).
High-throughput media processing server supporting:
- Single & Multiple Video Upload
- Intelligent Frame Extraction (Case A Native Sampling vs Case B Interpolation)
- Exact Frame Count Guarantee
- FFmpeg minterpolate filter integration
- Disk-streamed large file handling & ZIP creation
- Paginated/Progressive image gallery
"""

import os
import sys
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import modular services
from task1_services.config import (
    UPLOAD_DIR, OUTPUT_DIR, ZIP_DIR, TEMP_DIR,
    ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_IMAGE_COUNT, FFMPEG_BIN
)
from task1_services.validator import inspect_video, VideoValidationError
from task1_services.extractor import calculate_extraction_strategy
from task1_services.job_manager import job_manager

app = Flask(__name__, template_folder="templates", static_folder="templates")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES

# In-memory store for currently staged uploaded videos
uploaded_videos_cache = {}

@app.route("/")
def index():
    return render_template("task1_index.html")

@app.route("/api/upload", methods=["POST"])
def upload_videos():
    """
    Accepts single or multiple video files.
    Streams to temporary disk storage and inspects metadata.
    """
    files = request.files.getlist("videos") or request.files.getlist("video")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No video file provided."}), 400

    uploaded_list = []
    errors = []

    for file in files:
        if not file or not file.filename:
            continue

        raw_filename = file.filename
        safe_name = secure_filename(raw_filename)
        ext = Path(raw_filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{raw_filename}: Unsupported video format.")
            continue

        vid_uuid = str(uuid.uuid4())
        save_filename = f"{vid_uuid}_{safe_name}"
        save_path = UPLOAD_DIR / save_filename

        try:
            # Stream file to disk in chunks to avoid high RAM consumption
            file.save(str(save_path))

            info = inspect_video(save_path)
            default_strategy = calculate_extraction_strategy(info, min(100, info["total_frames"]))

            video_entry = {
                "id": vid_uuid,
                "filename": raw_filename,
                "saved_path": str(save_path),
                "info": info,
                "strategy": default_strategy
            }

            uploaded_videos_cache[vid_uuid] = video_entry
            uploaded_list.append(video_entry)

        except VideoValidationError as ve:
            if save_path.exists():
                save_path.unlink()
            errors.append(f"{raw_filename}: {str(ve)}")
        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            errors.append(f"{raw_filename}: Unexpected error reading video ({str(e)})")

    if not uploaded_list and errors:
        return jsonify({"error": " | ".join(errors)}), 400

    return jsonify({
        "videos": uploaded_list,
        "errors": errors if errors else None,
        "ffmpeg_available": bool(FFMPEG_BIN)
    })

@app.route("/api/calculate-strategy", methods=["POST"])
def calculate_strategy():
    """
    Calculates live extraction strategy based on target count and video metadata.
    """
    data = request.json or {}
    total_frames = int(data.get("total_frames", 300))
    duration = float(data.get("duration_seconds", 10.0))
    native_fps = float(data.get("fps", 30.0))
    requested_count = int(data.get("requested_count", 60))

    if requested_count <= 0:
        return jsonify({"error": "Number of images must be greater than 0."}), 400

    if requested_count > MAX_IMAGE_COUNT:
        return jsonify({"error": f"Image count exceeds maximum system limit of {MAX_IMAGE_COUNT}."}), 400

    dummy_meta = {
        "total_frames": total_frames,
        "duration_seconds": duration,
        "fps": native_fps
    }
    strategy = calculate_extraction_strategy(dummy_meta, requested_count)
    return jsonify(strategy)

@app.route("/api/extract", methods=["POST"])
def start_extraction():
    """
    Initiates batch extraction job for one or multiple uploaded videos.
    """
    data = request.json or {}
    video_ids = data.get("video_ids", [])
    if not video_ids and "video_id" in data:
        video_ids = [data["video_id"]]

    if not video_ids:
        return jsonify({"error": "No videos selected for extraction."}), 400

    # Retrieve cached videos
    matched_videos = []
    for vid in video_ids:
        if vid in uploaded_videos_cache:
            matched_videos.append(uploaded_videos_cache[vid])

    if not matched_videos:
        return jsonify({"error": "Uploaded video session has expired. Please re-upload your video."}), 404

    target_count = int(data.get("target_count", 60))
    if target_count <= 0:
        return jsonify({"error": "Please enter a number greater than 0."}), 400

    config = {
        "target_count": target_count,
        "format": data.get("format", "jpg"),
        "quality": int(data.get("quality", 92)),
        "resize": data.get("resize"),
    }

    batch_id = job_manager.create_batch_job(matched_videos, config)
    job_manager.start_processing(batch_id)

    return jsonify({
        "job_id": batch_id,
        "status": "queued",
        "total_videos": len(matched_videos)
    })

@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Returns overall batch and per-video status and progress."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found or expired."}), 404

    return jsonify({
        "batch_id": job["batch_id"],
        "status": job["status"],
        "total_videos": job["total_videos"],
        "completed_videos": job["completed_videos"],
        "master_zip_url": job["master_zip_url"],
        "videos": [
            {
                "video_id": v["video_id"],
                "filename": v["original_filename"],
                "status": v["status"],
                "progress": v["progress"],
                "current": v["current"],
                "total": v["total"],
                "stage": v["stage"],
                "first_frame_url": v["first_frame_url"],
                "total_images": len(v["images"]),
                "summary": v["summary"],
                "zip_url": v["zip_url"],
                "error": v["error"]
            }
            for v in job["videos"]
        ],
        "error": job["error"]
    })

@app.route("/api/jobs/<job_id>/images", methods=["GET"])
def get_job_images_paginated(job_id):
    """
    Progressively serves images to prevent browser memory exhaustion on large image sets (e.g. 2500+).
    Supports pagination: page, per_page (default 50).
    """
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    video_id = request.args.get("video_id")
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(200, max(10, int(request.args.get("per_page", 50))))

    # Select matching video or first completed
    selected_video = None
    for v in job["videos"]:
        if video_id and v["video_id"] == video_id:
            selected_video = v
            break
        elif not video_id and v["images"]:
            selected_video = v
            break

    if not selected_video or not selected_video["images"]:
        return jsonify({
            "images": [],
            "page": page,
            "total_images": 0,
            "total_pages": 0,
            "has_more": False
        })

    all_images = selected_video["images"]
    total_images = len(all_images)
    total_pages = (total_images + per_page - 1) // per_page

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_items = all_images[start_idx:end_idx]

    return jsonify({
        "video_id": selected_video["video_id"],
        "filename": selected_video["original_filename"],
        "page": page,
        "per_page": per_page,
        "total_images": total_images,
        "total_pages": total_pages,
        "has_more": page < total_pages,
        "summary": selected_video["summary"],
        "images": page_items
    })

@app.route("/api/jobs/<job_id>/image/<path:subpath>")
def serve_job_image(job_id, subpath):
    """Safely serves a single generated frame."""
    target_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(target_dir), subpath)

@app.route("/api/jobs/<job_id>/download-zip")
def download_master_zip(job_id):
    """Downloads master ZIP for multi-video batch or single video."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    if len(job["videos"]) == 1:
        v = job["videos"][0]
        zip_name = f"extracted_{v['clean_name']}_{v['video_id']}.zip"
        return send_from_directory(str(ZIP_DIR), zip_name, as_attachment=True)

    master_zip_name = f"batch_{job_id[:8]}_all_videos.zip"
    return send_from_directory(str(ZIP_DIR), master_zip_name, as_attachment=True)

@app.route("/api/jobs/<job_id>/download-zip/<video_id>")
def download_video_zip(job_id, video_id):
    """Downloads ZIP for an individual video in the batch."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    for v in job["videos"]:
        if v["video_id"] == video_id:
            zip_name = f"extracted_{v['clean_name']}_{video_id}.zip"
            return send_from_directory(str(ZIP_DIR), zip_name, as_attachment=True)

    return jsonify({"error": "Video not found in job."}), 404

if __name__ == "__main__":
    print("=================================================================")
    print("🚀 Task 1: Video -> Images Web Server (Enterprise Edition)")
    print(f"   FFmpeg Engine: {'Enabled (' + str(FFMPEG_BIN) + ')' if FFMPEG_BIN else 'Disabled (Python Fallback)'}")
    print("   Local UI URL:  http://127.0.0.1:5000")
    print("=================================================================")
    app.run(host="127.0.0.1", port=5000, debug=False)
