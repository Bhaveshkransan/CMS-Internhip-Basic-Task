import os
import sys
import uuid
import time
import zipfile
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='templates')
CORS(app)

BASE_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = BASE_DIR / 'task1_storage'
UPLOAD_DIR = STORAGE_DIR / 'uploads'
OUTPUT_DIR = STORAGE_DIR / 'outputs'
ZIP_DIR = STORAGE_DIR / 'zips'

for d in [STORAGE_DIR, UPLOAD_DIR, OUTPUT_DIR, ZIP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.ts'}

jobs = {}

def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return {
        'fps': round(fps, 2),
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_seconds': round(duration, 2),
        'duration_formatted': time.strftime('%H:%M:%S', time.gmtime(duration))
    }

def process_video_job(job_id: str, video_path: Path, config: dict):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = 'Could not open video file.'
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        
        mode = config.get('mode', 'count')
        resize_dim = config.get('resize')
        img_fmt = config.get('format', 'jpg').lower()
        quality = int(config.get('quality', 92))

        target_indices = set()
        if mode == 'count':
            count = max(1, int(config.get('target_count', 30)))
            if total_frames > 0:
                count = min(count, total_frames)
                target_indices = set(np.linspace(0, total_frames - 1, count, dtype=int))
            else:
                target_indices = set()
        elif mode == 'interval':
            interval = max(0.01, float(config.get('interval_seconds', 1.0)))
            step = max(1, int(round(fps * interval)))
            target_indices = set(range(0, max(total_frames, 1), step))
        elif mode == 'fps':
            target_fps = max(0.1, float(config.get('target_fps', 1.0)))
            step = max(1, int(round(fps / target_fps)))
            target_indices = set(range(0, max(total_frames, 1), step))
        elif mode == 'stride':
            stride = max(1, int(config.get('stride', 1)))
            target_indices = set(range(0, max(total_frames, 1), stride))

        total_to_extract = len(target_indices) if target_indices else (total_frames or 1)
        jobs[job_id]['total'] = total_to_extract
        jobs[job_id]['status'] = 'processing'

        job_out_dir = OUTPUT_DIR / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)

        extracted_count = 0
        frame_idx = 0
        extracted_files = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not target_indices or frame_idx in target_indices:
                if resize_dim and resize_dim[0] > 0 and resize_dim[1] > 0:
                    frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)

                file_name = f'frame_{extracted_count + 1:05d}_{frame_idx:06d}.{img_fmt}'
                out_path = job_out_dir / file_name

                if img_fmt in ['jpg', 'jpeg']:
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                elif img_fmt == 'png':
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                elif img_fmt == 'webp':
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
                else:
                    cv2.imwrite(str(out_path), frame)

                extracted_count += 1
                img_url = f'/api/image/{job_id}/{file_name}'
                
                if extracted_count == 1:
                    jobs[job_id]['first_frame_url'] = img_url

                extracted_files.append({
                    'filename': file_name,
                    'url': img_url,
                    'frame_idx': frame_idx,
                    'timestamp': round(frame_idx / fps, 2) if fps > 0 else 0
                })

                jobs[job_id]['current'] = extracted_count
                jobs[job_id]['progress'] = min(100.0, round((extracted_count / total_to_extract) * 100, 1))

            frame_idx += 1

        cap.release()

        zip_filename = f'extracted_frames_{job_id[:8]}.zip'
        zip_path = ZIP_DIR / zip_filename
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for item in extracted_files:
                fpath = job_out_dir / item['filename']
                if fpath.exists():
                    zf.write(fpath, arcname=item['filename'])

        jobs[job_id]['zip_url'] = f'/api/download-zip/{zip_filename}'
        jobs[job_id]['images'] = extracted_files
        jobs[job_id]['total'] = extracted_count
        jobs[job_id]['current'] = extracted_count
        jobs[job_id]['progress'] = 100.0
        jobs[job_id]['status'] = 'completed'

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('task1_index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported video format: {ext}'}), 400

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f'{job_id}{ext}'
    file.save(str(save_path))

    info = get_video_info(save_path)
    if not info:
        return jsonify({'error': 'Could not read video metadata.'}), 400

    jobs[job_id] = {
        'status': 'uploaded',
        'video_path': str(save_path),
        'filename': file.filename,
        'info': info,
        'progress': 0.0,
        'current': 0,
        'total': 0,
        'first_frame_url': None,
        'images': [],
        'zip_url': None,
        'error': None
    }

    return jsonify({
        'job_id': job_id,
        'filename': file.filename,
        'info': info
    })

@app.route('/api/extract', methods=['POST'])
def start_extraction():
    data = request.json or {}
    job_id = data.get('job_id')
    if not job_id or job_id not in jobs:
        return jsonify({'error': 'Invalid or expired job ID'}), 404

    job = jobs[job_id]
    if job['status'] == 'processing':
        return jsonify({'error': 'Job already in progress'}), 400

    config = {
        'mode': data.get('mode', 'count'),
        'target_count': int(data.get('target_count', 60)),
        'interval_seconds': float(data.get('interval_seconds', 1.0)),
        'target_fps': float(data.get('target_fps', 1.0)),
        'stride': int(data.get('stride', 5)),
        'format': data.get('format', 'jpg'),
        'quality': int(data.get('quality', 92)),
        'resize': tuple(data['resize']) if data.get('resize') else None
    }

    job['status'] = 'queued'
    job['progress'] = 0.0
    job['images'] = []
    job['first_frame_url'] = None

    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, Path(job['video_path']), config),
        daemon=True
    )
    thread.start()

    return jsonify({'message': 'Extraction started', 'job_id': job_id})

@app.route('/api/progress/<job_id>')
def get_progress(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'current': job['current'],
        'total': job['total'],
        'first_frame_url': job['first_frame_url'],
        'total_images': len(job['images']),
        'zip_url': job['zip_url'],
        'error': job['error']
    })

@app.route('/api/images/<job_id>')
def get_images(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'images': job['images'],
        'zip_url': job['zip_url']
    })

@app.route('/api/image/<job_id>/<filename>')
def serve_image(job_id, filename):
    img_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(img_dir), filename)

@app.route('/api/download-zip/<zip_filename>')
def download_zip(zip_filename):
    return send_from_directory(str(ZIP_DIR), zip_filename, as_attachment=True)

if __name__ == '__main__':
    print('Starting Task 1 Video to Images Server at http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=False)
