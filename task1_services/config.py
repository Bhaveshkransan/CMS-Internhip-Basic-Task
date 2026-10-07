"""
Configuration Module for Task 1: Video to Images Suite.
Supports local PC and server-ready environments via environment variables.
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
STORAGE_DIR = BASE_DIR / 'task1_storage'

# Storage Directories
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', str(STORAGE_DIR / 'uploads'))).resolve()
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(STORAGE_DIR / 'outputs'))).resolve()
ZIP_DIR = Path(os.getenv('ZIP_DIR', str(STORAGE_DIR / 'zips'))).resolve()
TEMP_DIR = Path(os.getenv('TEMP_DIR', str(STORAGE_DIR / 'temp'))).resolve()

for directory in [UPLOAD_DIR, OUTPUT_DIR, ZIP_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Processing Limits & Defaults
MAX_FILE_SIZE_BYTES = int(os.getenv('MAX_FILE_SIZE_MB', '2048')) * 1024 * 1024  # 2 GB default
MAX_IMAGE_COUNT = int(os.getenv('MAX_IMAGE_COUNT', '50000'))
DEFAULT_IMAGE_QUALITY = int(os.getenv('DEFAULT_IMAGE_QUALITY', '92'))

ALLOWED_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv',
    '.flv', '.webm', '.m4v', '.ts', '.mpeg', '.mpg'
}

# Auto-detect FFmpeg
def get_ffmpeg_path() -> str | None:
    env_path = os.getenv('FFMPEG_PATH')
    if env_path and Path(env_path).is_file():
        return env_path

    which_path = shutil.which('ffmpeg')
    if which_path:
        return which_path

    candidate_locations = [
        Path(r'C:\Users\Intern\AppData\Local\Programs\Git\cmd\ffmpeg.exe'),
        Path(r'C:\Program Files\Git\cmd\ffmpeg.exe'),
    ]
    for candidate in candidate_locations:
        if candidate.is_file():
            return str(candidate)

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass

    return None

FFMPEG_BIN = get_ffmpeg_path()
