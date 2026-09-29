import os
import csv

# Define project directory structure
DIRECTORIES = [
    "database",
    "known_faces",
    "input_videos"
]

REQUIREMENTS_CONTENT = """opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
torchreid>=1.4.0
ultralytics>=8.0.0
insightface>=0.7.3
onnxruntime>=1.15.0
pillow>=9.5.0
"""

README_CONTENT = """# Aivora Vision AI — Desktop CLI

A unified desktop computer vision system featuring:
1. **Body Re-Identification (Re-ID):** Powered by YOLOv8 (Person Detection) and OSNet (512-D Feature Embeddings).
2. **Masked Face Recognition:** Powered by InsightFace (ArcFace + RetinaFace) with real-time webcam streaming.
3. **Automated Event Logging:** Rate-limited auto-logging to `log.csv` with a built-in interactive log search CLI.

---

## Folder Structure

- `database/`: Stores `.npy` feature embedding files for Body Re-ID profiles.
- `known_faces/`: Place reference photos (`.jpg`, `.png`) here for real-time face recognition.
- `input_videos/`: Store local test video files for Re-ID video processing.
- `app_cli.py`: Main desktop interactive application.
- `log.csv`: Automatically populated detection logs.

---

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt