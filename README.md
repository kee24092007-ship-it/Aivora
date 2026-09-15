<div align="center">

# 🎭 Aivora Vision AI
### Enterprise-Grade Masked Face Recognition & Biometric System

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-ff69b4?style=for-the-badge)](https://github.com/deepinsight/insightface)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<p align="center">
  A high-performance deep-learning web application capable of detecting and recognizing individuals in real-time — even with covered facial features or medical masks. Powered by <b>FastAPI</b>, <b>InsightFace (RetinaFace + ArcFace)</b>, and a modern <b>Glassmorphism Web UI</b>.
</p>

</div>

---

## 🌟 Key Features

* **🎭 Periocular Deep Neural Recognition:** Accurately extracts 512-dimensional facial embeddings using eyes, eyebrows, and upper facial geometry when masks cover lower features.
* **👤 Live Web UI Registration:** Instant web-based enrollment interface to register new faces dynamically without restarting the server.
* **⚡ Ultra-Low Latency Streaming:** High-framerate real-time MJPEG video pipeline driven by optimized OpenCV frame buffering.
* **🧠 InsightFace Engine (`buffalo_l`):** Powered by state-of-the-art RetinaFace detection and ArcFace deep feature identification models.
* **💎 Modern Glassmorphism UI:** Enterprise SaaS interface designed with responsive CSS3 glassmorphism, glowing status accents, and real-time alerts.

---

## 🏗️ Architecture & Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) |
| **Deep Learning Model** | InsightFace (`buffalo_l` - ArcFace & RetinaFace) |
| **Inference Engine** | ONNX Runtime |
| **Computer Vision** | OpenCV (`opencv-python`) |
| **Frontend UI** | HTML5, CSS3 Glassmorphism, JavaScript, Jinja2 Templates |

---

## 📁 Project Directory Structure

```text
face recognition/
│
├── known_faces/          # Reference dataset directory (e.g., John.jpg, Sara.png)
├── templates/
│   └── index.html        # Aivora Glassmorphism Web Interface
├── venv/                 # Isolated Python Virtual Environment
├── main.py               # FastAPI server, REST API endpoints & inference logic
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation