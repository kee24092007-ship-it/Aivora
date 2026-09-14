# 🎭 Real-Time Masked Face Recognition Web Application

A high-performance, deep-learning powered web application capable of detecting and recognizing individuals in real-time — even when they are wearing face masks. Built with **FastAPI**, **InsightFace (ArcFace + RetinaFace)**, **OpenCV**, and **Jinja2**.

---

## 🌟 Key Features

* **🎭 Masked Face Identification:** Recognizes facial features using deep periocular embeddings (eyes, eyebrows, forehead) when lower facial features are covered by masks.
* **⚡ Ultra-Low Latency Streaming:** High-framerate real-time MJPEG video streaming via OpenCV frame buffering.
* **🧠 InsightFace Model Integration:** Powered by `buffalo_l` (RetinaFace for detection and ArcFace for 512-dimensional embedding extractions).
* **🎨 Modern Responsive UI:** Built with HTML5/CSS3 and Jinja2 templating, optimized for seamless viewing across devices.
* **📂 Automated Face Database:** Automatically scans and encodes new reference images directly from the `known_faces/` directory.

---

## 💻 Tech Stack & Dependencies

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI |
| **ASGI Server** | Uvicorn |
| **Deep Learning Engine** | InsightFace (`buffalo_l` model) |
| **Inference Runtime** | ONNX Runtime |
| **Computer Vision** | OpenCV (`opencv-python`) |
| **Frontend UI** | HTML5, CSS3, Jinja2 Templates |

---

## 📁 Project Directory Structure

```text
face recognition/
│
├── known_faces/          # Store reference photos here (e.g., John.jpg, Sara.png)
├── templates/
│   └── index.html        # HTML Frontend interface
├── venv/                 # Isolated Python Virtual Environment
├── main.py               # FastAPI backend & inference engine
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation