import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import csv
import time
import cv2
import torch
import numpy as np
import requests
import io
import threading
from datetime import datetime
from ultralytics import YOLO
import torchreid
import insightface
from insightface.app import FaceAnalysis

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
KNOWN_FACES_DIR = "known_faces"
DATABASE_DIR = "database"
LOG_FILE = "log.csv"

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN = "8568767043:AAHbiHIt0yTYt1p6bwtuzXa1DgIzPBVJRaw"
TELEGRAM_CHAT_ID = "8829507928"

# Alert Rate-Limiting (seconds between alerts for same target)
ALERT_COOLDOWN = 10 
LAST_ALERT_TIME = {}

# Ensure required directories exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# Initialize Models
print("[INFO] Loading YOLOv8 Person Detector...")
yolo_model = YOLO("yolov8n.pt")

print("[INFO] Loading OSNet Body Re-ID Model...")
reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    loss="softmax",
    pretrained=True
)
reid_model.eval()

print("[INFO] Loading InsightFace Masked Face Recognition Engine...")
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def log_event(source, matched_person, confidence, status, threshold):
    """Logs recognition event to log.csv."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(LOG_FILE)
    
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Source", "Matched Person", "Confidence Score", "Status", "Threshold Used"])
        writer.writerow([timestamp, source, matched_person, f"{confidence:.2f}", status, threshold])


def _send_telegram_request(image_bytes, caption):
    """Worker function to perform network request asynchronously."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}
        files = {"photo": image_bytes}

        response = requests.post(url, data=payload, files=files, timeout=15)
        res_json = response.json()
        if res_json.get("ok"):
            print(f"\n[INFO] 📲 Telegram alert sent successfully!")
        else:
            print(f"\n[ERROR] Telegram API failed: {res_json.get('description')}")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to send Telegram alert: {e}")


# Alert Tracking (Sends alert ONLY ONCE per person per session)
ALERTED_PERSONS = set()

def send_telegram_alert(frame, matched_person, confidence, source_type):
    """Sends alert snapshot to Telegram ONLY ONCE per matched individual."""
    global ALERTED_PERSONS
    
    # Check if alert was already sent for this person in this session
    if matched_person in ALERTED_PERSONS:
        return
            
    # Mark as alerted before sending to prevent duplicate threads
    ALERTED_PERSONS.add(matched_person)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[WARNING] Telegram Bot Token or Chat ID is missing!")
        return

    # Encode frame to JPEG memory buffer
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return

    image_bytes = io.BytesIO(encoded_image.tobytes())
    image_bytes.name = 'alert.jpg'

    caption = (
        f"🚨 *AIVORA VISION AI — ALERT*\n"
        f"👤 *Matched Individual:* {matched_person}\n"
        f"🎯 *Confidence:* {confidence:.2f}%\n"
        f"📍 *Source:* {source_type}\n"
        f"🕒 *Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Dispatch to background thread so camera stream stays smooth
    threading.Thread(target=_send_telegram_request, args=(image_bytes, caption), daemon=True).start()

# =====================================================================
# CORE MODULE FUNCTIONS
# =====================================================================
def register_reid_profile():
    print("\n---------------------------------------------------")
    print("      ACTION: REGISTER BODY RE-ID PROFILE (Video)")
    print("---------------------------------------------------")
    video_path = input("Enter video path (e.g., input_videos/walk1.mp4): ").strip().strip('"')
    person_name = input("Enter Person Name: ").strip()

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    embeddings = []

    print("[INFO] Processing video frames for feature extraction...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # Class 0: Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                    
                crop_resized = cv2.resize(crop, (128, 256))
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                
                with torch.no_grad():
                    feat = reid_model(tensor).numpy().flatten()
                    embeddings.append(feat)

    cap.release()

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        save_path = os.path.join(DATABASE_DIR, f"{person_name}.npy")
        np.save(save_path, avg_embedding)
        print(f"[SUCCESS] Re-ID profile saved successfully at '{save_path}'!")
    else:
        print("[ERROR] No person detected in the provided video.")


def recognize_reid_video():
    print("\n---------------------------------------------------")
    print("      ACTION: RECOGNIZE PERSON RE-ID (Video)")
    print("---------------------------------------------------")
    video_path = input("Enter test video path: ").strip().strip('"')

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    db_profiles = {}
    for file in os.listdir(DATABASE_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            db_profiles[name] = np.load(os.path.join(DATABASE_DIR, file))

    if not db_profiles:
        print("[WARNING] No registered Re-ID profiles found in database/ folder!")
        return

    cap = cv2.VideoCapture(video_path)
    print("[INFO] Processing video... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue

                crop_resized = cv2.resize(crop, (128, 256))
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

                with torch.no_grad():
                    feat = reid_model(tensor).numpy().flatten()

                best_match = "Unknown"
                best_score = 0.0

                for name, db_feat in db_profiles.items():
                    sim = np.dot(feat, db_feat) / (np.linalg.norm(feat) * np.linalg.norm(db_feat))
                    if sim > best_score:
                        best_score = sim
                        best_match = name

                label = f"{best_match} ({best_score*100:.1f}%)" if best_score > 0.6 else "Unknown"
                color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if best_match != "Unknown" and best_score > 0.6:
                    log_event("Re-ID Video", best_match, best_score * 100, "MATCH", "Cosine > 0.6")
                    send_telegram_alert(frame, best_match, best_score * 100, "Body Re-ID Video")

        cv2.imshow("Aivora Re-ID Matching", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_masked_face_stream():
    print("\n---------------------------------------------------")
    print("      ACTION: MASKED FACE REAL-TIME STREAM")
    print("---------------------------------------------------")
    
    known_embeddings = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            person_name = os.path.splitext(filename)[0]
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            img = cv2.imread(img_path)
            
            faces = face_app.get(img)
            if faces:
                known_embeddings[person_name] = faces[0].embedding

    if not known_embeddings:
        print("[WARNING] No known face photos found in known_faces/ directory!")
        return

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting Webcam Stream... Press 'q' to return to CLI menu.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding

            best_match = "Unknown"
            best_score = 0.0

            for name, known_emb in known_embeddings.items():
                sim = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
                if sim > best_score:
                    best_score = sim
                    best_match = name

            if best_score > 0.4:
                label = f"{best_match} ({best_score*100:.1f}%)"
                color = (0, 255, 0)
                log_event("Webcam Stream", best_match, best_score * 100, "MATCH", "Cosine > 0.4")
                send_telegram_alert(frame, best_match, best_score * 100, "Real-Time Masked Face Stream")
            else:
                label = "Unknown Face"
                color = (0, 0, 255)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Aivora Vision AI — Masked Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def list_profiles():
    print("\n---------------------------------------------------")
    print("      REGISTERED PROFILES SUMMARY")
    print("---------------------------------------------------")
    
    print("\n[Body Re-ID Vector Profiles (.npy)]")
    reid_files = [f for f in os.listdir(DATABASE_DIR) if f.endswith('.npy')]
    if reid_files:
        for f in reid_files:
            print(f" - {os.path.splitext(f)[0]}")
    else:
        print(" (No Re-ID profiles registered yet)")

    print("\n[Known Face Reference Images (.jpg/.png)]")
    face_files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if face_files:
        for f in face_files:
            print(f" - {os.path.splitext(f)[0]}")
    else:
        print(" (No face images found in known_faces/)")


def view_logs():
    print("\n---------------------------------------------------")
    print("      DETECTION & RECOGNITION LOGS (log.csv)")
    print("---------------------------------------------------")
    if not os.path.exists(LOG_FILE):
        print("[INFO] No log records found.")
        return

    with open(LOG_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            print(" | ".join(row))


# =====================================================================
# MAIN MENU LOOP
# =====================================================================
def main():
    while True:
        print("\n===================================================")
        print("        AIVORA VISION AI — SYSTEM DESKTOP CLI      ")
        print("===================================================")
        print(" [1] Register Person Re-ID (Video -> YOLOv8 + OSNet)")
        print(" [2] Recognize Person Re-ID from Video File")
        print(" [3] Start Real-Time Masked Face Camera Stream (InsightFace)")
        print(" [4] List All Registered Profiles (Re-ID & Faces)")
        print(" [5] View & Search Recognition Logs (log.csv)")
        print(" [6] Exit")
        print("===================================================")

        choice = input("Select option [1-6]: ").strip()

        if choice == "1":
            register_reid_profile()
        elif choice == "2":
            recognize_reid_video()
        elif choice == "3":
            run_masked_face_stream()
        elif choice == "4":
            list_profiles()
        elif choice == "5":
            view_logs()
        elif choice == "6":
            print("\n[INFO] Shutting down Aivora Vision AI engine...")
            sys.exit(0)
        else:
            print("[ERROR] Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Program stopped cleanly by user. Exiting...")
        sys.exit(0)