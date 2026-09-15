import os
import cv2
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import insightface
from insightface.app import FaceAnalysis

app = FastAPI(title="Aivora Vision AI")
templates = Jinja2Templates(directory="templates")

# 1. Initialize InsightFace model
print("Loading InsightFace Model (buffalo_l)...")
insight_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
# Resolution 320x320 for ultra-fast AI detection on CPU
insight_app.prepare(ctx_id=0, det_size=(320, 320))
print("Model loaded successfully!")

KNOWN_FACES_DIR = "known_faces"

# 2. Load Existing Reference Faces
def load_known_faces(known_faces_dir=KNOWN_FACES_DIR):
    embeddings = []
    names = []

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)

    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(known_faces_dir, filename)
            img = cv2.imread(filepath)
            if img is not None:
                faces = insight_app.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
                    names.append(os.path.splitext(filename)[0])
                    print(f"Loaded reference face: {filename}")
    return embeddings, names

known_embeddings, known_names = load_known_faces()

# 3. Cosine Similarity Function
def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# 4. Optimized Live Video Generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Camera resolution setting
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    cached_faces = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Frame Skipping: Run heavy AI detection on alternate frames (2x FPS Boost)
        if frame_count % 2 == 1:
            small_frame = cv2.resize(frame, (320, 240))
            faces = insight_app.get(small_frame)
            
            cached_faces = []
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240

            for face in faces:
                bbox = face.bbox.astype(float)
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)

                current_embedding = face.embedding
                name = "Unknown"
                max_sim = 0.0

                for k_emb, k_name in zip(known_embeddings, known_names):
                    sim = compute_similarity(current_embedding, k_emb)
                    if sim > max_sim:
                        max_sim = sim
                        best_name = k_name

                # Threshold: 0.40 for balanced recognition
                if max_sim > 0.40:
                    name = f"{best_name} ({int(max_sim * 100)}%)"

                cached_faces.append((x1, y1, x2, y2, name))

        # Render Bounding Boxes
        for (x1, y1, x2, y2, name) in cached_faces:
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Encode Frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# 5. FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/upload_face")
async def upload_face(person_name: str = Form(...), file: UploadFile = File(...)):
    global known_embeddings, known_names
    
    try:
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)

        # Validate File Extension
        file_extension = os.path.splitext(file.filename)[1]
        if file_extension.lower() not in ['.jpg', '.jpeg', '.png']:
            return JSONResponse(
                status_code=400, 
                content={"status": "error", "message": "Only JPG, JPEG, and PNG files are allowed!"}
            )

        # Read File Bytes Safely
        contents = await file.read()
        
        # Save File
        save_path = os.path.join(KNOWN_FACES_DIR, f"{person_name}{file_extension}")
        with open(save_path, "wb") as f:
            f.write(contents)

        # Decode directly using OpenCV Matrix
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            if os.path.exists(save_path):
                os.remove(save_path)
            return JSONResponse(
                status_code=400, 
                content={"status": "error", "message": "Corrupted or invalid image file!"}
            )

        # Extract Embedding
        faces = insight_app.get(img)
        if faces:
            known_embeddings.append(faces[0].embedding)
            known_names.append(person_name)
            print(f"✅ Successfully registered: {person_name}")
            return {"status": "success", "message": f"Successfully registered {person_name}!"}
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
            return JSONResponse(
                status_code=400, 
                content={"status": "error", "message": "No face detected in uploaded image!"}
            )

    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": f"Server Error: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)