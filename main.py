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

# Initialize InsightFace model
print("Loading InsightFace Model (buffalo_l)...")
insight_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
insight_app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded successfully!")

KNOWN_FACES_DIR = "known_faces"

# Load known reference faces
def load_known_faces(known_faces_dir=KNOWN_FACES_DIR):
    embeddings = []
    names = []

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Folder '{known_faces_dir}' created.")

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
                else:
                    print(f"Warning: No face found in {filename}")
    return embeddings, names

known_embeddings, known_names = load_known_faces()

def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Webcam could not be accessed.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = insight_app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            current_embedding = face.embedding
            name = "Unknown"
            max_sim = 0.0

            for k_emb, k_name in zip(known_embeddings, known_names):
                sim = compute_similarity(current_embedding, k_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_name = k_name

            if max_sim > 0.40:
                name = f"{best_name} ({int(max_sim * 100)}%)"

            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# New API Endpoint: Web UI moolama new face register panna
@app.post("/upload_face")
async def upload_face(person_name: str = Form(...), file: UploadFile = File(...)):
    global known_embeddings, known_names
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    file_extension = os.path.splitext(file.filename)[1]
    if file_extension.lower() not in ['.jpg', '.jpeg', '.png']:
        return JSONResponse(status_code=400, content={"message": "Only JPG, JPEG, and PNG files are allowed!"})

    # File-ai save pannudhu
    save_path = os.path.join(KNOWN_FACES_DIR, f"{person_name}{file_extension}")
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Face embedding reload pannudhu
    img = cv2.imread(save_path)
    if img is not None:
        faces = insight_app.get(img)
        if faces:
            # Memory-la direct-ah store pannu
            known_embeddings.append(faces[0].embedding)
            known_names.append(person_name)
            return {"status": "success", "message": f"Successfully registered {person_name}!"}
        else:
            os.remove(save_path) # Face illana file-ai delete pannidum
            return JSONResponse(status_code=400, content={"message": "No clear face detected in the image. Try another photo."})

    return JSONResponse(status_code=500, content={"message": "Failed to process image."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)