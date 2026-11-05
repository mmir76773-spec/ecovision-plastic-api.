import base64, os
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
model = YOLO(MODEL_PATH)

app = FastAPI(title="EcoVision AI Plastic Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"message": "EcoVision AI API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image must be 10 MB or smaller.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not read image.")
    result = model.predict(source=image, imgsz=640, conf=CONFIDENCE, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls.item())
            polygon = []
            if result.masks is not None and i < len(result.masks.xy):
                polygon = [[float(x), float(y)] for x, y in result.masks.xy[i].tolist()]
            detections.append({
                "class_name": str(model.names.get(cls_id, cls_id)),
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                "mask_polygon": polygon
            })
    ok, encoded = cv2.imencode(".jpg", result.plot())
    overlay = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode() if ok else None
    return {"success": True, "count": len(detections), "detections": detections,
            "annotated_image": overlay}
