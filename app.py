import os

# Limit numerical-library threads to reduce memory use on small instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import base64
import gc
import threading

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_IMAGE_SIDE = 1280
INFERENCE_SIZE = 512
MAX_DETECTIONS = 20

# Keep PyTorch CPU usage modest on Render's small free instance.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Load the trained model once when the service starts.
model = YOLO(MODEL_PATH)

# Prevent two image predictions from consuming memory at the same time.
prediction_lock = threading.Lock()

app = FastAPI(title="EcoVision AI Plastic Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "EcoVision AI API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    del data

    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image.")

    height, width = image.shape[:2]
    if height * width > 20_000_000:
        del image
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")

    # Resize unusually large images while preserving their aspect ratio.
    longest_side = max(height, width)
    if longest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(longest_side)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        del image
        image = resized

    try:
        with prediction_lock, torch.inference_mode():
            result = model.predict(
                source=image,
                imgsz=INFERENCE_SIZE,
                conf=CONFIDENCE,
                max_det=MAX_DETECTIONS,
                device="cpu",
                retina_masks=False,
                verbose=False,
            )[0]

            detections = []
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls.item())
                    polygon = []

                    if result.masks is not None and i < len(result.masks.xy):
                        points = np.asarray(result.masks.xy[i], dtype=np.float32)
                        if len(points) >= 3:
                            # Simplify large polygons to reduce response size.
                            contour = points.reshape(-1, 1, 2)
                            simplified = cv2.approxPolyDP(
                                contour, epsilon=2.0, closed=True
                            )
                            polygon = [
                                [float(x), float(y)]
                                for x, y in simplified.reshape(-1, 2)
                            ]

                    detections.append(
                        {
                            "class_name": str(model.names.get(cls_id, cls_id)),
                            "confidence": float(box.conf.item()),
                            "bbox_xyxy": [
                                float(v) for v in box.xyxy[0].tolist()
                            ],
                            "mask_polygon": polygon,
                        }
                    )

            # Keep the same response field used by the existing frontend.
            annotated = result.plot()
            ok, encoded = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if not ok:
                raise HTTPException(
                    status_code=500, detail="Could not encode annotated image."
                )

            overlay = (
                "data:image/jpeg;base64,"
                + base64.b64encode(encoded.tobytes()).decode("ascii")
            )

            response = {
                "success": True,
                "count": len(detections),
                "detections": detections,
                "annotated_image": overlay,
            }

            del result, annotated, encoded
            return response

    finally:
        del image
        gc.collect()
