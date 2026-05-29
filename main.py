import io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import tensorflow as tf
import os

# ─── Model loading ────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "output/gender_classification_model.keras")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Train the model first (python train.py) or set MODEL_PATH env var."
        )
    print(f"[startup] Loading model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[startup] Model ready ✓")
    yield
    model = None

app = FastAPI(title="Gender Classifier", lifespan=lifespan)

# ─── Preprocessing (must match training in data_loader.py) ───────────────────
IMG_SIZE = 128

def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode, resize, and batch an image exactly like data_loader.py does."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)          # BGR, same as cv2.imread
    if img is None:
        raise ValueError("Could not decode image — unsupported format.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))        # (128, 128, 3)
    img = img.astype(np.float32)                       # keep 0-255 range (no /255)
    return np.expand_dims(img, axis=0)                 # (1, 128, 128, 3)

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG, PNG, WebP or BMP."
        )

    raw = await file.read()
    try:
        tensor = preprocess(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    prob: float = float(model.predict(tensor, verbose=0)[0][0])  # sigmoid → P(Male)
    label = "Male" if prob >= 0.5 else "Female"
    confidence = prob if label == "Male" else 1.0 - prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),   # as percentage
        "raw_probability": round(prob, 6),           # raw sigmoid output
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ─── Serve static UI ──────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)