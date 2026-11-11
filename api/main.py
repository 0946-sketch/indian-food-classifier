import io
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE, "model", "model.onnx"))
CLASSES_PATH = os.getenv("CLASSES_PATH", os.path.join(BASE, "model", "classes.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1-mobilenetv3-large")
MAX_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not on every request
    state["session"] = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    state["input"] = state["session"].get_inputs()[0].name
    with open(CLASSES_PATH) as f:
        state["classes"] = json.load(f)
    yield
    state.clear()


app = FastAPI(title="Image Classifier API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(img: Image.Image) -> np.ndarray:
    # Mirrors training eval transform: resize(256) -> center crop(224) -> normalize
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "num_classes": len(state["classes"])}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported file type. Please upload a JPG, PNG or WEBP image.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large. Maximum size is 5 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The file is not a valid image.")

    logits = state["session"].run(None, {state["input"]: preprocess(img)})[0][0]
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    top = np.argsort(p)[::-1][:3]
    return {
        "predictions": [
            {"label": state["classes"][i], "confidence": round(float(p[i]), 4)} for i in top
        ],
        "model_version": MODEL_VERSION,
    }
