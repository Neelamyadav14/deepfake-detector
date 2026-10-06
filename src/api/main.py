"""
Serves predictions two ways:
- /predict       : upload a short video clip (Stage 2: CNN+LSTM)
- /predict_image : upload a single image (Stage 1: CNN only)
"""
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep.extract_frames import extract_faces_from_video, FACE_CASCADE, FACE_SIZE

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = DATA_DIR / "models"

app = FastAPI(title="Deepfake Detector API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cnn_model = None
_lstm_model = None
_feature_extractor = None


@app.on_event("startup")
def load_models():
    global _cnn_model, _lstm_model, _feature_extractor
    import tensorflow as tf
    from src.models.cnn_baseline import build_feature_extractor

    cnn_path = MODELS_DIR / "cnn_baseline.keras"
    lstm_path = MODELS_DIR / "lstm_temporal.keras"

    if cnn_path.exists():
        _cnn_model = tf.keras.models.load_model(cnn_path)
        _feature_extractor = build_feature_extractor(_cnn_model)
    if lstm_path.exists():
        _lstm_model = tf.keras.models.load_model(lstm_path)


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """Tests a single image through the Stage 1 CNN baseline."""
    if _cnn_model is None:
        return {"error": "No trained CNN model found. Run the training pipeline first."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not read this image file."}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(detected) == 0:
        face_resized = cv2.resize(frame, FACE_SIZE)
    else:
        x, y, w, h = max(detected, key=lambda b: b[2] * b[3])
        face_crop = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, FACE_SIZE)

    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    input_batch = np.expand_dims(face_rgb.astype("float32"), axis=0)

    prob = float(_cnn_model.predict(input_batch, verbose=0)[0, 0])

    return {
        "prediction": "fake" if prob > 0.5 else "real",
        "confidence": prob if prob > 0.5 else 1 - prob,
        "raw_fake_probability": prob,
        "face_detected": len(detected) > 0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Tests a video through Stage 2 (CNN+LSTM) -- requires the LSTM to be trained."""
    if _cnn_model is None:
        return {"error": "No trained model found. Run the training pipeline first (see README)."}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    faces = extract_faces_from_video(Path(tmp_path))
    if not faces:
        return {"error": "No face detected in this video."}

    # Convert BGR (OpenCV's default) to RGB -- the training pipeline
    # converts to RGB before feeding the model, so inference must match
    # exactly or the model sees color-inverted faces and predictions
    # become unreliable.
    faces_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces]
    faces_array = np.array(faces_rgb).astype("float32")

    if _lstm_model is not None:
        features = _feature_extractor.predict(faces_array, verbose=0)
        seq_len = _lstm_model.input_shape[1]
        if len(features) < seq_len:
            pad = np.zeros((seq_len - len(features), features.shape[1]))
            features = np.vstack([features, pad])
        else:
            features = features[:seq_len]
        prob = float(_lstm_model.predict(features[np.newaxis, ...], verbose=0)[0, 0])
        model_used = "cnn+lstm"
    else:
        frame_probs = _cnn_model.predict(faces_array, verbose=0).flatten()
        prob = float(np.mean(frame_probs))
        model_used = "cnn_only"

    return {
        "prediction": "fake" if prob > 0.5 else "real",
        "confidence": prob if prob > 0.5 else 1 - prob,
        "raw_fake_probability": prob,
        "model_used": model_used,
        "num_frames_analyzed": len(faces),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cnn_loaded": _cnn_model is not None,
        "lstm_loaded": _lstm_model is not None,
    }