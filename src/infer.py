"""Inference utilities for image and frame-level prediction."""
from __future__ import annotations

import joblib
import numpy as np

from src.config import Config, DEFAULT_CONFIG
from src.detector import FaceDetector
from src.embedder import FaceEmbedder


class FaceRecognizer:
    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.detector = FaceDetector(target_size=config.image_size, margin=config.crop_margin)
        self.embedder = FaceEmbedder()
        self.model = joblib.load(config.svm_model_path)
        self.encoder = joblib.load(config.label_encoder_path)

    def predict_face(self, face_rgb: np.ndarray) -> tuple[str, float]:
        emb = self.embedder.embed_one(face_rgb).reshape(1, -1)
        probs = self.model.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        name = str(self.encoder.inverse_transform([idx])[0])
        if conf < self.config.confidence_threshold:
            return "Unknown", conf
        return name, conf

    def predict_faces_in_image(self, image_rgb: np.ndarray) -> list[dict]:
        outputs = []
        for det in self.detector.extract_all_faces(image_rgb):
            if det.face is None or det.box is None:
                continue
            label, conf = self.predict_face(det.face)
            outputs.append({"box": det.box, "label": label, "confidence": conf})
        return outputs
