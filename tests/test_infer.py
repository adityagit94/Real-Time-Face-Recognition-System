import numpy as np

from src.config import Config
from src import infer


class DummyEmbedder:
    def embed_one(self, face_rgb):
        return np.ones(512, dtype=np.float32)


class DummyModel:
    def predict_proba(self, emb):
        return np.array([[0.25, 0.75]], dtype=np.float32)


class DummyLowConfModel:
    def predict_proba(self, emb):
        return np.array([[0.49, 0.51]], dtype=np.float32)


class DummyEncoder:
    def inverse_transform(self, idxs):
        labels = np.array(["alice", "bob"])
        return labels[idxs]


def _patched_recognizer(monkeypatch, model):
    monkeypatch.setattr(infer.joblib, "load", lambda path: model if "svm" in str(path) else DummyEncoder())
    monkeypatch.setattr(infer, "FaceEmbedder", lambda: DummyEmbedder())
    cfg = Config(confidence_threshold=0.6)
    return infer.FaceRecognizer(cfg)


def test_predict_face_returns_class_name(monkeypatch):
    recognizer = _patched_recognizer(monkeypatch, DummyModel())
    label, confidence = recognizer.predict_face(np.zeros((160, 160, 3), dtype=np.uint8))
    assert label == "bob"
    assert confidence > 0.6


def test_predict_face_returns_unknown_below_threshold(monkeypatch):
    recognizer = _patched_recognizer(monkeypatch, DummyLowConfModel())
    label, confidence = recognizer.predict_face(np.zeros((160, 160, 3), dtype=np.uint8))
    assert label == "Unknown"
    assert confidence < 0.6
