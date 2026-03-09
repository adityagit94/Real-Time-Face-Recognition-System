import numpy as np

from src.detector import FaceDetector


def test_clip_box_handles_negative_coordinates():
    box = FaceDetector._clip_box(-10, -5, 40, 30, (100, 100, 3))
    assert box == (0, 0, 30, 25)


def test_extract_primary_face_returns_none_when_no_detection(monkeypatch):
    detector = FaceDetector()
    monkeypatch.setattr(detector, "detect_faces", lambda _: [])
    result = detector.extract_primary_face(np.zeros((160, 160, 3), dtype=np.uint8))
    assert result.face is None
    assert result.box is None
