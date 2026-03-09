"""Dataset loading from class subfolders."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.detector import FaceDetector

LOGGER = logging.getLogger(__name__)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset_faces(dataset_dir: str | Path, detector: FaceDetector) -> tuple[np.ndarray, np.ndarray]:
    dataset_dir = Path(dataset_dir)
    faces: list[np.ndarray] = []
    labels: list[str] = []

    for person_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        loaded = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                LOGGER.warning("Skipping unreadable image: %s", img_path)
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = detector.extract_primary_face(image_rgb)
            if result.face is None:
                LOGGER.info("No face found in %s", img_path)
                continue
            faces.append(result.face)
            labels.append(person_dir.name)
            loaded += 1
        LOGGER.info("Loaded %d faces for class '%s'", loaded, person_dir.name)

    return np.asarray(faces), np.asarray(labels)
