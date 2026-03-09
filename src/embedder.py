"""FaceNet embedding wrapper."""
from __future__ import annotations

import numpy as np
from keras_facenet import FaceNet


class FaceEmbedder:
    def __init__(self) -> None:
        self.model = FaceNet()

    def embed_one(self, face_rgb: np.ndarray) -> np.ndarray:
        if face_rgb is None or face_rgb.ndim != 3:
            raise ValueError("face_rgb must be a 3D image array")
        sample = face_rgb.astype("float32")
        sample = np.expand_dims(sample, axis=0)
        embedding = self.model.embeddings(sample)
        return embedding[0]

    def embed_batch(self, faces_rgb: np.ndarray) -> np.ndarray:
        if faces_rgb is None or faces_rgb.ndim != 4:
            raise ValueError("faces_rgb must be a 4D array: (n, h, w, c)")
        samples = faces_rgb.astype("float32")
        return self.model.embeddings(samples)
