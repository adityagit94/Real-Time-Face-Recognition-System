"""Face detection and crop utilities using MTCNN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class DetectionResult:
    face: np.ndarray | None
    box: tuple[int, int, int, int] | None
    confidence: float | None
    keypoints: dict[str, tuple[int, int]] | None


class FaceDetector:
    def __init__(self, target_size: tuple[int, int] = (160, 160), margin: float = 0.0) -> None:
        self.target_size = target_size
        self.margin = margin
        self.detector = MTCNN()

    @staticmethod
    def _clip_box(x: int, y: int, w: int, h: int, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        img_h, img_w = image_shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        return x1, y1, x2, y2

    def _add_margin(self, box: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        x, y, w, h = box
        mx, my = int(w * self.margin), int(h * self.margin)
        return self._clip_box(x - mx, y - my, w + 2 * mx, h + 2 * my, image_shape)

    def detect_faces(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        if image_rgb is None or image_rgb.size == 0:
            return []
        return self.detector.detect_faces(image_rgb)

    def extract_primary_face(self, image_rgb: np.ndarray) -> DetectionResult:
        detections = self.detect_faces(image_rgb)
        if not detections:
            return DetectionResult(None, None, None, None)

        best = max(detections, key=lambda d: d.get("confidence", 0.0))
        x, y, w, h = best["box"]
        x1, y1, x2, y2 = self._add_margin((x, y, w, h), image_rgb.shape)
        if x2 <= x1 or y2 <= y1:
            return DetectionResult(None, None, best.get("confidence"), best.get("keypoints"))

        face = image_rgb[y1:y2, x1:x2]
        face = cv2.resize(face, self.target_size)
        return DetectionResult(face, (x1, y1, x2 - x1, y2 - y1), best.get("confidence"), best.get("keypoints"))

    def extract_all_faces(self, image_rgb: np.ndarray) -> list[DetectionResult]:
        outputs: list[DetectionResult] = []
        for d in self.detect_faces(image_rgb):
            x, y, w, h = d["box"]
            x1, y1, x2, y2 = self._add_margin((x, y, w, h), image_rgb.shape)
            if x2 <= x1 or y2 <= y1:
                continue
            face = cv2.resize(image_rgb[y1:y2, x1:x2], self.target_size)
            outputs.append(DetectionResult(face, (x1, y1, x2 - x1, y2 - y1), d.get("confidence"), d.get("keypoints")))
        return outputs
