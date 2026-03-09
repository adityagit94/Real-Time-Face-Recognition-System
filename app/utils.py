"""Utility helpers for app scripts."""
from __future__ import annotations

import cv2


def draw_prediction(frame_bgr, box, label: str, confidence: float) -> None:
    x, y, w, h = box
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame_bgr, text, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
