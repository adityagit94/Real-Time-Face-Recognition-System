from __future__ import annotations

import argparse

import cv2

from app.utils import draw_prediction
from src.config import DEFAULT_CONFIG
from src.infer import FaceRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time webcam face recognition")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=2, help="Run recognition every N frames")
    args = parser.parse_args()

    recognizer = FaceRecognizer(DEFAULT_CONFIG)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cached_preds = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % max(1, args.frame_skip) == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cached_preds = recognizer.predict_faces_in_image(frame_rgb)

        for pred in cached_preds:
            draw_prediction(frame_bgr, pred["box"], pred["label"], pred["confidence"])

        cv2.imshow("Real-Time Face Recognition", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
