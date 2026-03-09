from __future__ import annotations

import argparse

import cv2

from app.utils import draw_prediction
from src.config import DEFAULT_CONFIG
from src.infer import FaceRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run face recognition on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--save", default="reports/inference_output.jpg", help="Optional output image path")
    args = parser.parse_args()

    recognizer = FaceRecognizer(DEFAULT_CONFIG)
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    preds = recognizer.predict_faces_in_image(image_rgb)
    for pred in preds:
        draw_prediction(image_bgr, pred["box"], pred["label"], pred["confidence"])
        print(f"Detected: {pred['label']} ({pred['confidence']:.3f}) @ {pred['box']}")

    cv2.imwrite(args.save, image_bgr)
    print(f"Saved result to {args.save}")


if __name__ == "__main__":
    main()
