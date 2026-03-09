from __future__ import annotations

import argparse
import subprocess
import sys

from src.trainer import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Face Recognition System")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train SVM model from dataset")

    infer_parser = sub.add_parser("infer-image", help="Run inference on one image")
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--save", default="reports/inference_output.jpg")

    webcam_parser = sub.add_parser("webcam", help="Run webcam recognition")
    webcam_parser.add_argument("--camera", type=int, default=0)
    webcam_parser.add_argument("--frame-skip", type=int, default=2)

    args = parser.parse_args()

    if args.command == "train":
        metrics = train_pipeline()
        print(metrics)
    elif args.command == "infer-image":
        cmd = [sys.executable, "-m", "app.image_infer", "--image", args.image, "--save", args.save]
        subprocess.run(cmd, check=True)
    elif args.command == "webcam":
        cmd = [sys.executable, "-m", "app.webcam_app", "--camera", str(args.camera), "--frame-skip", str(args.frame_skip)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
