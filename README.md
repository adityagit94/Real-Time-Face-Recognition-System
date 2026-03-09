# Real-Time Face Recognition System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Project-Portfolio-green)

I built this project by turning a Colab prototype into a clean local pipeline for multi-class face recognition using **MTCNN** (detection), **FaceNet** (embeddings), and an **SVM** classifier.

## Overview
- Loads face images from class-based folders
- Detects and crops faces with MTCNN (safe bounding-box handling)
- Resizes faces to 160x160 and extracts FaceNet embeddings
- Trains an RBF-kernel SVM classifier
- Saves model + label encoder with joblib
- Evaluates with confusion matrix and classification metrics
- Supports single-image and real-time webcam inference
- Includes configurable **Unknown** label by confidence threshold

## Architecture / Pipeline
1. Dataset loading (`src/data_loader.py`)
2. Face detection/crop (`src/detector.py`)
3. Embedding extraction (`src/embedder.py`)
4. Label encoding + train/test split + SVM training (`src/trainer.py`)
5. Metrics and confusion matrix export (`src/evaluator.py`)
6. Inference utilities (`src/infer.py`)

## Folder Structure
```text
face-recognition-system/
├── app/
├── src/
├── models/
├── reports/
├── data/
├── notebooks/
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format
```text
data/raw/
   person_1/
      img1.jpg
      img2.jpg
   person_2/
      img1.jpg
      img2.jpg
```

## Train
```bash
python main.py train
```
Artifacts:
- `models/svm_model.pkl`
- `models/label_encoder.pkl`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

## Infer on Image
```bash
python main.py infer-image --image path/to/image.jpg --save reports/inference_output.jpg
```

## Webcam Demo
```bash
python main.py webcam --camera 0 --frame-skip 2
```
Press `q` to exit.

## Evaluation
The training pipeline exports:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Classification report
- Confusion matrix plot

## Resume-Ready Highlights
- Built modular, reproducible face-recognition pipeline from notebook prototype
- Implemented robust face crop handling (negative bbox + no-face cases)
- Added persistence, evaluation artifacts, and local inference scripts
- Added basic tests for detector, embedder, and training smoke path

## Future Improvements
- Add face alignment before embedding
- Add embedding cache for faster retraining
- Add tracking for stable webcam labels
- Add threshold tuning script for unknown detection


## What I changed from the notebook
- Removed Colab-only steps (Drive mount + file downloads)
- Split monolithic cells into reusable Python modules
- Added structured outputs (`models/` + `reports/`)
- Added confidence-threshold `Unknown` handling in inference
- Added lightweight tests for core pipeline behavior
