"""Evaluation helpers for classifier performance."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report,
                             precision_recall_fscore_support)


def evaluate_and_save(y_true, y_pred, class_names, metrics_path: str, confusion_matrix_path: str) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": report,
    }

    mpath = Path(metrics_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with mpath.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cpath = Path(confusion_matrix_path)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(cpath)
    plt.close(fig)

    return metrics
