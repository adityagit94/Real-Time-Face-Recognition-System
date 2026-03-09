"""Model training pipeline."""
from __future__ import annotations

import logging

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.config import Config, DEFAULT_CONFIG
from src.data_loader import load_dataset_faces
from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.evaluator import evaluate_and_save
from src.paths import ensure_runtime_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def train_pipeline(config: Config = DEFAULT_CONFIG) -> dict:
    ensure_runtime_dirs()
    detector = FaceDetector(target_size=config.image_size, margin=config.crop_margin)
    faces, labels = load_dataset_faces(config.dataset_dir, detector)
    if len(faces) == 0:
        raise RuntimeError(f"No faces were loaded from dataset directory: {config.dataset_dir}")

    embedder = FaceEmbedder()
    embeddings = embedder.embed_batch(faces)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y if len(set(y)) > 1 else None,
        shuffle=True,
    )

    model = SVC(kernel="rbf", probability=True)
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    metrics = evaluate_and_save(
        y_test,
        y_pred_test,
        class_names=list(encoder.classes_),
        metrics_path=config.metrics_path,
        confusion_matrix_path=config.confusion_matrix_path,
    )
    metrics["train_accuracy"] = float((y_pred_train == y_train).mean())

    joblib.dump(model, config.svm_model_path)
    joblib.dump(encoder, config.label_encoder_path)

    LOGGER.info("Training complete. Train acc=%.4f Test acc=%.4f", metrics["train_accuracy"], metrics["accuracy"])
    LOGGER.info("Saved model: %s", config.svm_model_path)
    LOGGER.info("Saved label encoder: %s", config.label_encoder_path)
    return metrics
