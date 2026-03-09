"""Central configuration for training and inference."""
from dataclasses import dataclass

from src.paths import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR


@dataclass(frozen=True)
class Config:
    """Project runtime configuration.

    Attributes are string paths to keep argparse interop straightforward.
    """

    dataset_dir: str = str(RAW_DATA_DIR)
    svm_model_path: str = str(MODELS_DIR / "svm_model.pkl")
    label_encoder_path: str = str(MODELS_DIR / "label_encoder.pkl")
    metrics_path: str = str(REPORTS_DIR / "metrics.json")
    confusion_matrix_path: str = str(REPORTS_DIR / "confusion_matrix.png")
    image_size: tuple[int, int] = (160, 160)
    crop_margin: float = 0.1
    test_size: float = 0.2
    random_seed: int = 17
    confidence_threshold: float = 0.6


DEFAULT_CONFIG = Config()
