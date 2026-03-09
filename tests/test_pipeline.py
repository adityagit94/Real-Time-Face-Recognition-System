import numpy as np

from src.config import Config
from src import trainer


class DummyEmbedder:
    def embed_batch(self, faces):
        return np.random.rand(len(faces), 512)


class DummyModel:
    def fit(self, x, y):
        self.classes_ = np.unique(y)

    def predict(self, x):
        return np.zeros(len(x), dtype=int)


def test_training_pipeline_smoke(monkeypatch, tmp_path):
    faces = np.random.randint(0, 255, size=(6, 160, 160, 3), dtype=np.uint8)
    labels = np.array(["a", "a", "a", "b", "b", "b"])

    monkeypatch.setattr(trainer, "load_dataset_faces", lambda *args, **kwargs: (faces, labels))
    monkeypatch.setattr(trainer, "FaceEmbedder", lambda: DummyEmbedder())
    monkeypatch.setattr(trainer, "SVC", lambda **kwargs: DummyModel())

    cfg = Config(
        dataset_dir="data/raw",
        svm_model_path=str(tmp_path / "svm_model.pkl"),
        label_encoder_path=str(tmp_path / "label_encoder.pkl"),
        metrics_path=str(tmp_path / "metrics.json"),
        confusion_matrix_path=str(tmp_path / "cm.png"),
        test_size=0.33,
    )
    metrics = trainer.train_pipeline(cfg)
    assert "accuracy" in metrics
