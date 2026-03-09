import numpy as np
import pytest

from src.embedder import FaceEmbedder


class DummyModel:
    def embeddings(self, x):
        return np.ones((x.shape[0], 512), dtype=np.float32)


def test_embed_one_shape(monkeypatch):
    embedder = FaceEmbedder()
    monkeypatch.setattr(embedder, "model", DummyModel())
    out = embedder.embed_one(np.zeros((160, 160, 3), dtype=np.uint8))
    assert out.shape == (512,)


def test_embed_batch_input_validation():
    embedder = FaceEmbedder()
    with pytest.raises(ValueError):
        embedder.embed_batch(np.zeros((160, 160, 3), dtype=np.uint8))
