"""
Tests for model architecture correctness and the frame->sequence grouping
logic. These use random weights/data (no real dataset needed) so they can
run in any environment, including CI.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.lstm_temporal import build_lstm_temporal_model


def test_lstm_model_builds_and_predicts():
    model = build_lstm_temporal_model()
    dummy_sequences = np.random.randn(4, 15, 128).astype("float32")
    preds = model.predict(dummy_sequences, verbose=0)
    assert preds.shape == (4, 1)
    assert np.all((preds >= 0) & (preds <= 1))


def test_lstm_masking_handles_padded_frames():
    """A video with fewer real frames (zero-padded) shouldn't crash the model."""
    model = build_lstm_temporal_model()
    sequences = np.random.randn(2, 15, 128).astype("float32")
    sequences[0, 8:] = 0.0  # simulate a video with only 8 real frames
    preds = model.predict(sequences, verbose=0)
    assert preds.shape == (2, 1)


def test_build_sequences_groups_by_video():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from train import build_sequences

    class DummyExtractor:
        def predict(self, X, batch_size=32, verbose=0):
            return np.random.randn(len(X), 128).astype("float32")

    X = np.random.randn(23, 224, 224, 3).astype("float32")
    y = np.array([0] * 15 + [1] * 8)
    video_ids = ["videoA"] * 15 + ["videoB"] * 8

    seqs, labels = build_sequences(X, y, video_ids, DummyExtractor())
    assert seqs.shape == (2, 15, 128)
    assert list(labels) == [0, 1]
    # videoB only had 8 real frames -> remaining 7 should be zero-padded
    assert np.allclose(seqs[1][8:], 0)
