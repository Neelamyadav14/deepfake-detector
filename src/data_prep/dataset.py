"""
Loads the extracted face-crop frames (from extract_frames.py) into
train/val/test splits.

IMPORTANT: splits are done by VIDEO, not by individual frame. If you
split by frame, frames from the same video can end up in both train and
test, letting the model "cheat" by recognizing the specific video rather
than learning general deepfake patterns — a real, common mistake worth
avoiding and worth mentioning if asked about your evaluation methodology.
"""
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)


def load_frame_dataset(frames_dir: Path, test_size=0.2, val_size=0.1, seed=42):
    """
    Returns (train, val, test), each a tuple of (image_arrays, labels, video_ids).
    video_ids let Stage 2 group frames back into per-video sequences.
    """
    real_files = sorted((frames_dir / "real").glob("*.jpg"))
    fake_files = sorted((frames_dir / "fake").glob("*.jpg"))

    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    video_ids = [f.stem.rsplit("_frame", 1)[0] for f, _ in all_files]

    unique_videos = sorted(set(video_ids))
    train_videos, temp_videos = train_test_split(
        unique_videos, test_size=(test_size + val_size), random_state=seed
    )
    val_videos, test_videos = train_test_split(
        temp_videos, test_size=test_size / (test_size + val_size), random_state=seed
    )

    def build_split(video_set):
        images, labels, ids = [], [], []
        for (fpath, label), vid in zip(all_files, video_ids):
            if vid in video_set:
                img = cv2.imread(str(fpath))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(label)
                ids.append(vid)
        return np.array(images), np.array(labels), ids

    train = build_split(set(train_videos))
    val = build_split(set(val_videos))
    test = build_split(set(test_videos))

    print(f"Train: {len(train[0])} frames from {len(train_videos)} videos")
    print(f"Val:   {len(val[0])} frames from {len(val_videos)} videos")
    print(f"Test:  {len(test[0])} frames from {len(test_videos)} videos")

    return train, val, test