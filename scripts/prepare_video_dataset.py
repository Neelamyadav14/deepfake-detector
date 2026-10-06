"""
Organizes the Celeb-DF v2 dataset into the real/fake video folder
structure that extract_frames.py expects.
"""
import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = PROJECT_ROOT / "data" / "raw_videos"


def collect_real_videos(source_dir: Path):
    celeb_real = list((source_dir / "Celeb-real").glob("*.mp4"))
    youtube_real = list((source_dir / "YouTube-real").glob("*.mp4"))
    return celeb_real + youtube_real


def main(source: str, n_per_class: int, seed: int = 42):
    source_dir = Path(source)
    random.seed(seed)

    real_videos = collect_real_videos(source_dir)
    fake_videos = list((source_dir / "Celeb-synthesis").glob("*.mp4"))

    print(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    chosen_real = random.sample(real_videos, min(n_per_class, len(real_videos)))
    chosen_fake = random.sample(fake_videos, min(n_per_class, len(fake_videos)))

    real_dest = DEST_DIR / "real"
    fake_dest = DEST_DIR / "fake"
    real_dest.mkdir(parents=True, exist_ok=True)
    fake_dest.mkdir(parents=True, exist_ok=True)

    for i, video_path in enumerate(chosen_real):
        shutil.copy(video_path, real_dest / f"real_{i:04d}.mp4")
    for i, video_path in enumerate(chosen_fake):
        shutil.copy(video_path, fake_dest / f"fake_{i:04d}.mp4")

    print(f"Copied {len(chosen_real)} real videos to {real_dest}")
    print(f"Copied {len(chosen_fake)} fake videos to {fake_dest}")
    print(f"\nDone. Next: python src/data_prep/extract_frames.py --video-dir {DEST_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--n-per-class", type=int, default=150)
    args = parser.parse_args()

    main(args.source, args.n_per_class)