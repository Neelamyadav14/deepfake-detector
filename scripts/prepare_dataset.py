"""
Copies a random subset of images from the downloaded Kaggle dataset
into this project's expected flat structure: data/frames/real/, data/frames/fake/.
"""
import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = PROJECT_ROOT / "data" / "frames"


def copy_subset(source_dir: Path, dest_dir: Path, label: str, n: int, seed: int = 42):
    src = source_dir / label.capitalize()
    dst = dest_dir / label.lower()
    dst.mkdir(parents=True, exist_ok=True)

    all_images = list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png"))
    print(f"Found {len(all_images)} images in {src}")

    random.seed(seed)
    chosen = random.sample(all_images, min(n, len(all_images)))

    for i, img_path in enumerate(chosen):
        dest_name = f"{label.lower()}img{i:05d}_frame0.jpg"
        shutil.copy(img_path, dst / dest_name)

    print(f"Copied {len(chosen)} '{label}' images to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--n-per-class", type=int, default=2000)
    args = parser.parse_args()

    source_dir = Path(args.source)
    copy_subset(source_dir, DEST_DIR, "real", args.n_per_class)
    copy_subset(source_dir, DEST_DIR, "fake", args.n_per_class)

    print("\nDone. Next: python src/train.py --stage cnn")