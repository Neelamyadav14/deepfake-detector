"""
Extracts frames from videos and crops the face region from each frame.

Expects an input directory structured as:
    <video-dir>/real/*.mp4
    <video-dir>/fake/*.mp4

(This matches how most public deepfake datasets, including Celeb-DF and
FaceForensics++ subsets, are typically organized.)

Uses OpenCV's built-in Haar cascade for face detection — deliberately not
a heavier dependency like MTCNN or dlib, since those require extra
compilation steps that add friction for a solo portfolio project. This is
a real tradeoff: Haar cascades are less accurate than modern face
detectors, and that's worth naming honestly in your writeup rather than
pretending otherwise.
"""
import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm

FRAMES_PER_VIDEO = 15  # sample this many evenly-spaced frames per video
FACE_SIZE = (224, 224)  # standard input size for EfficientNetB0

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_faces_from_video(video_path: Path, frames_per_video: int = FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = [
        int(i * total_frames / frames_per_video) for i in range(frames_per_video)
    ]

    faces = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(detected) == 0:
            continue

        # Take the largest detected face (most likely the main subject)
        x, y, w, h = max(detected, key=lambda b: b[2] * b[3])
        face_crop = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, FACE_SIZE)
        faces.append(face_resized)

    cap.release()
    return faces


def process_directory(video_dir: Path, output_dir: Path):
    for label in ["real", "fake"]:
        src_dir = video_dir / label
        if not src_dir.exists():
            print(f"WARNING: {src_dir} does not exist, skipping")
            continue

        out_dir = output_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        video_files = list(src_dir.glob("*.mp4")) + list(src_dir.glob("*.avi"))
        print(f"Processing {len(video_files)} '{label}' videos...")

        frame_count = 0
        for video_path in tqdm(video_files):
            faces = extract_faces_from_video(video_path)
            for i, face in enumerate(faces):
                out_path = out_dir / f"{video_path.stem}_frame{i}.jpg"
                cv2.imwrite(str(out_path), face)
                frame_count += 1

        print(f"  Saved {frame_count} face crops for '{label}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True,
                         help="Directory with real/ and fake/ subfolders of videos")
    parser.add_argument("--output-dir", type=str, default="data/frames")
    args = parser.parse_args()

    process_directory(Path(args.video_dir), Path(args.output_dir))
