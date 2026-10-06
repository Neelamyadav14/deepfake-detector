# Deepfake Video Detector

Detects manipulated (deepfake) images and videos by combining a CNN
(transfer learning) for per-frame visual features with an LSTM for
catching temporal inconsistencies across frames — the kind of subtle
flicker/warping that a single-frame classifier misses but a sequence
model can catch.

## Staged build

This project was deliberately built in stages, each one a complete,
demoable checkpoint on its own.

### Stage 1: CNN baseline (frame-level classification) — complete
- Extract frames from images/videos, detect and crop faces
- Fine-tune a pretrained CNN (EfficientNetB0 via transfer learning) to
  classify individual frames/images as real or fake

### Stage 2: LSTM temporal model — complete
- Take the CNN's per-frame features (not its final prediction — the
  intermediate feature vector) for a sequence of frames from the same video
- Feed that sequence into an LSTM, which learns to spot inconsistencies
  across frames that no single frame reveals on its own

### Stage 3 (optional, not attempted): GAN-based data augmentation
- Deepfake datasets are limited and imbalanced; a GAN can generate
  additional synthetic training examples to help the classifier generalize
- Left as a documented future extension rather than attempted, to protect
  the project's actual finish line

## Why each technique is actually here (not just keyword-checking)

- **Transfer learning**: training a CNN from scratch on a limited deepfake
  dataset would perform badly. A model pretrained on millions of general
  images already knows how to see edges, textures, faces — we just adapt
  that knowledge to this specific task.
- **Fine-tuning**: after initial training with the pretrained base fully
  frozen, unfreezing the top layers and continuing training with a low
  learning rate lets the model adapt its visual features to this specific
  task rather than relying only on generic ImageNet features. This was
  the single biggest accuracy improvement for Stage 1 (see Results).
- **Data augmentation**: random flips, rotation, and zoom during training
  help the model generalize instead of memorizing exact pixel patterns
  from a few thousand training images.
- **LSTM**: deepfakes often look convincing frame-by-frame but flicker or
  warp slightly across frames. A single-frame CNN can't see that; a
  sequence model can.
- **GAN (optional, not built)**: the same generative technique used to
  *create* deepfakes can help generate synthetic training data to
  *detect* them — a real, documented technique in this field, left as a
  future extension.

## Data

**Stage 1 (images):** the ["Deepfake and Real Images" dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
(~190,000 images, pre-split into Train/Validation/Test, Real/Fake
folders). A 10,000-image subset (5,000 real, 5,000 fake) was used for
training to keep training time realistic on a CPU-only machine.

**Stage 2 (video):** [Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)
(890 real videos across Celeb-real + YouTube-real, 5,639 fake videos in
Celeb-synthesis). A 1,000-video subset (500 real, 500 fake) was used, out
of the 6,529 available, to keep frame extraction and training time
realistic on CPU-only hardware.

## Quickstart

```bash
pip install -r requirements.txt

# --- Stage 1: CNN baseline (images) ---
python scripts/prepare_dataset.py --source "<path_to_Dataset>\Train" --n-per-class 5000
python src/train.py --stage cnn
python src/train.py --stage finetune    # fine-tune top layers for better accuracy
python src/evaluate.py --stage cnn

# --- Stage 2: LSTM temporal model (video) ---
python scripts/prepare_video_dataset.py --source "<path_to_Celeb-DF>" --n-per-class 500
python src/data_prep/extract_frames.py --video-dir data/raw_videos
python src/train.py --stage lstm
python src/evaluate.py --stage lstm

# --- Serve + demo ---
uvicorn src.api.main:app --reload
# open dashboard/index.html (Video tab = Stage 2, Image tab = Stage 1)
```

**Important:** `data/frames` is used as the working folder for BOTH
Stage 1 images and Stage 2 extracted video frames. Clear it
(`rmdir /s /q data\frames` on Windows) before switching between preparing
an image dataset and a video dataset, or the two will mix (see the real
bug documented below).

## Results

| Model | Accuracy | AUC |
|---|---|---|
| Stage 1: CNN baseline (frame-level, image dataset) | 84.2% | 0.924 |
| Stage 2: CNN + LSTM (video-level, Celeb-DF v2, 1,000 videos) | 75.1% | 0.824 |

**Stage 1 methodology:** trained on a 10,000-image subset (5,000 real,
5,000 fake) of the "Deepfake and Real Images" dataset. Used transfer
learning on EfficientNetB0 with data augmentation (flip, rotation, zoom),
then fine-tuned the top 30 layers of the pretrained base with a low
learning rate (1e-5) after initial training. Precision/recall are
balanced across both classes (0.83–0.86), and video-level accuracy
matches frame-level exactly, indicating consistent predictions.

**Stage 1 improvement path:** started at 77.0%/0.841 AUC with a frozen
base on 2,000 images/class. Increasing to 5,000 images/class raised
validation accuracy to 80.4%. Fine-tuning the top 30 layers of the
pretrained base pushed final test accuracy to 84.2%/0.924 AUC.

**Stage 2 methodology:** trained on 500 real videos (Celeb-real +
YouTube-real) and 500 fake videos (Celeb-synthesis) from Celeb-DF v2, 15
frames sampled per video with face detection/cropping, run through the
Stage 1 CNN for per-frame features, then an LSTM trained for 25 epochs on
these sequences. Evaluated on 201 held-out videos never seen during
training.

**Stage 2 improvement path:** an initial run on 600 videos (300/class, 15
epochs) scored 72.7%/0.794 AUC. Increasing to 1,000 videos (500/class)
and 25 epochs raised this to 75.1%/0.824 AUC, despite validation loss
showing mild overfitting in later epochs — the larger, more diverse
training set outweighed this.

## Two real bugs worth documenting

**1. Data contamination between stages.** An early version of the Stage 2
evaluation incorrectly showed 87.2%/0.942 AUC. Root cause: `data/frames`
still contained ~10,000 leftover Stage 1 images when video frames were
extracted into the same folder, so the LSTM's training data was secretly
a mix of real video sequences and degenerate "1-frame videos" from the
image dataset. Caught by noticing the logged video count during training
(7,209) didn't match the number of videos actually prepared (300). Fixed
by clearing `data/frames` before extracting video frames — the corrected,
video-only result is the 72.7%/0.794 AUC figure above.

**2. Train/serve skew in the live video API.** The `/predict` endpoint
was misclassifying fake videos as real with high confidence during
manual dashboard testing, despite the formal evaluation showing much
better accuracy. Root cause: OpenCV reads video frames in BGR channel
order, and the training pipeline explicitly converts to RGB before
feeding the model — but the live video inference endpoint was missing
this conversion, feeding color-inverted frames to the model. Fixed by
adding the same BGR-to-RGB conversion used in training and in the
`/predict_image` endpoint (which already had it correctly).

## Honest limitations

- Video-level detection (75.1%) still underperforms the image-level
  baseline (84.2%) — video introduces harder real-world variation (face
  detection misses frames, motion blur, compression artifacts) that a
  curated image dataset doesn't have.
- Only 1,000 of the 6,529 available Celeb-DF videos were used, and only
  10,000 of ~190,000 available images, to keep training time realistic
  on CPU-only hardware. More data would likely improve both stages
  further.
- This detects known-style manipulation artifacts; it will NOT reliably
  catch novel deepfake generation techniques absent from training.
- Face detection uses OpenCV's Haar cascade (not a modern deep
  learning-based detector like MTCNN), a deliberate tradeoff to avoid
  extra heavy dependencies — some videos had fewer faces detected than
  expected, which reduces prediction reliability on those specific cases.

## What's next

- Train on more of the available data (full Celeb-DF, more of the image
  dataset) given more time/compute
- Stage 3: GAN-based synthetic data augmentation (documented, not built)
- Replace the Haar cascade face detector with a modern deep
  learning-based one for more reliable face detection