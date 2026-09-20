# ArcTrace 🎯

**Face identity search across video** — upload a reference photo, drop in your footage, and ArcTrace finds every frame where that face appears (independent of clothing/outfit, since only the face is compared).

## What it does

- 📸 Takes **one reference photo** of any person / character
- 🎬 Scans video frame-by-frame using face-recognition embeddings
- 🟢 Draws bounding boxes + confidence scores on matched frames
- 🖼️ Shows **full-resolution, lossless face-crop thumbnails** for every match point
- ⬇️ Lets you download the **annotated video**

## Stack

FastAPI + a vanilla HTML/JS frontend (`main.py` + `static/`), using **InsightFace buffalo_l** for face detection, embedding, and gender attribute.

## Quick start

```bash
# 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
uvicorn main:app --reload
# open http://localhost:8000

# (Windows) if you ever see "InsightFace not installed" in the UI, it means the
# server started with the global Python instead of .venv (insightface only lives
# in .venv). Either activate .venv first, or just double-click / run:
run.bat
```

## Features

- **Server-Sent Events** for live progress streaming, including live in-video scan
  progress (percentage + frame count) so long videos don't look frozen mid-scan
- Live thumbnail strip appearing as matches are found
- Animated progress bar
- Stats dashboard (total matches, frames scanned, duration)
- Annotated video download with labelled bounding boxes
- Face-crop thumbnails kept at native size/aspect ratio and encoded as lossless PNG — no stretching, no compression artifacts

## Settings

Match sensitivity, scan rate, and face-detect confidence are adjustable sliders in the UI; the gender filter is a toggle. Server-side defaults (also the clamped safety bounds) are:

| Setting | Default | Description |
|---------|---------|-------------|
| **Match threshold** | 0.50 | Cosine similarity cutoff. Lower = more matches, more false positives. |
| **Scan interval** | every 2nd frame | How often to sample a frame from the video. |
| **Face detect confidence** | 0.70 | Minimum detection confidence to consider a face at all. |
| **Gender filter** | on | Blocks opposite-gender candidates using the reference photo's detected gender (InsightFace's built-in gender attribute, run on the same aligned crop the detector already produced). |
| **Minimum face size** | 32px | Faces smaller than this on their short side are skipped — too small to embed reliably. |

## Model

Uses **InsightFace buffalo_l**. All processing happens locally — nothing is sent to any server.
