# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ArcTrace — face-identity search across video. Upload one reference photo of a person and one or more videos; it scans the video frame-by-frame, computes a face-recognition embedding per detected face, compares it to the reference via cosine similarity, and reports timestamps/thumbnails where the same person appears (independent of clothing/outfit changes, since only the face is embedded).

Single implementation: **FastAPI + static HTML/JS**, `main.py` + `static/`, face model is ArcFace via **InsightFace** (`buffalo_l`).

Run: `uvicorn main:app --reload` (serves `http://localhost:8000`)

> History note: this repo used to also ship a Streamlit UI (`streamlit_app.py`, DeepFace-based) as a second, independent implementation. It was removed because the two apps never shared code and drifted out of sync — a fix in one had zero effect on the other, which was the single biggest source of wasted cycles in this repo's history. If you see a stray reference to Streamlit/DeepFace anywhere, it's stale — `main.py` is the only app now.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Run
uvicorn main:app --reload          # http://localhost:8000

# Syntax/import check (no test suite exists in this repo)
python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read())"
```

There is no linter, formatter, or test suite configured — don't invent commands for them. Sanity-check changes by importing the module (`python -c "import main"`) and, for anything touching frame/crop geometry, a small synthetic-video/synthetic-array script: build a tiny `cv2.VideoWriter` clip or a `np.zeros` array in-process, run the function under test directly, assert on the numbers — no fixtures directory exists.

## Architecture

### Pipeline shape

1. Build a normalized embedding from the reference photo (must contain exactly one clear face; the larger/best-scoring face is used if multiple are detected).
2. Open the video with `cv2.VideoCapture`, iterate frames, run face detection on a sampled subset (`sample_every` frames — not every frame, for speed).
3. For each detected face above a detection-confidence gate: embed it, cosine-similarity it against the reference embedding, keep it if `score >= threshold`.
4. Draw boxes on every frame of an output video (`cv2.VideoWriter`) and periodically snapshot a face-crop thumbnail (base64 PNG, lossless) for the UI, throttled by a minimum time gap between recorded detections so near-duplicate frames don't spam the results.

### Known false-positive sources (check first when a bug report is "it matched the wrong person / gave a nonsense high score")

- **Tiny faces:** a face bbox smaller than `MIN_FACE_PX` (32px) on its short side upscales to almost nothing at the recognizer's 112×112 input and produces an unreliable embedding.
- **Letterbox/pillarbox bars in thumbnails (`_content_rect`):** if you ever touch this function, preserve the union-across-sampled-frames approach (a column/row counts as a black bar only if it's black in *every* sampled frame, sampled evenly across the whole video). The earlier version picked the tightest box from a single frame, so one dark scene or a fade-in intro got mistaken for permanent letterboxing and clamped every subsequent crop into a too-narrow window — reproduced and fixed via a synthetic pillarboxed video with one all-black frame injected.

### Thumbnail/crop quality (`_tight_crop`)

- Crops are kept at their **native pixel size and aspect ratio** — never force-resized to a fixed square. An earlier version did `cv2.resize(crop, (192, 192))`, which stretched/distorted any non-square face box (most of them). Don't reintroduce a forced resize here; if a display size is needed, do it in CSS (`object-fit: cover` on a fixed-size tile — already wired up in `static/styles.css` for `.live-thumb` and `.gallery-thumb`), not by warping the source pixels.
- Thumbnails are encoded as **PNG** (`cv2.imencode(".png", ...)`), which is lossless regardless of the compression-level parameter — PNG's compression level only trades encode speed for file size, never image quality. Do not switch this back to JPEG (even at quality 100, JPEG is still lossy) unless payload size becomes a real problem, and if so, raise it with the user first since it's a quality tradeoff.
- The frontend (`static/app.js`) expects `data:image/png;base64,...` — if the encoding format ever changes, update both call sites there (`live-thumb` and `gallery-thumb` rendering) in the same change.

### Frontend controls (`static/index.html`)

Match sensitivity (`threshold`), scan rate (`sample_every`), and face-detect confidence (`det_conf`) are exposed as manual sliders in the controls row, defaulting to the same values as `main.py`'s `Form()` defaults (0.50 / 2 frames / 0.70). `_clamp_params()` still enforces safe bounds server-side regardless of what the client sends, so a bad/edited value from the browser can't push the pipeline outside sane limits. The gender-filter checkbox is the other user-facing control.

### Live per-video progress (SSE)

`_process_video` takes an optional `progress_cb(frames_scanned, total_frames)` and calls it roughly every 0.5s of wall-clock time (plus once more at the end). `/api/analyze/stream` bridges this thread-side callback into the async SSE generator via an `asyncio.Queue` (`loop.call_soon_threadsafe(queue.put_nowait, ...)`), emitting a `video_progress` event consumed by `static/app.js`. This exists because `_process_video` used to run to completion inside a single executor call with zero intermediate events — on a long/high-res video on CPU that could take minutes with the progress bar frozen at "video_start," which read as the app being stuck even though it was working. Don't remove the progress-callback plumbing or go back to a single blocking `run_in_executor` call without it.

### Known inconsistency: thumbnail MIME type

`_tight_crop` encodes thumbnails as PNG (`cv2.imencode(".png", ...)`), but `static/app.js` renders them as `data:image/jpeg;base64,...` (both `live-thumb` and `gallery-thumb` call sites). This works in practice because Chrome/Firefox sniff the actual image bytes rather than trusting the declared data-URI MIME type, but it's not something to rely on indefinitely — if thumbnails ever fail to render in some browser/context, this mismatch is the first thing to check. Fixing it means changing both `data:image/jpeg` occurrences in `app.js` to `data:image/png`.

### `main.py` specifics

- InsightFace `buffalo_l` is loaded once at import time into the module-level `face_app` global (`_load_model()`), not per-request.
- Gender filtering uses InsightFace's own `face['gender']` attribute (0=Female, 1=Male — confirmed against `insightface/model_zoo/attribute.py`, `gender = np.argmax(pred[:2])`) to hard-block opposite-gender candidates when the "gender filter" toggle is on and the reference photo's gender was determined. This runs on the same aligned crop the detector already produced, not a second re-crop/re-detect pass.
- `/api/analyze` (plain JSON response) and `/api/analyze/stream` (SSE, `event: start/ref_ready/video_start/video_progress/video_done/video_error/done`) share `_clamp_params()` and `_save_job_uploads()` — extend both if you add a new request parameter, don't duplicate clamping logic back into each route.
- Uploaded files and rendered output videos live under `data/uploads/` and `data/outputs/` respectively, keyed by a per-request `job_id` (uuid4 hex); `/media` is mounted as a static file server over `data/` for the frontend to fetch annotated output videos and `/static` serves the vanilla JS/HTML/CSS frontend in `static/`.
- If the app throws `HTTPException(503, "InsightFace not installed.")`, it almost always means the process was started with a Python that isn't `.venv` (e.g. the global/Windows-Store Python, which can have `uvicorn` on its PATH without `insightface`). Use `run.bat` (always invokes `.venv\Scripts\python.exe -m uvicorn`) or activate `.venv` before running `uvicorn` directly.

## Requirements/dependency notes

`requirements.txt` currently lists both the FastAPI/InsightFace stack (insightface, onnxruntime, fastapi, uvicorn, opencv-python, numpy) **and** leftover Streamlit/DeepFace/tf-keras entries from before the Streamlit app was removed. Those three packages install but are unused by anything in this repo — `streamlit_app.py` no longer exists. Harmless but bloats `pip install`; trim them if you're touching this file anyway, but it's not required for the app to run.
