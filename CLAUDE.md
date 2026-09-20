# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ArcTrace — face-identity search across video. Upload one or more reference photos (each with a character/actor name) and one or more videos; it scans the video frame-by-frame, computes a face-recognition embedding per detected face, compares it against every reference via cosine similarity, and reports per-character timestamps/thumbnails where that person appears (independent of clothing/outfit changes, since only the face is embedded).

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

1. Build a normalized embedding for **each** reference photo (each must contain exactly one clear face; the larger/best-scoring face is used if multiple are detected) — `ref_ids: list[RefIdentity]`, one per character, each carrying its own `name`.
2. Open the video with `cv2.VideoCapture`, iterate frames, run face detection **once per sampled frame** regardless of how many characters are being searched for. Sampling is rate-based, not frame-count-based (see `scan_fps` below).
3. For each detected face above a detection-confidence gate: embed it **once**, then loop over every reference and cosine-similarity it against that reference's embedding, keeping it as a match for that character if `score >= threshold`. Detection + embedding extraction (the expensive part) never repeats per character — only the cheap dot-product comparison does.
4. Draw boxes on every frame of an output video (`cv2.VideoWriter`), one color per character (`CHARACTER_COLORS`, cycled), labelled with the character's name and score. Separately, per character, track the single highest-confidence match across the whole video and keep **one** full-frame snapshot for it (see below) — not a per-detection thumbnail gallery.

### Known false-positive sources (check first when a bug report is "it matched the wrong person / gave a nonsense high score")

- **Tiny faces:** a face bbox smaller than `MIN_FACE_PX` (32px) on its short side upscales to almost nothing at the recognizer's 112×112 input and produces an unreliable embedding.

### Multi-character search

- `_process_video` takes `ref_ids: list[RefIdentity]` (one per character) instead of a single reference, and every per-video data structure (`detections`, `carry`, `last_det_sec`, `best_overall_*`) is a **list indexed by character**, not a scalar. The frame loop still runs face detection/embedding once per sampled frame; only the `for ci, ref in enumerate(ref_ids):` cosine-similarity comparison repeats per character — don't change this to re-run `face_app.get()` per character, that would multiply the (already CPU-bound) detection cost by the character count for no reason.
- The gender-filter check happens per (face, character) pair, not once per face — a face can be filtered out for one character's gender but still compared against another's, since each reference has its own `gender_int`.
- Route responses nest results as `video["characters"] = [{"name", "detections", "thumbnails", "matchCount"}, ...]` instead of flat `detections`/`thumbnails`/`matchCount` on the video dict. `video["matchCount"]` is now the **sum across characters** (kept for the top-level badge). If you add a new per-match field, add it inside the per-character dict, not the video dict.
- `references` (files) and `character_names` (strings) are parsed by FastAPI as `list[UploadFile]`/`list[str]` from **repeated form fields with the same name**, paired by position — `_save_job_uploads` zips them. The frontend relies on this: each character row in `static/index.html` has its own `<input name="references">` and `<input name="character_names">`, and native `new FormData(form)` collects same-named inputs in DOM order, so row N's photo always pairs with row N's name without any manual JS assembly. Don't reorder rows independently of their photo/name pair, and don't rename these fields without updating both sides.
- `len(character_names) != len(references)` and any blank name is rejected with a 400 before any processing starts (see the validation block at the top of both `/api/analyze` and `/api/analyze/stream`) — keep both routes' validation in sync, same as `_clamp_params`.

### Best-match thumbnail (`_frame_to_png_b64`, per-character best-overall tracking in `_process_video`)

- The thumbnail is **one full video frame per character, not a face crop.** `_process_video` tracks `best_overall_score[ci]`/`best_overall_frame[ci]` across the whole scan per character and keeps the frame with that character's single highest cosine-similarity score (drawn with bounding boxes for **all** characters matched in that frame already annotated — `frame.copy()` is taken once, after the full per-character draw loop, and shared across whichever characters happen to have their best-match frame be this one). `_process_video` returns each character's `thumbnails` as a list with **at most one entry** — the best match for that character — never a gallery of every hit. This was a deliberate change from an earlier version that produced a tight face-only crop (via a now-removed `_tight_crop`/`_content_rect` pair) for every throttled detection: the user explicitly wants the whole body/scene visible, one result per character, and it should be the most confident (least likely to be a false positive) one. Don't reintroduce a per-detection thumbnail loop or a face-only crop without discussing it first.
- If two characters both match in the same frame, each one's "best" thumbnail may show the same frame (with both boxes drawn, since drawing happens once per frame for all characters before any snapshot is taken) — this is intentional context ("who else is in this shot"), not a bug to fix by re-rendering separate clean frames per character.
- The best-overall check (`if sampled:` — i.e. `frame_idx % frame_step == 0`) only considers genuinely-detected frames, not frames using `carry` (the stale boxes reused on non-sampled frames) — so a carried box can't "win" the best-match slot on a technicality.
- The `detections` list (timeline/chips, per-character `matchCount`) is untouched by this — it still records every throttled (≥0.5s apart, tracked independently per character via `last_det_sec[ci]`) match instance for the timeline, independent of which single frame becomes that character's thumbnail.
- Thumbnails are encoded as **PNG** (`cv2.imencode(".png", ...)`), which is lossless regardless of the compression-level parameter — PNG's compression level only trades encode speed for file size, never image quality. Do not switch this to JPEG (even at quality 100, JPEG is still lossy) unless payload size becomes a real problem, and if so, raise it with the user first since it's a quality tradeoff.
- `static/styles.css`'s `.gallery-thumb`/`.live-thumb` use `object-fit: contain` (not `cover`) against a dark background, specifically because the thumbnail is now a whole (likely non-square, e.g. 16:9) frame — `cover` would crop most of it away, defeating the point of showing the whole body/scene. Don't switch back to `cover` for these.

### Frontend controls (`static/index.html`)

Match sensitivity (`threshold`), scan rate (`scan_fps`), and face-detect confidence (`det_conf`) are exposed as manual sliders in the controls row, defaulting to the same values as `main.py`'s `Form()` defaults (0.50 / 2 fps / 0.70). `_clamp_params()` still enforces safe bounds server-side regardless of what the client sends, so a bad/edited value from the browser can't push the pipeline outside sane limits. The gender-filter checkbox is the other user-facing control. The character list itself (photo + name rows, add/remove) is managed in `app.js` via `wireCharacterRow()`/`updateRemoveButtons()` — a freshly-added row is cloned from the first `.character-row` and must be re-wired (event listeners don't survive `cloneNode`).

### Scan rate is fps-based, not frame-count-based (`scan_fps`)

`_process_video` takes `scan_fps` (1-4, default 2 — "analyze N frames per second of video") instead of the old `sample_every` ("analyze every Nth raw frame"). It converts this to an actual frame step once per video: `frame_step = max(1, round(fps / scan_fps))`. This matters because "every 2nd frame" on a 60fps video means 30 analyzed frames/sec — 15x more CPU-bound detection work than the same setting on a 4fps video — so the old frame-count knob silently punished high-fps footage. Rate-based sampling keeps analysis cost proportional to video *duration*, not resolution/frame-rate. If you ever need a frame-count-based knob back for some reason, don't reintroduce it as the primary control — derive it from `scan_fps` and the video's own fps, the same way `frame_step` does now.

### CPU thread tuning for InsightFace (`_load_model`) — benchmark before changing

`_load_model()` pins ONNX Runtime to `intra_op_num_threads=4`, `inter_op_num_threads=1`, `ORT_SEQUENTIAL`. This is **not the obvious "use more cores = faster" choice** — it was benchmarked against real video frames on a 16-core machine and the relationship is a U-shaped curve, not monotonic:

| config | ms/frame |
|---|---|
| intra=1 | 1572 |
| intra=2 | 746 |
| intra=3 | 521 |
| **intra=4** | **455** ← fastest |
| intra=6 | 544 |
| intra=8 | 632 |
| untouched ORT default (no sess_options) | 781 |
| intra=16, inter=16, ORT_PARALLEL ("use all cores") | 1387 ← slower than doing nothing |

buffalo_l's per-model compute graphs (SCRFD detector, ArcFace recognizer, genderage) are small and run sequentially per frame — past ~4 threads, thread-scheduling/synchronization overhead per inference call outweighs the extra parallelism, and `ORT_PARALLEL` + high `inter_op_num_threads` makes it worse since these models have no independent parallel branches to exploit. If you're tempted to "use all cores" for CPU speed here, don't — re-run a benchmark like the one above on the target machine first; the optimal thread count depends on model size, not core count, and can be *worse than the default* if set too high.

### Live per-video progress (SSE)

`_process_video` takes an optional `progress_cb(frames_scanned, total_frames)` and calls it roughly every 0.5s of wall-clock time (plus once more at the end). `/api/analyze/stream` bridges this thread-side callback into the async SSE generator via an `asyncio.Queue` (`loop.call_soon_threadsafe(queue.put_nowait, ...)`), emitting a `video_progress` event consumed by `static/app.js`. This exists because `_process_video` used to run to completion inside a single executor call with zero intermediate events — on a long/high-res video on CPU that could take minutes with the progress bar frozen at "video_start," which read as the app being stuck even though it was working. Don't remove the progress-callback plumbing or go back to a single blocking `run_in_executor` call without it.

### `main.py` specifics

- InsightFace `buffalo_l` is loaded once at import time into the module-level `face_app` global (`_load_model()`), not per-request.
- Gender filtering uses InsightFace's own `face['gender']` attribute (0=Female, 1=Male — confirmed against `insightface/model_zoo/attribute.py`, `gender = np.argmax(pred[:2])`) to hard-block opposite-gender candidates per (face, character) pair when the "gender filter" toggle is on and that character's reference photo's gender was determined. This runs on the same aligned crop the detector already produced, not a second re-crop/re-detect pass.
- `/api/analyze` (plain JSON response) and `/api/analyze/stream` (SSE, `event: start/ref_ready/video_start/video_progress/video_done/video_error/done`) share `_clamp_params()` and `_save_job_uploads()` — extend both if you add a new request parameter, don't duplicate clamping/validation logic back into each route.
- Uploaded files and rendered output videos live under `data/uploads/` and `data/outputs/` respectively, keyed by a per-request `job_id` (uuid4 hex); reference photos are saved as `{job_id}_ref{i}{suffix}` (one per character, index-ordered). `/media` is mounted as a static file server over `data/` for the frontend to fetch annotated output videos and `/static` serves the vanilla JS/HTML/CSS frontend in `static/`.
- If the app throws `HTTPException(503, "InsightFace not installed.")`, it almost always means the process was started with a Python that isn't `.venv` (e.g. the global/Windows-Store Python, which can have `uvicorn` on its PATH without `insightface`). Use `run.bat` (always invokes `.venv\Scripts\python.exe -m uvicorn`) or activate `.venv` before running `uvicorn` directly.

## Requirements/dependency notes

`requirements.txt` currently lists both the FastAPI/InsightFace stack (insightface, onnxruntime, fastapi, uvicorn, opencv-python, numpy) **and** leftover Streamlit/DeepFace/tf-keras entries from before the Streamlit app was removed. Those three packages install but are unused by anything in this repo — `streamlit_app.py` no longer exists. Harmless but bloats `pip install`; trim them if you're touching this file anyway, but it's not required for the app to run.
