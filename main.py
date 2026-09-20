"""ArcTrace – ArcFace identity search across video."""
from __future__ import annotations

import asyncio
import base64
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, AsyncGenerator, Callable

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ArcTrace")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/media",  StaticFiles(directory=DATA_DIR), name="media")

face_app: "FaceAnalysis | None" = None

def _load_model() -> None:
    global face_app
    if INSIGHTFACE_AVAILABLE and face_app is None:
        fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=0, det_size=(640, 640))
        face_app = fa

_load_model()

# ── InsightFace returns face['gender'] as int: 0=Female, 1=Male ─────────────
GENDER_INT_MAP = {0: "F", 1: "M"}

# Faces smaller than this (px, shorter bbox side) give unreliable embeddings.
MIN_FACE_PX = 32

@dataclass
class RefIdentity:
    embedding: np.ndarray
    gender_int: int | None   # 0=F, 1=M, None=unknown


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def ts(s: float) -> str:
    m, r = divmod(s, 60)
    return f"{int(m):02d}:{r:05.2f}"


def build_reference(path: Path) -> RefIdentity:
    if face_app is None:
        raise HTTPException(503, "InsightFace not installed.")
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, "Reference image could not be read.")
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(400, "No face detected in the reference photo. Use a clear, well-lit, front-facing image.")
    best = max(faces, key=lambda f: f.det_score)
    emb  = normalize(best.embedding.astype(np.float32))

    # InsightFace buffalo_l stores gender as face['gender']: 0=F, 1=M
    raw_gender = best.get("gender")   # uses __getitem__ on Face object
    gender_int = int(raw_gender) if raw_gender is not None and int(raw_gender) in (0, 1) else None

    return RefIdentity(embedding=emb, gender_int=gender_int)


# ─── Content-rect detection: find non-black area (letterbox/pillarbox) ───────

def _content_rect(cap: cv2.VideoCapture, width: int, height: int, samples: int = 24) -> tuple[int,int,int,int]:
    """
    Find real letterbox/pillarbox bars by sampling frames spread across the
    whole video and UNIONING their non-black regions.

    A column/row only counts as a black bar if it is black in EVERY sampled
    frame. This is deliberate: taking the tightest box from a single frame
    (the previous approach) means one dark scene or a fade-in intro gets
    mistaken for letterboxing, and every crop for the rest of the video gets
    clamped into that too-narrow window.
    """
    saved_pos   = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions   = sorted(set(int(i * total / samples) for i in range(samples))) if total > samples \
                  else list(range(total or samples))

    col_ever = np.zeros(width,  dtype=bool)
    row_ever = np.zeros(height, dtype=bool)
    checked  = 0

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # A column/row is "content" in this frame if >8% of its pixels are above threshold 15
        col_ever |= (grey > 15).sum(axis=0) > (height * 0.08)
        row_ever |= (grey > 15).sum(axis=1) > (width  * 0.08)
        checked += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

    if checked == 0:
        return 0, 0, width, height

    cols = np.where(col_ever)[0]
    rows = np.where(row_ever)[0]
    if cols.size < 20 or rows.size < 20:
        return 0, 0, width, height
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


# ─── Tight face crop (no black bars) ─────────────────────────────────────────

def _tight_crop(frame: np.ndarray, box: np.ndarray,
                content: tuple[int,int,int,int]) -> str:
    """
    Crop proportionally around the face, strictly clamped to content area.

    Key fix: padding on each side is min(desired_pad, available_space_in_content_area)
    so we NEVER go into the black letterbox bars.

    The crop is kept at its native pixel size/aspect ratio and encoded losslessly
    (PNG) — no forced square resize (which used to stretch non-square face boxes)
    and no JPEG compression artifacts. The frontend's `object-fit: cover` tiles
    already display any aspect ratio cleanly without warping the source pixels.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cx1, cy1, cx2, cy2 = content

    # Clamp face bbox to content area first
    x1 = max(x1, cx1); y1 = max(y1, cy1)
    x2 = min(x2, cx2); y2 = min(y2, cy2)
    if x2 <= x1 or y2 <= y1:
        return ""

    fw, fh = x2 - x1, y2 - y1

    # Desired padding: 30% of face width / height
    want_px = int(fw * 0.30)
    want_py = int(fh * 0.30)

    # Available space within content area (no going into black bars)
    avail_left  = x1 - cx1
    avail_right = cx2 - x2
    avail_top   = y1 - cy1
    avail_bot   = cy2 - y2

    pad_x = max(0, min(want_px, avail_left,  avail_right))
    pad_y = max(0, min(want_py, avail_top,   avail_bot))

    cx1r = x1 - pad_x
    cy1r = y1 - pad_y
    cx2r = x2 + pad_x
    cy2r = y2 + pad_y

    crop = frame[cy1r:cy2r, cx1r:cx2r]
    if crop.size == 0:
        return ""

    # Reject very dark crops (still in black zone)
    if cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean() < 12:
        return ""

    ok, buf = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


# ─── Shared request handling ──────────────────────────────────────────────────

def _clamp_params(threshold: float, sample_every: int, det_conf: float) -> tuple[float, int, float]:
    threshold    = max(0.25, min(float(threshold),    0.90))
    sample_every = max(1,    min(int(sample_every),    30))
    det_conf     = max(0.30, min(float(det_conf),     0.95))
    return threshold, sample_every, det_conf


def _save_job_uploads(
    job_id: str, reference: UploadFile, videos: list[UploadFile],
) -> tuple[Path, list[tuple[Path, str]]]:
    ref_path = UPLOAD_DIR / f"{job_id}_ref{Path(reference.filename or '.jpg').suffix}"
    save_upload(reference, ref_path)

    video_paths: list[tuple[Path, str]] = []
    for i, vid in enumerate(videos):
        suffix = Path(vid.filename or ".mp4").suffix or ".mp4"
        vp = UPLOAD_DIR / f"{job_id}_{i}{suffix}"
        save_upload(vid, vp)
        video_paths.append((vp, vid.filename or f"video-{i+1}"))

    return ref_path, video_paths


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "insightface": INSIGHTFACE_AVAILABLE}


@app.get("/api/debug-ref")
async def debug_ref(img_path: str = "") -> dict:
    """Debug: inspect what InsightFace returns for a face."""
    if not img_path or face_app is None:
        return {"error": "Provide ?img_path=... and ensure InsightFace is loaded."}
    img = cv2.imread(img_path)
    if img is None:
        return {"error": f"Cannot read {img_path}"}
    faces = face_app.get(img)
    out = []
    for f in faces:
        out.append({
            "det_score": float(f.det_score),
            "gender_raw": f.get("gender"),           # 0=F, 1=M
            "age":        f.get("age"),
            "bbox":       f.bbox.tolist(),
        })
    return {"faces": out}


# ─── SSE streaming endpoint ───────────────────────────────────────────────────

@app.post("/api/analyze/stream")
async def analyze_stream(
    character_name: Annotated[str,   Form()],
    reference:      Annotated[UploadFile, File()],
    videos:         Annotated[list[UploadFile], File()],
    threshold:      Annotated[float, Form()] = 0.50,
    sample_every:   Annotated[int,   Form()] = 2,
    max_thumbs:     Annotated[int,   Form()] = 30,
    det_conf:       Annotated[float, Form()] = 0.70,
    use_gender:     Annotated[int,   Form()] = 1,
) -> StreamingResponse:
    if not character_name.strip():
        raise HTTPException(400, "Enter a character name.")
    if not videos:
        raise HTTPException(400, "Upload at least one video.")
    threshold, sample_every, det_conf = _clamp_params(threshold, sample_every, det_conf)

    job_id = uuid.uuid4().hex
    ref_path, video_paths = _save_job_uploads(job_id, reference, videos)

    async def event_gen() -> AsyncGenerator[str, None]:
        def sse(event: str, payload: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

        yield sse("start", {"jobId": job_id, "characterName": character_name.strip(),
                             "videoCount": len(video_paths)})
        try:
            ref_id = await asyncio.get_event_loop().run_in_executor(
                None, build_reference, ref_path)
        except HTTPException as exc:
            yield sse("error", {"message": exc.detail})
            return

        gender_label = GENDER_INT_MAP.get(ref_id.gender_int, "unknown")
        yield sse("ref_ready", {"gender": gender_label,
                                "gender_filter": bool(use_gender) and ref_id.gender_int is not None})

        all_results: list[dict] = []
        loop = asyncio.get_event_loop()
        for idx, (vp, vn) in enumerate(video_paths):
            yield sse("video_start", {"index": idx, "name": vn})

            # Bridge _process_video's synchronous, thread-side progress callback
            # into the async SSE stream via a thread-safe queue — without this,
            # the frontend gets zero feedback for the entire duration of a single
            # video's scan and looks frozen on slow (CPU, long/high-res) videos.
            progress_q: asyncio.Queue = asyncio.Queue()

            def progress_cb(scanned: int, total: int, _q=progress_q, _loop=loop) -> None:
                _loop.call_soon_threadsafe(_q.put_nowait, (scanned, total))

            task = loop.run_in_executor(
                None, _process_video,
                vp, vn, ref_id, job_id, idx,
                threshold, sample_every, max_thumbs, det_conf, bool(use_gender),
                progress_cb,
            )

            while not task.done():
                try:
                    scanned, total = await asyncio.wait_for(progress_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                yield sse("video_progress", {
                    "index": idx, "name": vn,
                    "framesScanned": scanned, "totalFrames": total,
                })

            while not progress_q.empty():
                scanned, total = progress_q.get_nowait()
                yield sse("video_progress", {
                    "index": idx, "name": vn,
                    "framesScanned": scanned, "totalFrames": total,
                })

            try:
                result = task.result()
            except Exception as exc:
                yield sse("video_error", {"index": idx, "name": vn, "message": str(exc)})
                continue
            all_results.append(result)
            yield sse("video_done", result)

        yield sse("done", {
            "jobId":         job_id,
            "characterName": character_name.strip(),
            "totalMatches":  sum(r["matchCount"] for r in all_results),
            "videos":        all_results,
        })

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ─── Non-streaming fallback ───────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    character_name: Annotated[str,   Form()],
    reference:      Annotated[UploadFile, File()],
    videos:         Annotated[list[UploadFile], File()],
    threshold:      Annotated[float, Form()] = 0.50,
    sample_every:   Annotated[int,   Form()] = 2,
    max_thumbs:     Annotated[int,   Form()] = 30,
    det_conf:       Annotated[float, Form()] = 0.70,
    use_gender:     Annotated[int,   Form()] = 1,
) -> dict:
    if not character_name.strip():
        raise HTTPException(400, "Enter a character name.")
    if not videos:
        raise HTTPException(400, "Upload at least one video.")
    threshold, sample_every, det_conf = _clamp_params(threshold, sample_every, det_conf)

    job_id = uuid.uuid4().hex
    ref_path, video_paths = _save_job_uploads(job_id, reference, videos)
    ref_id = build_reference(ref_path)

    results = [
        _process_video(vp, vn, ref_id, job_id, i,
                        threshold, sample_every, max_thumbs, det_conf, bool(use_gender))
        for i, (vp, vn) in enumerate(video_paths)
    ]
    return {
        "jobId": job_id, "characterName": character_name.strip(),
        "threshold": threshold, "sampleEvery": sample_every,
        "videos": results, "totalMatches": sum(r["matchCount"] for r in results),
    }


# ─── Core video processing ────────────────────────────────────────────────────

def _process_video(
    video_path:   Path,
    video_name:   str,
    ref_id:       RefIdentity,
    job_id:       str,
    vid_idx:      int,
    threshold:    float = 0.50,
    sample_every: int   = 2,
    max_thumbs:   int   = 30,
    det_conf:     float = 0.70,
    use_gender:   bool  = True,
    progress_cb:  "Callable[[int, int], None] | None" = None,
) -> dict:
    if face_app is None:
        raise RuntimeError("InsightFace model not loaded.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_name}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Detect letterbox/pillarbox content area, sampled across the whole video
    content = _content_rect(cap, width, height)

    out_name = f"{job_id}_{vid_idx}_out.mp4"
    writer   = cv2.VideoWriter(
        str(OUTPUT_DIR / out_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    detections:    list[dict] = []
    frames_scanned = 0
    frame_idx      = 0
    last_det_sec   = -999.0
    carry: list[tuple[np.ndarray, float]] = []
    last_progress_emit = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current: list[tuple[np.ndarray, float]] = []

            if frame_idx % sample_every == 0:
                frames_scanned += 1
                carry = []

                try:
                    raw_faces = face_app.get(frame)
                except Exception:
                    raw_faces = []

                for face in raw_faces:
                    # Gate 1: detection confidence
                    if float(face.det_score) < det_conf:
                        continue

                    # Gate 1b: minimum face size — tiny faces (far/background
                    # people) upscale to almost nothing at the recognizer's
                    # 112x112 input and produce unreliable embeddings that can
                    # spuriously score high against any reference.
                    bx1, by1, bx2, by2 = face.bbox
                    if min(bx2 - bx1, by2 - by1) < MIN_FACE_PX:
                        continue

                    # Gate 2: gender filter
                    # InsightFace sets face['gender'] = 0 (F) or 1 (M)
                    if use_gender and ref_id.gender_int is not None:
                        face_gender = face.get("gender")
                        if face_gender is not None and int(face_gender) != ref_id.gender_int:
                            continue   # definitively wrong gender → skip

                    # Gate 3: ArcFace cosine similarity
                    emb   = normalize(face.embedding.astype(np.float32))
                    score = float(np.dot(ref_id.embedding, emb))
                    if score >= threshold:
                        current.append((face.bbox.astype(int), score))

                carry = current

                cur_sec = frame_idx / fps
                if current and cur_sec - last_det_sec >= 0.5:
                    best_score = max(s for _, s in current)
                    best_box   = max(current, key=lambda x: x[1])[0]
                    thumb      = _tight_crop(frame, best_box, content)
                    if thumb:
                        detections.append({
                            "time":      ts(cur_sec),
                            "seconds":   round(cur_sec, 2),
                            "score":     round(best_score, 3),
                            "thumbnail": thumb,
                        })
                        last_det_sec = cur_sec
            else:
                current = carry   # carry boxes to non-scanned frames

            for (x1, y1, x2, y2), score in current:
                c = (45, 214, 159)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
                lbl = f"{score:.0%}"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(frame, (x1, max(0, y1-lh-12)), (x1+lw+8, y1), c, -1)
                cv2.putText(frame, lbl, (x1+4, max(lh, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (16, 32, 30), 2)

            writer.write(frame)
            frame_idx += 1

            if progress_cb and total_frames > 0:
                now = time.monotonic()
                if now - last_progress_emit >= 0.5:
                    progress_cb(frame_idx, total_frames)
                    last_progress_emit = now
    finally:
        cap.release()
        writer.release()
        if progress_cb and total_frames > 0:
            progress_cb(total_frames, total_frames)

    clean = [{k: v for k, v in d.items() if k != "thumbnail"} for d in detections]
    return {
        "name":          video_name,
        "duration":      round(frame_idx / fps, 2),
        "fps":           round(fps, 2),
        "totalFrames":   frame_idx,
        "framesScanned": frames_scanned,
        "detections":    clean,
        "thumbnails":    [{"seconds": d["seconds"], "time": d["time"],
                           "score": d["score"], "thumbnail": d["thumbnail"]}
                          for d in detections],
        "matchCount":    len(detections),
        "videoUrl":      f"/media/outputs/{out_name}",
    }


def save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
