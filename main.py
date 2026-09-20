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

# Distinct BGR colors cycled per character so multiple people are visually
# distinguishable in the annotated video and in "who else is in this frame"
# best-match thumbnails.
CHARACTER_COLORS = [
    (45, 214, 159),   # mint
    (95, 115, 239),   # rust-blue
    (236, 156, 93),   # amber
    (60, 200, 230),   # yellow-cyan
    (220, 100, 200),  # magenta
    (140, 214, 45),   # lime
]


@dataclass
class RefIdentity:
    name: str
    embedding: np.ndarray
    gender_int: int | None   # 0=F, 1=M, None=unknown


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def ts(s: float) -> str:
    m, r = divmod(s, 60)
    return f"{int(m):02d}:{r:05.2f}"


def build_reference(path: Path, name: str) -> RefIdentity:
    if face_app is None:
        raise HTTPException(503, "InsightFace not installed.")
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, f"Reference photo for '{name}' could not be read.")
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(400, f"No face detected in the reference photo for '{name}'. Use a clear, well-lit, front-facing image.")
    best = max(faces, key=lambda f: f.det_score)
    emb  = normalize(best.embedding.astype(np.float32))

    # InsightFace buffalo_l stores gender as face['gender']: 0=F, 1=M
    raw_gender = best.get("gender")   # uses __getitem__ on Face object
    gender_int = int(raw_gender) if raw_gender is not None and int(raw_gender) in (0, 1) else None

    return RefIdentity(name=name, embedding=emb, gender_int=gender_int)


# ─── Full-frame thumbnail (whole body/scene, not just the face) ─────────────

def _frame_to_png_b64(frame: np.ndarray) -> str:
    """
    Encode a full video frame losslessly (PNG — compression level only trades
    encode speed for file size, never image quality). Used for the single
    best-match thumbnail: the whole frame is kept (not cropped to the face)
    so the person's whole body/context is visible, not just a face patch.
    """
    ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


# ─── Shared request handling ──────────────────────────────────────────────────

def _clamp_params(threshold: float, sample_every: int, det_conf: float) -> tuple[float, int, float]:
    threshold    = max(0.25, min(float(threshold),    0.90))
    sample_every = max(1,    min(int(sample_every),    30))
    det_conf     = max(0.30, min(float(det_conf),     0.95))
    return threshold, sample_every, det_conf


def _save_job_uploads(
    job_id: str, references: list[UploadFile], character_names: list[str], videos: list[UploadFile],
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    ref_entries: list[tuple[Path, str]] = []
    for i, (ref, name) in enumerate(zip(references, character_names)):
        suffix = Path(ref.filename or ".jpg").suffix or ".jpg"
        rp = UPLOAD_DIR / f"{job_id}_ref{i}{suffix}"
        save_upload(ref, rp)
        ref_entries.append((rp, name.strip()))

    video_paths: list[tuple[Path, str]] = []
    for i, vid in enumerate(videos):
        suffix = Path(vid.filename or ".mp4").suffix or ".mp4"
        vp = UPLOAD_DIR / f"{job_id}_{i}{suffix}"
        save_upload(vid, vp)
        video_paths.append((vp, vid.filename or f"video-{i+1}"))

    return ref_entries, video_paths


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
    character_names: Annotated[list[str], Form()],
    references:      Annotated[list[UploadFile], File()],
    videos:          Annotated[list[UploadFile], File()],
    threshold:       Annotated[float, Form()] = 0.50,
    sample_every:    Annotated[int,   Form()] = 2,
    max_thumbs:      Annotated[int,   Form()] = 30,
    det_conf:        Annotated[float, Form()] = 0.70,
    use_gender:      Annotated[int,   Form()] = 1,
) -> StreamingResponse:
    if len(character_names) != len(references):
        raise HTTPException(400, "Each character needs exactly one reference photo.")
    if not any(n.strip() for n in character_names):
        raise HTTPException(400, "Enter at least one character name.")
    if any(not n.strip() for n in character_names):
        raise HTTPException(400, "Every character row needs a name.")
    if not videos:
        raise HTTPException(400, "Upload at least one video.")
    threshold, sample_every, det_conf = _clamp_params(threshold, sample_every, det_conf)

    job_id = uuid.uuid4().hex
    ref_entries, video_paths = _save_job_uploads(job_id, references, character_names, videos)

    async def event_gen() -> AsyncGenerator[str, None]:
        def sse(event: str, payload: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

        names = [n for _, n in ref_entries]
        yield sse("start", {"jobId": job_id, "characterNames": names,
                             "videoCount": len(video_paths)})
        try:
            ref_ids = await asyncio.get_event_loop().run_in_executor(
                None, lambda: [build_reference(p, n) for p, n in ref_entries])
        except HTTPException as exc:
            yield sse("error", {"message": exc.detail})
            return

        yield sse("ref_ready", {"refs": [
            {"name": r.name, "gender": GENDER_INT_MAP.get(r.gender_int, "unknown"),
             "genderFilter": bool(use_gender) and r.gender_int is not None}
            for r in ref_ids
        ]})

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
                vp, vn, ref_ids, job_id, idx,
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
            "jobId":          job_id,
            "characterNames": names,
            "totalMatches":   sum(r["matchCount"] for r in all_results),
            "videos":         all_results,
        })

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ─── Non-streaming fallback ───────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    character_names: Annotated[list[str], Form()],
    references:      Annotated[list[UploadFile], File()],
    videos:          Annotated[list[UploadFile], File()],
    threshold:       Annotated[float, Form()] = 0.50,
    sample_every:    Annotated[int,   Form()] = 2,
    max_thumbs:      Annotated[int,   Form()] = 30,
    det_conf:        Annotated[float, Form()] = 0.70,
    use_gender:      Annotated[int,   Form()] = 1,
) -> dict:
    if len(character_names) != len(references):
        raise HTTPException(400, "Each character needs exactly one reference photo.")
    if not any(n.strip() for n in character_names):
        raise HTTPException(400, "Enter at least one character name.")
    if any(not n.strip() for n in character_names):
        raise HTTPException(400, "Every character row needs a name.")
    if not videos:
        raise HTTPException(400, "Upload at least one video.")
    threshold, sample_every, det_conf = _clamp_params(threshold, sample_every, det_conf)

    job_id = uuid.uuid4().hex
    ref_entries, video_paths = _save_job_uploads(job_id, references, character_names, videos)
    ref_ids = [build_reference(p, n) for p, n in ref_entries]

    results = [
        _process_video(vp, vn, ref_ids, job_id, i,
                        threshold, sample_every, max_thumbs, det_conf, bool(use_gender))
        for i, (vp, vn) in enumerate(video_paths)
    ]
    return {
        "jobId": job_id, "characterNames": [n for _, n in ref_entries],
        "threshold": threshold, "sampleEvery": sample_every,
        "videos": results, "totalMatches": sum(r["matchCount"] for r in results),
    }


# ─── Core video processing ────────────────────────────────────────────────────

def _process_video(
    video_path:   Path,
    video_name:   str,
    ref_ids:      list[RefIdentity],
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

    out_name = f"{job_id}_{vid_idx}_out.mp4"
    writer   = cv2.VideoWriter(
        str(OUTPUT_DIR / out_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    n_chars        = len(ref_ids)
    detections:    list[list[dict]] = [[] for _ in range(n_chars)]
    last_det_sec:  list[float]      = [-999.0] * n_chars
    carry:         list[list[tuple[np.ndarray, float]]] = [[] for _ in range(n_chars)]
    frames_scanned = 0
    frame_idx      = 0
    last_progress_emit = time.monotonic()

    # Track only the single highest-confidence match per character across the
    # whole video — not a crop of the face, the full frame (whole body/scene)
    # at that moment, so each character's thumbnail can't be a noisy face-only
    # patch and there's exactly one per character.
    best_overall_score:   list[float] = [-1.0] * n_chars
    best_overall_seconds: list[float] = [0.0] * n_chars
    best_overall_frame:   list["np.ndarray | None"] = [None] * n_chars

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current: list[list[tuple[np.ndarray, float]]] = [[] for _ in range(n_chars)]
            sampled = frame_idx % sample_every == 0

            if sampled:
                frames_scanned += 1

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

                    # Detection/embedding runs once per face regardless of how
                    # many characters we're searching for — only the (cheap)
                    # cosine-similarity comparison repeats per character.
                    face_gender = face.get("gender")   # InsightFace: 0=F, 1=M
                    emb = normalize(face.embedding.astype(np.float32))
                    box = face.bbox.astype(int)

                    for ci, ref in enumerate(ref_ids):
                        # Gate 2: gender filter
                        if use_gender and ref.gender_int is not None:
                            if face_gender is not None and int(face_gender) != ref.gender_int:
                                continue   # definitively wrong gender → skip

                        # Gate 3: ArcFace cosine similarity
                        score = float(np.dot(ref.embedding, emb))
                        if score >= threshold:
                            current[ci].append((box, score))

                carry = current

                cur_sec = frame_idx / fps
                for ci in range(n_chars):
                    if current[ci] and cur_sec - last_det_sec[ci] >= 0.5:
                        best_score = max(s for _, s in current[ci])
                        detections[ci].append({
                            "time":    ts(cur_sec),
                            "seconds": round(cur_sec, 2),
                            "score":   round(best_score, 3),
                        })
                        last_det_sec[ci] = cur_sec
            else:
                current = carry   # carry boxes to non-scanned frames

            for ci in range(n_chars):
                color = CHARACTER_COLORS[ci % len(CHARACTER_COLORS)]
                name  = ref_ids[ci].name
                for (x1, y1, x2, y2), score in current[ci]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    lbl = f"{name} {score:.0%}"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, max(0, y1-lh-12)), (x1+lw+8, y1), color, -1)
                    cv2.putText(frame, lbl, (x1+4, max(lh, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 32, 30), 2)

            # Keep only the single highest-confidence match per character, as the
            # full annotated frame (whole body/scene) — never a face-only crop,
            # never a gallery of every hit. Restricted to genuinely-detected
            # (non-carried) frames so a stale carried box can't win on a technicality.
            # Snapshot is taken after ALL characters' boxes are drawn, so if two
            # characters both matched in the same frame, each one's "best" thumbnail
            # will show the other's box too — that's intentional context, not a bug.
            if sampled:
                for ci in range(n_chars):
                    if current[ci]:
                        frame_best_score = max(s for _, s in current[ci])
                        if frame_best_score > best_overall_score[ci]:
                            best_overall_score[ci]   = frame_best_score
                            best_overall_seconds[ci] = frame_idx / fps
                            best_overall_frame[ci]   = frame.copy()

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

    characters: list[dict] = []
    for ci, ref in enumerate(ref_ids):
        thumbs: list[dict] = []
        if best_overall_frame[ci] is not None:
            thumb = _frame_to_png_b64(best_overall_frame[ci])
            if thumb:
                thumbs.append({
                    "seconds":   round(best_overall_seconds[ci], 2),
                    "time":      ts(best_overall_seconds[ci]),
                    "score":     round(best_overall_score[ci], 3),
                    "thumbnail": thumb,
                })
        characters.append({
            "name":        ref.name,
            "detections":  detections[ci],
            "thumbnails":  thumbs,
            "matchCount":  len(detections[ci]),
        })

    return {
        "name":          video_name,
        "duration":      round(frame_idx / fps, 2),
        "fps":           round(fps, 2),
        "totalFrames":   frame_idx,
        "framesScanned": frames_scanned,
        "characters":    characters,
        "matchCount":    sum(c["matchCount"] for c in characters),
        "videoUrl":      f"/media/outputs/{out_name}",
    }


def save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
