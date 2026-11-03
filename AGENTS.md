# AGENTS.md

## Project Overview
Eagle Eye is a one-day MVP web dashboard that streams webcam frames annotated with YOLOv8n object detections via FastAPI and OpenCV.

## Architecture
```
eagle-eye/
├── main.py              # FastAPI app, lifespan startup/shutdown, routes, static mount
├── templates/
│   └── index.html       # Dark-mode dashboard: <img src="/video_feed">, stats panel
├── static/
│   ├── style.css        # Dark-mode theme
│   └── app.js           # Polls /api/stats, updates counters
├── models/
│   └── yolov8n.pt       # Pre-downloaded YOLOv8n weights
└── requirements.txt     # Pinned dependencies (pip freeze)
```
- `main.py` owns routes (`/`, `/video_feed`, `/api/stats`), startup/shutdown, and static mounting.
- Templates render the dashboard; static assets hold CSS/JS.
- Capture, inference, and HTTP are decoupled: one camera reader thread, one detector thread, and clients only read the latest JPEG.

## Critical Rules
- Must use `cv2.CAP_PROP_BUFFERSIZE = 1` when configuring the camera.
- `/video_feed` must stream MJPEG (`multipart/x-mixed-replace; boundary=frame`) without blocking the event loop.
  - Use a sync generator so Starlette runs it in its thread pool.
  - Never run inference, `imencode`, or `yield` while holding the state lock.
  - Drop stale frames: always serve the latest processed JPEG, never a growing queue.
  - Never start per-request camera reads or inference; use shared reader/detector threads per process.

## Commands
```bash
# Install
pip install -r requirements.txt

# Run
uvicorn main:app --reload
```
