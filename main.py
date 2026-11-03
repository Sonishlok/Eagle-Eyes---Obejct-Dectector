import logging
import os
import shutil
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"
CAMERA_INDEX = 0
CAMERA_NAME = "DEV_CAM_01"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMGSZ = 640
CONF_THRESHOLD = 0.40
JPEG_QUALITY = 80
ENGINE_LABEL = "CPU"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
logger = logging.getLogger("eagle-eye")


def memory_mb():
    if os.name != "nt":
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:
            return 0.0
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return round(counters.WorkingSetSize / (1024 * 1024), 1)
    except Exception:
        return 0.0


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.latest_raw_frame = None
        self.raw_frame_id = 0
        self.latest_jpeg = None
        self.jpeg_frame_id = 0
        self.counts = {}
        self.detections = []
        self.total = 0
        self.fps = 0.0
        self.inference_ms = 0.0
        self.memory_mb = 0.0
        self.processed_at = None
        self.camera_ok = False
        self.camera_width = 0
        self.camera_height = 0
        self.camera_fps = 0.0

    def publish_raw(self, frame):
        with self.condition:
            self.raw_frame_id += 1
            self.latest_raw_frame = frame
            self.condition.notify_all()

    def wait_for_raw(self, last_frame_id, timeout):
        with self.condition:
            if self.raw_frame_id <= last_frame_id:
                self.condition.wait(timeout)
            if self.raw_frame_id > last_frame_id and self.latest_raw_frame is not None:
                return self.raw_frame_id, self.latest_raw_frame
            return None

    def publish_processed(self, jpeg, counts, detections, inference_ms, fps):
        used_memory = memory_mb()
        processed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self.condition:
            self.jpeg_frame_id += 1
            self.latest_jpeg = jpeg
            self.counts = counts
            self.detections = detections
            self.total = sum(counts.values())
            self.inference_ms = inference_ms
            self.fps = fps
            self.memory_mb = used_memory
            self.processed_at = processed_at
            self.condition.notify_all()

    def wait_for_jpeg(self, last_frame_id, timeout):
        with self.condition:
            if self.jpeg_frame_id <= last_frame_id:
                self.condition.wait(timeout)
            if self.jpeg_frame_id > last_frame_id and self.latest_jpeg is not None:
                return self.jpeg_frame_id, self.latest_jpeg
            return None

    def set_camera_ok(self, ok):
        with self.condition:
            self.camera_ok = ok
            self.condition.notify_all()

    def set_camera_info(self, width, height, fps):
        with self.condition:
            self.camera_width = width
            self.camera_height = height
            self.camera_fps = fps

    def is_camera_ok(self):
        with self.lock:
            return self.camera_ok

    def snapshot(self):
        with self.lock:
            return {
                "counts": dict(self.counts),
                "detections": list(self.detections),
                "total": self.total,
                "frame_id": self.jpeg_frame_id,
                "fps": round(self.fps, 1),
                "inference_ms": round(self.inference_ms, 1),
                "memory_mb": round(self.memory_mb, 1),
                "camera_ok": self.camera_ok,
                "camera_name": CAMERA_NAME,
                "camera_width": self.camera_width,
                "camera_height": self.camera_height,
                "camera_fps": round(self.camera_fps, 1),
                "engine": ENGINE_LABEL,
                "timestamp": self.processed_at,
            }


class CameraReader(threading.Thread):
    def __init__(self, state):
        super().__init__(name="CameraReader", daemon=True)
        self.state = state
        self.stop_event = threading.Event()
        self.cap = None
        self.read_times = deque(maxlen=30)

    def _open(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(CAMERA_INDEX, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()

        if self.cap is None:
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or FRAME_WIDTH)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or FRAME_HEIGHT)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.state.set_camera_info(width, height, fps)
        self.state.set_camera_ok(True)
        logger.info("Camera opened at %sx%s @ %.1f FPS", width, height, fps)
        return True

    def run(self):
        while not self.stop_event.is_set():
            if self.cap is None:
                if not self._open():
                    self.state.set_camera_ok(False)
                    self.stop_event.wait(2.0)
                    continue

            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                logger.warning("Camera read failed: %s", exc)
                ok, frame = False, None

            if not ok or frame is None:
                self.state.set_camera_ok(False)
                logger.warning("Camera disconnected")
                self.cap.release()
                self.cap = None
                self.stop_event.wait(1.0)
                continue

            self.state.set_camera_ok(True)
            self.state.publish_raw(frame)

            self.read_times.append(time.monotonic())
            if len(self.read_times) > 1:
                elapsed = self.read_times[-1] - self.read_times[0]
                if elapsed > 0:
                    measured_fps = (len(self.read_times) - 1) / elapsed
                    self.state.set_camera_info(frame.shape[1], frame.shape[0], measured_fps)

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.state.set_camera_ok(False)

    def stop(self):
        self.stop_event.set()


class DetectorWorker(threading.Thread):
    def __init__(self, state):
        super().__init__(name="DetectorWorker", daemon=True)
        self.state = state
        self.stop_event = threading.Event()
        self.model = None
        self.names = {}
        self.timestamps = deque(maxlen=30)

    def _load_model(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            self.model = YOLO(str(MODEL_PATH))
        else:
            self.model = YOLO("yolov8n.pt")
            downloaded = BASE_DIR / "yolov8n.pt"
            if downloaded.exists():
                try:
                    shutil.move(str(downloaded), str(MODEL_PATH))
                except Exception:
                    pass
        self.names = self.model.names
        self.model(
            np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8),
            imgsz=IMGSZ,
            verbose=False,
            device="cpu",
        )
        logger.info("YOLOv8n loaded and warmed up")

    def _class_name(self, class_id):
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))
        try:
            return str(self.names[class_id])
        except Exception:
            return str(class_id)

    def _render(self, frame, result):
        annotated = frame.copy()
        counts = {}
        detections = []
        boxes = result.boxes
        if boxes is None:
            return annotated, counts, detections

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for index, (box, class_id, confidence) in enumerate(zip(xyxy, classes, confidences), start=1):
            x1, y1, x2, y2 = [int(value) for value in box]
            class_id = int(class_id)
            name = self._class_name(class_id)
            confidence = float(confidence)
            counts[name] = counts.get(name, 0) + 1
            detections.append(
                {
                    "id": f"obj_{index:03d}",
                    "label": name,
                    "confidence": round(confidence, 2),
                }
            )

            color = (122, 226, 52)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name.capitalize()} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_top = max(y1 - text_height - 8, 0)
            cv2.rectangle(annotated, (x1, label_top), (x1 + text_width + 8, y1), (10, 10, 12), -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(y1 - 4, text_height + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return annotated, counts, detections

    def run(self):
        try:
            self._load_model()
        except Exception as exc:
            logger.exception("Detector failed to load: %s", exc)
            return

        last_raw_id = 0
        while not self.stop_event.is_set():
            item = self.state.wait_for_raw(last_raw_id, timeout=0.5)
            if item is None:
                continue

            raw_id, frame = item
            if raw_id <= last_raw_id:
                continue
            last_raw_id = raw_id

            try:
                started = time.perf_counter()
                results = self.model(
                    frame,
                    imgsz=IMGSZ,
                    conf=CONF_THRESHOLD,
                    verbose=False,
                    device="cpu",
                )
                inference_ms = (time.perf_counter() - started) * 1000.0

                annotated, counts, detections = self._render(frame, results[0])
                ok, buffer = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if not ok:
                    logger.warning("JPEG encoding failed")
                    continue

                self.timestamps.append(time.monotonic())
                fps = 0.0
                if len(self.timestamps) > 1:
                    elapsed = self.timestamps[-1] - self.timestamps[0]
                    if elapsed > 0:
                        fps = (len(self.timestamps) - 1) / elapsed

                self.state.publish_processed(
                    buffer.tobytes(),
                    counts,
                    detections,
                    inference_ms,
                    fps,
                )
            except Exception as exc:
                logger.exception("Detection frame failed: %s", exc)

    def stop(self):
        self.stop_event.set()


shared_state = None
camera_reader = None
detector_worker = None


@asynccontextmanager
async def lifespan(app):
    global shared_state, camera_reader, detector_worker
    shared_state = SharedState()
    camera_reader = CameraReader(shared_state)
    detector_worker = DetectorWorker(shared_state)
    camera_reader.start()
    detector_worker.start()
    logger.info("Eagle Eye workers started")
    try:
        yield
    finally:
        camera_reader.stop()
        detector_worker.stop()
        camera_reader.join(timeout=3.0)
        detector_worker.join(timeout=5.0)
        logger.info("Eagle Eye workers stopped")


app = FastAPI(title="Eagle Eye", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"), media_type="text/html")


@app.get("/video_feed")
def video_feed():
    if shared_state is None:
        return JSONResponse({"error": "camera unavailable"}, status_code=503)

    def generate():
        last_frame_id = 0
        while True:
            item = shared_state.wait_for_jpeg(last_frame_id, timeout=1.0)
            if item is None:
                continue
            frame_id, jpeg = item
            if frame_id <= last_frame_id or jpeg is None:
                continue
            last_frame_id = frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/stats")
def api_stats():
    if shared_state is None:
        return JSONResponse(
            {
                "counts": {},
                "detections": [],
                "total": 0,
                "frame_id": 0,
                "fps": 0,
                "inference_ms": 0,
                "memory_mb": 0,
                "camera_ok": False,
                "camera_name": CAMERA_NAME,
                "camera_width": 0,
                "camera_height": 0,
                "camera_fps": 0,
                "engine": ENGINE_LABEL,
                "timestamp": None,
            }
        )
    return JSONResponse(shared_state.snapshot())
