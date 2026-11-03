const dom = {
  livePill: document.getElementById("livePill"),
  liveDot: document.getElementById("liveDot"),
  liveText: document.getElementById("liveText"),
  cameraDot: document.getElementById("cameraDot"),
  cameraLabel: document.getElementById("cameraLabel"),
  latencyLabel: document.getElementById("latencyLabel"),
  detectionLog: document.getElementById("detectionLog"),
  itemCount: document.getElementById("itemCount"),
  inferenceValue: document.getElementById("inferenceValue"),
  engineLabel: document.getElementById("engineLabel"),
  offlineOverlay: document.getElementById("offlineOverlay"),
  statusCameraDot: document.getElementById("statusCameraDot"),
  cameraStatus: document.getElementById("cameraStatus"),
  frameRateLabel: document.getElementById("frameRateLabel"),
  ramLabel: document.getElementById("ramLabel"),
  syncDot: document.getElementById("syncDot"),
  syncLabel: document.getElementById("syncLabel"),
};

const POLL_MS = 1000;
let lastFrameId = -1;
let lastFrameChangeAt = Date.now();

function titleCase(value) {
  return String(value || "object").replace(/\b\w/g, (character) => character.toUpperCase());
}

function setDot(element, mode) {
  element.classList.remove("bad", "neutral");
  if (mode === "bad") {
    element.classList.add("bad");
  } else if (mode === "neutral") {
    element.classList.add("neutral");
  }
}

function renderDetections(data) {
  const detections = Array.isArray(data.detections) ? data.detections : [];
  const counts = data.counts && typeof data.counts === "object" ? data.counts : {};

  dom.itemCount.textContent = `${detections.length} ${detections.length === 1 ? "ITEM" : "ITEMS"}`;
  dom.detectionLog.replaceChildren();

  if (detections.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-log";
    empty.textContent = data.camera_ok ? "No objects detected" : "Waiting for camera stream...";
    dom.detectionLog.appendChild(empty);
    return;
  }

  detections.slice(0, 30).forEach((detection) => {
    const label = titleCase(detection.label);
    const count = Number(counts[detection.label]) || 1;

    const item = document.createElement("div");
    item.className = "log-item";

    const top = document.createElement("div");
    top.className = "log-top";

    const name = document.createElement("span");
    name.className = "log-name";
    name.textContent = `${label} (${count})`;

    const score = document.createElement("span");
    score.className = "score";
    score.textContent = Number(detection.confidence || 0).toFixed(2);

    top.append(name, score);

    const id = document.createElement("div");
    id.className = "log-id";
    id.textContent = `ID: ${detection.id || "obj_000"}`;

    item.append(top, id);
    dom.detectionLog.appendChild(item);
  });
}

function render(data) {
  const cameraOk = Boolean(data.camera_ok);
  const inferenceMs = Number(data.inference_ms) || 0;
  const cameraFps = Math.round(Number(data.camera_fps) || 0);

  dom.cameraLabel.textContent = `${data.camera_name || "DEV_CAM_01"} | ${data.camera_width || 0}x${data.camera_height || 0} @ ${cameraFps}fps`;
  dom.latencyLabel.textContent = `LATENCY: ${Math.round(inferenceMs)}ms`;
  dom.inferenceValue.textContent = `${inferenceMs.toFixed(1)}ms`;
  dom.engineLabel.textContent = data.engine || "CPU";
  dom.frameRateLabel.textContent = `Frame Rate: ${Math.round(Number(data.fps) || 0)} FPS`;
  dom.ramLabel.textContent = `RAM: ${Math.round(Number(data.memory_mb) || 0)} MB`;

  dom.liveText.textContent = cameraOk ? "LIVE" : "OFFLINE";
  dom.livePill.classList.toggle("offline", !cameraOk);
  setDot(dom.liveDot, cameraOk ? "ok" : "bad");
  setDot(dom.cameraDot, cameraOk ? "ok" : "bad");
  setDot(dom.statusCameraDot, cameraOk ? "ok" : "bad");

  dom.cameraStatus.textContent = cameraOk ? "Camera Active" : "Camera Offline";

  const frameId = Number(data.frame_id) || 0;
  if (frameId !== lastFrameId) {
    lastFrameId = frameId;
    lastFrameChangeAt = Date.now();
  }

  const warmingUp = cameraOk && frameId === 0;
  dom.offlineOverlay.textContent = warmingUp ? "Warming up engine..." : "Camera stream unavailable";
  dom.offlineOverlay.hidden = !warmingUp && cameraOk;

  const stalled = !cameraOk || Date.now() - lastFrameChangeAt > 3000;
  dom.syncLabel.textContent = stalled ? "Engine Sync: STALLED" : "Engine Sync: OK";
  dom.syncLabel.parentElement.classList.toggle("ok", !stalled);
  dom.syncLabel.parentElement.classList.toggle("bad", stalled);
  setDot(dom.syncDot, stalled ? "bad" : "ok");

  renderDetections(data);
}

function renderOffline() {
  dom.liveText.textContent = "OFFLINE";
  dom.livePill.classList.add("offline");
  setDot(dom.liveDot, "bad");
  setDot(dom.cameraDot, "bad");
  setDot(dom.statusCameraDot, "bad");
  setDot(dom.syncDot, "bad");
  dom.cameraStatus.textContent = "Camera Offline";
  dom.syncLabel.textContent = "Engine Sync: OFFLINE";
  dom.syncLabel.parentElement.classList.remove("ok");
  dom.syncLabel.parentElement.classList.add("bad");
  dom.offlineOverlay.textContent = "Camera stream unavailable";
  dom.offlineOverlay.hidden = false;
  dom.itemCount.textContent = "0 ITEMS";

  dom.detectionLog.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-log";
  empty.textContent = "API unavailable. Retrying...";
  dom.detectionLog.appendChild(empty);
}

async function refresh() {
  try {
    const response = await fetch("/api/stats", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`stats request failed: ${response.status}`);
    }
    render(await response.json());
  } catch (error) {
    renderOffline();
  }
}

refresh();
setInterval(refresh, POLL_MS);
