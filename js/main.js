// main.js — orchestration: camera, detection loop, per-face overlay + waveform.

import { createDetector, FaceTracker } from './face.js';
import { grabFrame, sampleFace, roiRects } from './roi.js';
import { ColorMagnifier } from './magnify.js';
import { runSelfTest } from './selftest.js';

const UPDATE_EVERY = 15; // recompute BPM every N frames

const els = {
  start: document.getElementById('startBtn'),
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  wave: document.getElementById('wave'),
  status: document.getElementById('status'),
  faces: document.getElementById('faces'),
  stage: document.getElementById('stage'),
  magnified: document.getElementById('magnified'),
  ampRange: document.getElementById('ampRange'),
};

const tracker = new FaceTracker();
const magnifier = new ColorMagnifier();
if (els.ampRange) {
  magnifier.alpha = Number(els.ampRange.value);
  els.ampRange.addEventListener('input', () => {
    magnifier.alpha = Number(els.ampRange.value);
  });
}
let detector = null;
let frameIdx = 0;
let running = false;

function setStatus(msg, kind = 'info') {
  els.status.textContent = msg;
  els.status.dataset.kind = kind;
}

async function start() {
  els.start.disabled = true;
  setStatus('Requesting camera…');

  if (!window.isSecureContext) {
    setStatus('Camera needs HTTPS or localhost (a secure context).', 'error');
    els.start.disabled = false;
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
        facingMode: 'user',
      },
      audio: false,
    });
  } catch (err) {
    setStatus(`Camera unavailable: ${err.name || err.message}. Grant permission and retry.`, 'error');
    els.start.disabled = false;
    return;
  }

  els.video.srcObject = stream;
  await els.video.play();
  await lockCameraAuto(stream);

  setStatus('Loading face model…');
  try {
    detector = await createDetector();
  } catch (err) {
    setStatus(`Failed to load face model: ${err.message}`, 'error');
    return;
  }

  els.stage.classList.add('live');
  sizeCanvases();
  setStatus('Hold still and well-lit. Estimating… (≈10 s warm-up)');
  running = true;
  requestAnimationFrame(loop);
}

// Auto-exposure and auto-white-balance continuously "correct" the very
// brightness/color changes rPPG measures, fighting the signal. Lock them to
// manual where the browser/camera supports it. Best-effort: ignore failures.
async function lockCameraAuto(stream) {
  const track = stream.getVideoTracks()[0];
  if (!track || !track.getCapabilities) return;
  let caps = {};
  try {
    caps = track.getCapabilities();
  } catch {
    return;
  }
  const advanced = [];
  if (caps.exposureMode && caps.exposureMode.includes('manual')) {
    advanced.push({ exposureMode: 'manual' });
  }
  if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes('manual')) {
    advanced.push({ whiteBalanceMode: 'manual' });
  }
  if (caps.focusMode && caps.focusMode.includes('manual')) {
    advanced.push({ focusMode: 'manual' });
  }
  if (!advanced.length) return;
  try {
    await track.applyConstraints({ advanced });
    console.log('Locked camera auto-controls:', advanced);
  } catch (err) {
    console.warn('Could not lock camera auto-controls:', err.message);
  }
}

function sizeCanvases() {
  const w = els.video.videoWidth;
  const h = els.video.videoHeight;
  els.overlay.width = w;
  els.overlay.height = h;
  els.magnified.width = w;
  els.magnified.height = h;
  els.wave.width = els.wave.clientWidth || 480;
  els.wave.height = els.wave.clientHeight || 120;
}

function loop() {
  if (!running) return;
  if (els.video.readyState >= 2) {
    if (els.overlay.width !== els.video.videoWidth) sizeCanvases();
    frameIdx++;
    const nowMs = performance.now();
    const result = detector.detectForVideo(els.video, nowMs);
    const detections = result.faceLandmarks || [];
    const active = tracker.update(detections, frameIdx);

    grabFrame(els.video);
    const tSec = nowMs / 1000;
    for (const track of active) {
      const rgb = sampleFace(track.landmarks);
      if (rgb) track.estimator.push(tSec, rgb);
      if (frameIdx % UPDATE_EVERY === 0) track.estimator.update();
    }

    drawOverlay(active);
    magnifier.process(els.video, active, els.magnified);
    drawWave(active);
    updatePanel(active);
  }
  requestAnimationFrame(loop);
}

// The displayed video is mirrored (selfie view); flip x so overlays line up,
// while keeping text un-mirrored.
function flipX(x, w) {
  return w - x * w;
}

function drawOverlay(tracks) {
  const ctx = els.overlay.getContext('2d');
  const W = els.overlay.width;
  const H = els.overlay.height;
  ctx.clearRect(0, 0, W, H);

  for (const t of tracks) {
    const b = t.box;
    // Mirror the box horizontally.
    const x = flipX(b.maxX, W);
    const y = b.minY * H;
    const w = (b.maxX - b.minX) * W;
    const h = (b.maxY - b.minY) * H;

    ctx.lineWidth = 3;
    ctx.strokeStyle = t.color;
    ctx.strokeRect(x, y, w, h);

    // Draw the ROI rectangles (forehead + cheeks) faintly.
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = t.color + '99';
    for (const r of roiRects(t.landmarks)) {
      const rx = flipX(r.x + r.w, W);
      ctx.strokeRect(rx, r.y * H, r.w * W, r.h * H);
    }

    // Label.
    const est = t.estimator;
    const label = est.bpm != null ? `${Math.round(est.bpm)} BPM` : '…';
    ctx.font = 'bold 22px system-ui, sans-serif';
    const tw = ctx.measureText(label).width;
    const pad = 6;
    const ly = Math.max(0, y - 30);
    ctx.fillStyle = t.color;
    ctx.fillRect(x, ly, tw + pad * 2, 28);
    ctx.fillStyle = '#06121a';
    ctx.fillText(label, x + pad, ly + 21);
  }
}

function drawWave(tracks) {
  const ctx = els.wave.getContext('2d');
  const W = els.wave.width;
  const H = els.wave.height;
  ctx.clearRect(0, 0, W, H);

  // Plot the most confident face's pulse waveform.
  let best = null;
  for (const t of tracks) {
    if (t.estimator.pulse.length && (!best || t.estimator.confidence > best.estimator.confidence)) {
      best = t;
    }
  }
  if (!best) return;

  const sig = best.estimator.pulse;
  let min = Infinity, max = -Infinity;
  for (const v of sig) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;

  ctx.beginPath();
  for (let i = 0; i < sig.length; i++) {
    const px = (i / (sig.length - 1)) * W;
    const py = H - ((sig[i] - min) / range) * (H - 8) - 4;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = best.color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function updatePanel(tracks) {
  if (!tracks.length) {
    els.faces.innerHTML = '<li class="empty">No face detected — center your face in view.</li>';
    return;
  }
  const sorted = [...tracks].sort((a, b) => a.id - b.id);
  els.faces.innerHTML = sorted
    .map((t) => {
      const est = t.estimator;
      const bpm = est.bpm != null ? `${Math.round(est.bpm)}` : '—';
      const conf = Math.round(Math.min(1, est.confidence * 6) * 100); // scaled for display
      return `<li>
        <span class="dot" style="background:${t.color}"></span>
        <span class="fid">Face ${t.id}</span>
        <span class="bpm">${bpm}<small>BPM</small></span>
        <span class="conf" title="signal quality">
          <span class="bar" style="width:${conf}%"></span>
        </span>
      </li>`;
    })
    .join('');
}

// ---- entry point ----
const params = new URLSearchParams(location.search);
if (params.get('selftest') === '1') {
  setStatus('Running DSP self-test — see console.');
  runSelfTest();
} else {
  els.start.addEventListener('click', start);
}
