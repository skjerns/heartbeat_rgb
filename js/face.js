// face.js — MediaPipe FaceLandmarker wrapper + a lightweight multi-face tracker.
//
// MediaPipe does not guarantee a stable ordering of faces across frames, so we
// match each detection to an existing track by nearest bounding-box centroid.
// Each track owns its own PulseEstimator, color, and bounding box.

import {
  FaceLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35';
import { PulseEstimator } from './rppg.js';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const MAX_FACES = 4;
const MATCH_GATE = 0.18; // max normalized centroid distance to reuse a track
const DROP_AFTER = 15;   // frames a track may be unseen before removal

const COLORS = ['#36d399', '#3abff8', '#f87272', '#fbbd23', '#a78bfa', '#22d3ee'];

export async function createDetector() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: MAX_FACES,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

// Bounding box {minX,minY,maxX,maxY,cx,cy} from a landmark list (normalized).
function bbox(landmarks) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
}

let nextId = 1;

class Track {
  constructor(landmarks, box, frameIdx) {
    this.id = nextId++;
    this.color = COLORS[(this.id - 1) % COLORS.length];
    this.estimator = new PulseEstimator();
    this.landmarks = landmarks;
    this.box = box;
    this.lastSeen = frameIdx;
  }
}

export class FaceTracker {
  constructor() {
    this.tracks = [];
  }

  // detections: array of landmark lists (result.faceLandmarks).
  // Returns the list of currently-active tracks.
  update(detections, frameIdx) {
    const dets = detections.map((lm) => ({ lm, box: bbox(lm) }));
    const usedTracks = new Set();
    const usedDets = new Set();

    // Greedy nearest-centroid matching between existing tracks and detections.
    const pairs = [];
    for (let ti = 0; ti < this.tracks.length; ti++) {
      for (let di = 0; di < dets.length; di++) {
        const dx = this.tracks[ti].box.cx - dets[di].box.cx;
        const dy = this.tracks[ti].box.cy - dets[di].box.cy;
        pairs.push({ ti, di, d: Math.hypot(dx, dy) });
      }
    }
    pairs.sort((a, b) => a.d - b.d);
    for (const p of pairs) {
      if (p.d > MATCH_GATE) break;
      if (usedTracks.has(p.ti) || usedDets.has(p.di)) continue;
      const track = this.tracks[p.ti];
      const det = dets[p.di];
      track.landmarks = det.lm;
      track.box = det.box;
      track.lastSeen = frameIdx;
      usedTracks.add(p.ti);
      usedDets.add(p.di);
    }

    // Unmatched detections become new tracks.
    for (let di = 0; di < dets.length; di++) {
      if (usedDets.has(di)) continue;
      this.tracks.push(new Track(dets[di].lm, dets[di].box, frameIdx));
    }

    // Drop stale tracks.
    this.tracks = this.tracks.filter((t) => frameIdx - t.lastSeen <= DROP_AFTER);

    // Only return tracks seen this frame (have current landmarks).
    return this.tracks.filter((t) => t.lastSeen === frameIdx);
  }
}
