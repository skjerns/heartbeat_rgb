// roi.js — turn FaceLandmarker landmarks into skin regions of interest and
// sample their mean RGB from a downscaled copy of the current video frame.
//
// We sample the forehead and both cheeks (the strongest, most stable rPPG skin
// regions, away from eyes/brows/mouth/hair). All landmark coordinates are
// normalized (0..1) in the raw, un-mirrored video frame — which is exactly the
// frame we draw to the sampling canvas, so no mirroring is needed here.

// MediaPipe FaceMesh landmark indices used as anchors.
const LM = {
  faceLeft: 234,   // right edge of face in image space
  faceRight: 454,  // left edge of face in image space
  foreheadTop: 10, // hairline center
  glabella: 151,   // forehead center, just above brows
  browL: 105,      // left brow top
  browR: 334,      // right brow top
  cheekL: 50,      // left cheek
  cheekR: 280,     // right cheek
};

// A small offscreen canvas we draw the (downscaled) video into for sampling.
let sampleCanvas = null;
let sampleCtx = null;
const SAMPLE_W = 256; // width of sampling canvas; height follows video aspect

function ensureCanvas(videoW, videoH) {
  const h = Math.max(1, Math.round((SAMPLE_W * videoH) / videoW));
  if (!sampleCanvas) {
    sampleCanvas = document.createElement('canvas');
    sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (sampleCanvas.width !== SAMPLE_W || sampleCanvas.height !== h) {
    sampleCanvas.width = SAMPLE_W;
    sampleCanvas.height = h;
  }
  return { w: SAMPLE_W, h };
}

// Draw the current video frame into the sampling canvas. Call once per frame
// before sampling any faces.
export function grabFrame(video) {
  const { w, h } = ensureCanvas(video.videoWidth, video.videoHeight);
  sampleCtx.drawImage(video, 0, 0, w, h);
}

// Build the list of ROI rectangles (in normalized coords) for one face.
export function roiRects(landmarks) {
  const lx = landmarks[LM.faceLeft].x;
  const rx = landmarks[LM.faceRight].x;
  const faceW = Math.abs(rx - lx);
  if (!(faceW > 0)) return [];

  const cheekSize = faceW * 0.16;

  // Forehead: centered on glabella x, spanning from above the brows up toward
  // the hairline. Clamp the height so we stay on skin.
  const browY = Math.min(landmarks[LM.browL].y, landmarks[LM.browR].y);
  const topY = landmarks[LM.foreheadTop].y;
  const foreTop = topY + (browY - topY) * 0.15;
  const foreBottom = browY - (browY - topY) * 0.15;
  const foreCx = landmarks[LM.glabella].x;
  const foreW = faceW * 0.45;
  const forehead = {
    x: foreCx - foreW / 2,
    y: Math.min(foreTop, foreBottom),
    w: foreW,
    h: Math.abs(foreBottom - foreTop),
  };

  const cheek = (idx) => ({
    x: landmarks[idx].x - cheekSize / 2,
    y: landmarks[idx].y - cheekSize / 2,
    w: cheekSize,
    h: cheekSize,
  });

  return [forehead, cheek(LM.cheekL), cheek(LM.cheekR)];
}

// Skin test in YCbCr (Hsu et al.) — robust to illumination and excludes
// eyebrows, lips, background, and shadows that would otherwise add noise.
function isSkin(r, g, b) {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  return cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173 && y > 40 && y < 250;
}

const MIN_SKIN_PIXELS = 40; // require enough skin to trust the sample

// Sample mean RGB over the given face's ROIs from the current sampling canvas,
// averaging only skin-colored pixels. Returns { r, g, b } in 0..255, or null if
// the regions are degenerate / off frame / lack enough visible skin.
export function sampleFace(landmarks) {
  if (!sampleCanvas) return null;
  const rects = roiRects(landmarks);
  if (!rects.length) return null;

  const W = sampleCanvas.width;
  const H = sampleCanvas.height;
  let sumR = 0, sumG = 0, sumB = 0, count = 0;

  for (const rect of rects) {
    const x0 = Math.round(rect.x * W);
    const y0 = Math.round(rect.y * H);
    const w = Math.round(rect.w * W);
    const h = Math.round(rect.h * H);
    // Clip to canvas bounds; skip if it lands outside the frame.
    const cx = Math.max(0, x0);
    const cy = Math.max(0, y0);
    const cw = Math.min(W, x0 + w) - cx;
    const ch = Math.min(H, y0 + h) - cy;
    if (cw <= 0 || ch <= 0) continue;

    const data = sampleCtx.getImageData(cx, cy, cw, ch).data;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      if (!isSkin(r, g, b)) continue;
      sumR += r;
      sumG += g;
      sumB += b;
      count++;
    }
  }

  if (count < MIN_SKIN_PIXELS) return null;
  return { r: sumR / count, g: sumG / count, b: sumB / count };
}
