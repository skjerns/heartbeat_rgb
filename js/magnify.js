// magnify.js — real-time Eulerian color magnification.
//
// Exaggerates the tiny per-frame color changes that carry the pulse, so you can
// *see* the heartbeat as a rhythmic flush on the skin. We use the classic
// real-time approximation of Eulerian Video Magnification (Wu et al., SIGGRAPH
// 2012): a temporal band-pass built from the difference of two IIR low-pass
// filters per pixel, amplified and added back to the frame. Magnification is
// applied only inside the bounding boxes of faces with a detected heart rate.

const PROC_W = 160; // processing width (downscaling also denoises spatially)

// IIR coefficients for a ~0.7–3 Hz band at ~30 fps: r = 1 - exp(-2π·fc/fs).
const R_HIGH = 0.47; // upper edge (~3 Hz)  -> faster low-pass
const R_LOW = 0.14;  // lower edge (~0.7 Hz) -> slower low-pass

function clamp8(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

export class ColorMagnifier {
  constructor() {
    this.proc = document.createElement('canvas');
    this.pctx = this.proc.getContext('2d', { willReadFrequently: true });
    this.lp1 = null; // faster low-pass state (per channel)
    this.lp2 = null; // slower low-pass state
    this.band = null; // band-passed signal
    this.mask = null;
    this.alpha = 50; // amplification factor (user-adjustable)
  }

  reset() {
    this.lp1 = null;
  }

  // Render the magnified view of `video` into `outCanvas`, exaggerating color
  // only inside the boxes of `tracks` that have a detected BPM.
  process(video, tracks, outCanvas) {
    if (!video.videoWidth) return;
    const w = PROC_W;
    const h = Math.max(1, Math.round((PROC_W * video.videoHeight) / video.videoWidth));
    if (this.proc.width !== w || this.proc.height !== h) {
      this.proc.width = w;
      this.proc.height = h;
      this.lp1 = null; // geometry changed -> reinitialize filter state
    }

    this.pctx.drawImage(video, 0, 0, w, h);
    const img = this.pctx.getImageData(0, 0, w, h);
    const d = img.data;
    const npx = w * h;

    if (!this.lp1) {
      this.lp1 = new Float32Array(npx * 3);
      this.lp2 = new Float32Array(npx * 3);
      this.band = new Float32Array(npx * 3);
      for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
        this.lp1[j] = this.lp2[j] = d[i];
        this.lp1[j + 1] = this.lp2[j + 1] = d[i + 1];
        this.lp1[j + 2] = this.lp2[j + 2] = d[i + 2];
      }
    }

    const { lp1, lp2, band } = this;
    // Update both low-pass filters and form the band-pass = lp1 - lp2.
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
      for (let c = 0; c < 3; c++) {
        const v = d[i + c];
        lp1[j + c] += R_HIGH * (v - lp1[j + c]);
        lp2[j + c] += R_LOW * (v - lp2[j + c]);
        band[j + c] = lp1[j + c] - lp2[j + c];
      }
    }

    // Build a mask of pixels belonging to faces with a detected pulse.
    let mask = this.mask;
    if (!mask || mask.length !== npx) mask = this.mask = new Uint8Array(npx);
    mask.fill(0);
    for (const t of tracks) {
      if (t.estimator.bpm == null) continue;
      const b = t.box;
      const x0 = Math.max(0, Math.floor(b.minX * w));
      const x1 = Math.min(w, Math.ceil(b.maxX * w));
      const y0 = Math.max(0, Math.floor(b.minY * h));
      const y1 = Math.min(h, Math.ceil(b.maxY * h));
      for (let y = y0; y < y1; y++) {
        const row = y * w;
        for (let x = x0; x < x1; x++) mask[row + x] = 1;
      }
    }

    // Composite: amplified band-pass added back, masked to detected faces.
    const alpha = this.alpha;
    for (let p = 0, i = 0, j = 0; p < npx; p++, i += 4, j += 3) {
      if (mask[p]) {
        d[i] = clamp8(d[i] + alpha * band[j]);
        d[i + 1] = clamp8(d[i + 1] + alpha * band[j + 1]);
        d[i + 2] = clamp8(d[i + 2] + alpha * band[j + 2]);
      }
    }

    this.pctx.putImageData(img, 0, 0);
    const octx = outCanvas.getContext('2d');
    octx.imageSmoothingEnabled = true;
    octx.drawImage(this.proc, 0, 0, outCanvas.width, outCanvas.height);
  }
}
