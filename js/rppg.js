// rppg.js — per-face heart-rate estimator using the POS algorithm.
//
// POS ("Plane-Orthogonal-to-Skin"), Wang, den Brinker, Stuijk & de Haan,
// "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
//
// Each PulseEstimator owns a sliding window of mean-RGB samples (one per video
// frame) plus their timestamps, runs POS over the window to recover the blood-
// volume pulse, and derives a smoothed BPM with a confidence value.

import { mean, std, estimateBpm } from './dsp.js';

const WINDOW_SECONDS = 10;   // length of analysis window
const MIN_SECONDS = 6;       // need at least this much data before reporting
const MAX_SAMPLES = 600;     // hard cap on buffer length (safety)
const BPM_EMA_ALPHA = 0.25;  // smoothing of reported BPM
const MIN_CONFIDENCE = 0.06; // gate: ignore low-SNR estimates

export class PulseEstimator {
  constructor() {
    this.t = [];   // timestamps (seconds)
    this.r = [];
    this.g = [];
    this.b = [];
    this.bpm = null;        // smoothed, reported value
    this.rawBpm = null;     // latest raw estimate
    this.confidence = 0;
    this.pulse = new Float64Array(0); // latest POS waveform (for plotting)
  }

  // Add one mean-RGB sample taken at time `tSec` (seconds, monotonic).
  push(tSec, rgb) {
    this.t.push(tSec);
    this.r.push(rgb.r);
    this.g.push(rgb.g);
    this.b.push(rgb.b);

    // Drop samples older than the window.
    const cutoff = tSec - WINDOW_SECONDS;
    while (this.t.length > 1 && this.t[0] < cutoff) {
      this.t.shift();
      this.r.shift();
      this.g.shift();
      this.b.shift();
    }
    if (this.t.length > MAX_SAMPLES) {
      const drop = this.t.length - MAX_SAMPLES;
      this.t.splice(0, drop);
      this.r.splice(0, drop);
      this.g.splice(0, drop);
      this.b.splice(0, drop);
    }
  }

  get spanSeconds() {
    const n = this.t.length;
    return n < 2 ? 0 : this.t[n - 1] - this.t[0];
  }

  get sampleRate() {
    const span = this.spanSeconds;
    return span > 0 ? (this.t.length - 1) / span : 0;
  }

  // Run POS + spectral analysis on the current window. Updates bpm/confidence.
  update() {
    const n = this.t.length;
    if (this.spanSeconds < MIN_SECONDS || n < 32) return;

    const pulse = pos(this.r, this.g, this.b);
    this.pulse = pulse;

    const fs = this.sampleRate;
    const { bpm, confidence } = estimateBpm(pulse, fs);
    this.rawBpm = bpm;
    this.confidence = confidence;

    if (bpm != null && confidence >= MIN_CONFIDENCE) {
      this.bpm = this.bpm == null
        ? bpm
        : BPM_EMA_ALPHA * bpm + (1 - BPM_EMA_ALPHA) * this.bpm;
    }
  }
}

// Core POS transform. Inputs are arrays of per-frame mean R/G/B over the window.
// Returns the recovered pulse waveform (Float64Array, zero-mean).
export function pos(rArr, gArr, bArr) {
  const n = rArr.length;
  const out = new Float64Array(n);

  // Temporal normalization: divide each channel by its window mean.
  const mr = mean(rArr) || 1;
  const mg = mean(gArr) || 1;
  const mb = mean(bArr) || 1;

  const s1 = new Float64Array(n);
  const s2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const rn = rArr[i] / mr;
    const gn = gArr[i] / mg;
    const bn = bArr[i] / mb;
    // POS projection matrix [[0, 1, -1], [-2, 1, 1]] applied to (rn, gn, bn):
    s1[i] = gn - bn;
    s2[i] = -2 * rn + gn + bn;
  }

  // h = s1 + (std(s1)/std(s2)) * s2
  const sd2 = std(s2) || 1;
  const alpha = std(s1) / sd2;
  for (let i = 0; i < n; i++) out[i] = s1[i] + alpha * s2[i];

  // Zero-mean the result.
  const m = mean(out);
  for (let i = 0; i < n; i++) out[i] -= m;
  return out;
}
