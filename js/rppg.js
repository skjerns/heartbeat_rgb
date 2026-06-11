// rppg.js — per-face heart-rate estimator using the POS algorithm.
//
// POS ("Plane-Orthogonal-to-Skin"), Wang, den Brinker, Stuijk & de Haan,
// "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
//
// Each PulseEstimator owns a sliding window of mean-RGB samples (one per video
// frame) plus their timestamps, runs POS over the window to recover the blood-
// volume pulse, and derives a smoothed BPM with a confidence value.

import {
  mean,
  std,
  nextPow2,
  resampleUniform,
  welchSpectrum,
  findBandPeaks,
  bandEnergy,
  HR_MIN_HZ,
  HR_MAX_HZ,
} from './dsp.js';

const WINDOW_SECONDS = 14;   // length of analysis window (longer -> finer freq)
const MIN_SECONDS = 7;       // need at least this much data before reporting
const MAX_SAMPLES = 900;     // hard cap on buffer length (safety)
const TARGET_FS = 30;        // uniform resampling rate (Hz)
const WELCH_SEG_SEC = 7;     // Welch segment length
const WELCH_OVERLAP = 0.5;   // Welch segment overlap
const FREQ_EMA_ALPHA = 0.3;  // smoothing of the tracked pulse frequency
const MAX_JUMP_BPM = 12;     // beat-to-beat change treated as suspect
const HIGH_CONFIDENCE = 0.4; // above this, allow larger jumps
const MIN_CONFIDENCE = 0.06; // gate: ignore low-SNR estimates
const PROX_SIGMA_HZ = 0.08;  // ~5 BPM: width of the history-proximity bonus

export class PulseEstimator {
  constructor() {
    this.t = [];   // timestamps (seconds)
    this.r = [];
    this.g = [];
    this.b = [];
    this.bpm = null;        // smoothed, reported value
    this.rawBpm = null;     // latest raw estimate
    this.confidence = 0;
    this.trackedFreq = null; // tracked pulse frequency (Hz) for peak tracking
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

    // Resample RGB onto a uniform grid so the FFT sees evenly-spaced samples.
    const fs = TARGET_FS;
    const ru = resampleUniform(this.t, this.r, fs);
    const gu = resampleUniform(this.t, this.g, fs);
    const bu = resampleUniform(this.t, this.b, fs);

    const pulse = pos(ru, gu, bu);
    this.pulse = pulse;

    // Welch-averaged spectrum (lower variance) with fine bin spacing.
    const segLen = Math.round(WELCH_SEG_SEC * fs);
    const step = Math.max(1, Math.round(segLen * (1 - WELCH_OVERLAP)));
    const nfft = nextPow2(segLen * 4);
    const { freqs, power } = welchSpectrum(pulse, fs, segLen, step, nfft);

    const peaks = findBandPeaks(freqs, power, HR_MIN_HZ, HR_MAX_HZ, 4);
    if (!peaks.length) return;
    const energy = bandEnergy(freqs, power, HR_MIN_HZ, HR_MAX_HZ);

    const chosen = this.selectPeak(peaks);
    const confidence = energy > 0 ? chosen.power / energy : 0;
    this.rawBpm = chosen.freq * 60;
    this.confidence = confidence;
    if (confidence < MIN_CONFIDENCE) return;

    // Track the pulse frequency, damping suspiciously large jumps unless the
    // estimate is highly confident.
    const newFreq = chosen.freq;
    if (this.trackedFreq == null) {
      this.trackedFreq = newFreq;
    } else {
      const jumpBpm = Math.abs(newFreq - this.trackedFreq) * 60;
      const alpha =
        jumpBpm > MAX_JUMP_BPM && confidence < HIGH_CONFIDENCE ? 0.08 : FREQ_EMA_ALPHA;
      this.trackedFreq = alpha * newFreq + (1 - alpha) * this.trackedFreq;
    }
    this.bpm = this.trackedFreq * 60;
  }

  // Peak-tracking + harmonic guard: among candidate peaks, prefer one that is
  // both strong and consistent with the tracked frequency. This stops the
  // estimate from locking onto a 2x / 0.5x harmonic of the true rate.
  selectPeak(peaks) {
    if (this.trackedFreq == null) return peaks[0];
    const top = peaks[0].power || 1;
    let best = peaks[0];
    let bestScore = -Infinity;
    for (const p of peaks) {
      const df = p.freq - this.trackedFreq;
      const prox = Math.exp(-(df * df) / (2 * PROX_SIGMA_HZ * PROX_SIGMA_HZ));
      const score = p.power / top + 0.7 * prox;
      if (score > bestScore) {
        bestScore = score;
        best = p;
      }
    }
    return best;
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
