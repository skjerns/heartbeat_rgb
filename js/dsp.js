// dsp.js — shared digital-signal-processing helpers for rPPG.
// All functions are pure and operate on plain JS number arrays / Float64Arrays.

export const HR_MIN_HZ = 0.7; //  42 BPM
export const HR_MAX_HZ = 4.0; // 240 BPM

export function mean(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
}

export function std(x, m = mean(x)) {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / x.length);
}

// Remove the mean (DC component) in place-safe fashion.
export function removeMean(x) {
  const m = mean(x);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] - m;
  return out;
}

// Remove a linear trend (least-squares fit of a line) — kills slow drift.
export function detrend(x) {
  const n = x.length;
  if (n < 2) return Float64Array.from(x);
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += i;
    sy += x[i];
    sxx += i * i;
    sxy += i * x[i];
  }
  const denom = n * sxx - sx * sx;
  const slope = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = x[i] - (slope * i + intercept);
  return out;
}

// Hann window — reduces FFT spectral leakage.
export function hann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

export function applyWindow(x, w) {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * w[i];
  return out;
}

// Next power of two >= n.
export function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// In-place iterative radix-2 Cooley–Tukey FFT.
// re/im are Float64Arrays of equal, power-of-two length. Transforms in place.
export function fft(re, im) {
  const n = re.length;
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = i + k + len / 2;
        const tRe = re[b] * curRe - im[b] * curIm;
        const tIm = re[b] * curIm + im[b] * curRe;
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const nRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nRe;
      }
    }
  }
}

// Compute the one-sided power spectrum of a real signal.
// Returns { freqs, power } where freqs are in Hz given sampleRate.
export function powerSpectrum(signal, sampleRate) {
  const n = nextPow2(signal.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < signal.length; i++) re[i] = signal[i];
  fft(re, im);
  const half = n >> 1;
  const freqs = new Float64Array(half);
  const power = new Float64Array(half);
  for (let i = 0; i < half; i++) {
    freqs[i] = (i * sampleRate) / n;
    power[i] = re[i] * re[i] + im[i] * im[i];
  }
  return { freqs, power };
}

// Estimate heart rate from a pulse signal.
// Returns { bpm, confidence } where confidence is the in-band peak's share of
// in-band spectral energy (0..1). bpm is null if no valid in-band peak.
export function estimateBpm(signal, sampleRate, fMin = HR_MIN_HZ, fMax = HR_MAX_HZ) {
  if (!signal.length || !isFinite(sampleRate) || sampleRate <= 0) {
    return { bpm: null, confidence: 0 };
  }
  const prepped = applyWindow(detrend(signal), hann(signal.length));
  const { freqs, power } = powerSpectrum(prepped, sampleRate);

  let peakIdx = -1;
  let peakVal = -Infinity;
  let bandEnergy = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] < fMin || freqs[i] > fMax) continue;
    bandEnergy += power[i];
    if (power[i] > peakVal) {
      peakVal = power[i];
      peakIdx = i;
    }
  }
  if (peakIdx < 0 || bandEnergy <= 0) return { bpm: null, confidence: 0 };

  // Parabolic interpolation around the peak bin for sub-bin frequency accuracy.
  const f = parabolicPeakHz(freqs, power, peakIdx);
  const confidence = peakVal / bandEnergy;
  return { bpm: f * 60, confidence };
}

function parabolicPeakHz(freqs, power, i) {
  if (i <= 0 || i >= power.length - 1) return freqs[i];
  const a = power[i - 1], b = power[i], c = power[i + 1];
  const denom = a - 2 * b + c;
  const delta = denom === 0 ? 0 : (0.5 * (a - c)) / denom;
  const df = freqs[1] - freqs[0];
  return freqs[i] + delta * df;
}

// Linear-resample irregular samples (values v at times t, seconds) onto a
// uniform grid at `fs` Hz spanning [t[0], t[last]]. rAF frame timing is jittery,
// which smears the spectrum; resampling first sharpens the heart-rate peak.
export function resampleUniform(t, v, fs) {
  const n = t.length;
  if (n < 2) return Float64Array.from(v);
  const span = t[n - 1] - t[0];
  const m = Math.max(2, Math.floor(span * fs) + 1);
  const out = new Float64Array(m);
  let j = 0;
  for (let i = 0; i < m; i++) {
    const tt = t[0] + i / fs;
    while (j < n - 2 && t[j + 1] < tt) j++;
    const ta = t[j], tb = t[j + 1];
    const frac = tb > ta ? (tt - ta) / (tb - ta) : 0;
    out[i] = v[j] + frac * (v[j + 1] - v[j]);
  }
  return out;
}

// Welch power spectrum: average the periodograms of overlapping, Hann-windowed
// segments. Reduces spectral variance for a steadier estimate. Each segment is
// zero-padded to `nfft` for fine bin spacing (spectral interpolation).
export function welchSpectrum(signal, fs, segLen, step, nfft) {
  segLen = Math.min(segLen, signal.length);
  nfft = nextPow2(Math.max(nfft, segLen));
  const half = nfft >> 1;
  const power = new Float64Array(half);
  const win = hann(segLen);
  let segs = 0;
  for (let start = 0; start + segLen <= signal.length; start += step) {
    const seg = detrend(signal.subarray(start, start + segLen));
    const re = new Float64Array(nfft);
    const im = new Float64Array(nfft);
    for (let i = 0; i < segLen; i++) re[i] = seg[i] * win[i];
    fft(re, im);
    for (let i = 0; i < half; i++) power[i] += re[i] * re[i] + im[i] * im[i];
    segs++;
  }
  if (segs === 0) {
    // Signal shorter than one segment — fall back to a single periodogram.
    return powerSpectrum(applyWindow(detrend(signal), hann(signal.length)), fs);
  }
  for (let i = 0; i < half; i++) power[i] /= segs;
  const freqs = new Float64Array(half);
  for (let i = 0; i < half; i++) freqs[i] = (i * fs) / nfft;
  return { freqs, power };
}

// Total spectral energy within [fMin, fMax].
export function bandEnergy(freqs, power, fMin = HR_MIN_HZ, fMax = HR_MAX_HZ) {
  let e = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= fMin && freqs[i] <= fMax) e += power[i];
  }
  return e;
}

// Top-k in-band spectral peaks as { freq, power }, strongest first. Uses
// parabolic interpolation for sub-bin frequency accuracy. Falls back to the
// single in-band maximum if no local maxima are present.
export function findBandPeaks(freqs, power, fMin = HR_MIN_HZ, fMax = HR_MAX_HZ, k = 3) {
  const peaks = [];
  for (let i = 1; i < freqs.length - 1; i++) {
    if (freqs[i] < fMin || freqs[i] > fMax) continue;
    if (power[i] > power[i - 1] && power[i] >= power[i + 1]) {
      peaks.push({ freq: parabolicPeakHz(freqs, power, i), power: power[i] });
    }
  }
  if (!peaks.length) {
    let bi = -1, bv = -Infinity;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] < fMin || freqs[i] > fMax) continue;
      if (power[i] > bv) { bv = power[i]; bi = i; }
    }
    if (bi >= 0) peaks.push({ freq: freqs[bi], power: bv });
  }
  peaks.sort((a, b) => b.power - a.power);
  return peaks.slice(0, k);
}
