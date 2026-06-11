// selftest.js — verify the POS + FFT pipeline without a camera.
//
// We synthesize a plausible skin-RGB signal: a DC skin tone plus a small pulse
// modulation at a known heart rate (with the green channel responding most, as
// real skin does), some slow lighting drift, and noise. The recovered BPM must
// land within tolerance of the truth. Run via index.html?selftest=1.

import { pos } from './rppg.js';
import {
  estimateBpm,
  resampleUniform,
  welchSpectrum,
  findBandPeaks,
  nextPow2,
} from './dsp.js';

function synth({ bpm, fs, seconds, seed = 1 }) {
  const f = bpm / 60;
  const n = Math.round(fs * seconds);
  const r = new Float64Array(n);
  const g = new Float64Array(n);
  const b = new Float64Array(n);
  // Tiny deterministic PRNG so results are reproducible.
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const pulse = Math.sin(2 * Math.PI * f * t);
    const drift = 4 * Math.sin(2 * Math.PI * 0.05 * t); // slow lighting drift
    // Base skin tone with green most strongly modulated by the pulse.
    r[i] = 180 + drift + 0.6 * pulse + 1.5 * rand();
    g[i] = 120 + drift + 1.8 * pulse + 1.5 * rand();
    b[i] = 110 + drift + 0.4 * pulse + 1.5 * rand();
  }
  return { r, g, b, fs };
}

function check(name, trueBpm, tol = 2.5) {
  const fs = 30;
  const { r, g, b } = synth({ bpm: trueBpm, fs, seconds: 12 });
  const pulse = pos(r, g, b);
  const { bpm, confidence } = estimateBpm(pulse, fs);
  const err = bpm == null ? Infinity : Math.abs(bpm - trueBpm);
  const ok = err <= tol;
  console.log(
    `%c${ok ? 'PASS' : 'FAIL'}%c ${name}: true=${trueBpm} got=${bpm == null ? 'null' : bpm.toFixed(1)} ` +
      `err=${err.toFixed(2)} conf=${confidence.toFixed(3)}`,
    `font-weight:bold;color:${ok ? '#16a34a' : '#dc2626'}`,
    'color:inherit',
  );
  return ok;
}

// End-to-end check of the improved pipeline: synthesize RGB, jitter the frame
// timestamps (as rAF does), resample to a uniform grid, run POS, then recover
// the rate with the Welch spectrum + peak finder used at runtime.
function checkPipeline(name, trueBpm, tol = 2.5) {
  const fps = 30;
  const seconds = 14;
  const { r, g, b } = synth({ bpm: trueBpm, fs: fps, seconds, seed: 7 });
  // Build jittered timestamps (±35% of a frame interval).
  let jit = 123;
  const jrand = () => {
    jit = (jit * 1103515245 + 12345) & 0x7fffffff;
    return jit / 0x7fffffff - 0.5;
  };
  const t = new Float64Array(r.length);
  let acc = 0;
  for (let i = 0; i < r.length; i++) {
    t[i] = acc;
    acc += (1 / fps) * (1 + 0.7 * jrand());
  }
  const fs = 30;
  const ru = resampleUniform(t, r, fs);
  const gu = resampleUniform(t, g, fs);
  const bu = resampleUniform(t, b, fs);
  const pulse = pos(ru, gu, bu);
  const segLen = Math.round(7 * fs);
  const { freqs, power } = welchSpectrum(pulse, fs, segLen, Math.round(segLen / 2), nextPow2(segLen * 4));
  const peaks = findBandPeaks(freqs, power, 0.7, 4, 4);
  const bpm = peaks.length ? peaks[0].freq * 60 : null;
  const err = bpm == null ? Infinity : Math.abs(bpm - trueBpm);
  const ok = err <= tol;
  console.log(
    `%c${ok ? 'PASS' : 'FAIL'}%c ${name}: true=${trueBpm} got=${bpm == null ? 'null' : bpm.toFixed(1)} err=${err.toFixed(2)}`,
    `font-weight:bold;color:${ok ? '#16a34a' : '#dc2626'}`,
    'color:inherit',
  );
  return ok;
}

export function runSelfTest() {
  console.log('%c== rPPG DSP self-test ==', 'font-weight:bold');
  const results = [
    check('48 BPM (rest, low)', 48),
    check('72 BPM (typical)', 72),
    check('100 BPM (elevated)', 100),
    check('150 BPM (exercise)', 150),
    checkPipeline('72 BPM (jittered+resample+Welch)', 72),
    checkPipeline('108 BPM (jittered+resample+Welch)', 108),
  ];
  const passed = results.filter(Boolean).length;
  const allOk = passed === results.length;
  console.log(
    `%c${passed}/${results.length} passed`,
    `font-weight:bold;color:${allOk ? '#16a34a' : '#dc2626'}`,
  );
  const banner = document.getElementById('status');
  if (banner) {
    banner.textContent = `Self-test: ${passed}/${results.length} passed (see console).`;
    banner.dataset.kind = allOk ? 'info' : 'error';
  }
  return allOk;
}
