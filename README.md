# Pulse from Pixels — webcam heart rate in the browser (rPPG)

Open the page, allow your camera, and watch your heart rate appear — measured
**contactlessly from the color of your skin**. If several people are in frame,
each face gets its own live BPM label.

Everything runs **100% in your browser**. No video, image, or measurement is
ever uploaded.

## The method (the "exact model")

This is **remote photoplethysmography (rPPG)**. Every heartbeat sends a
blood-volume pulse through the capillaries in your skin, which slightly changes
how much light (especially green) the skin absorbs. Averaging the skin's RGB
over a region across video frames produces a tiny periodic signal at your pulse
frequency.

The pulse is recovered with the **POS — "Plane-Orthogonal-to-Skin"** algorithm
(Wang, den Brinker, Stuijk & de Haan, *"Algorithmic Principles of Remote PPG"*,
IEEE Transactions on Biomedical Engineering, 2017). POS is a training-free,
lighting-invariant improvement over the earlier CHROM method, so it runs
entirely client-side with no ML inference for the signal step. We then band-pass
the signal to 0.7–4 Hz (42–240 BPM) and take the dominant spectral peak (FFT) as
the heart rate.

Face detection / ROI selection (forehead + cheeks) uses Google's
**[MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js)**,
which also runs in-browser via WASM and supports multiple faces.

A second **pulse-amplified view** uses real-time Eulerian color magnification
(Wu et al., SIGGRAPH 2012) to exaggerate the color pulsing on faces with a
detected beat, so you can *see* the heartbeat.

> ⚠️ This is a fun estimate, **not a medical device**. Accuracy depends on
> lighting, stillness, skin visibility, and webcam quality.

### Accuracy techniques

To tighten the estimate, the pipeline:

- **Locks the camera's auto-exposure and white-balance** (where supported) so
  the camera stops "correcting" the very signal we measure.
- **Resamples the RGB onto a uniform time grid** before the FFT — `rAF` frame
  timing is jittery, which would otherwise smear the spectral peak.
- **Averages only skin-colored pixels** (YCbCr test) inside the forehead/cheek
  ROIs, excluding brows, lips, hair, and background.
- Uses **Welch spectral averaging** over a longer (14 s) window for a steadier,
  lower-variance estimate.
- **Tracks the pulse frequency** with a harmonic guard, so the reading doesn't
  snap to 2× / ½× the true rate or jump on a transient.

## Project layout

```
index.html                 # UI shell
css/style.css              # styles
js/dsp.js                  # FFT, band-pass, detrend, peak→BPM
js/rppg.js                 # POS algorithm + per-face sliding-window estimator
js/roi.js                  # forehead/cheek ROI sampling (mean RGB)
js/face.js                 # MediaPipe FaceLandmarker + multi-face tracker
js/main.js                 # camera, render loop, overlay + waveform UI
js/selftest.js             # DSP self-test (no camera needed)
.github/workflows/deploy-pages.yml   # static deploy to GitHub Pages
```

No build step, no dependencies to install — libraries load from a CDN.

## Run locally

A camera needs a **secure context**, so serve over `http://localhost` (which
counts) rather than opening the file directly:

```bash
cd heartbeat_rgb
python3 -m http.server 8000
# then open http://localhost:8000
```

### Verify the math without a camera

Open <http://localhost:8000/?selftest=1>. It feeds synthetic signals at known
heart rates through the POS + FFT pipeline and logs `PASS/FAIL` per case to the
browser console (expects all to pass within ±2.5 BPM).

## Deploy to GitHub Pages

This repo is already initialized with a Pages workflow. To publish:

1. Create an empty repo on GitHub (no README/license).
2. Point this folder at it and push:
   ```bash
   git remote add origin git@github.com:<you>/<repo>.git
   git branch -M main
   git push -u origin main
   ```
3. In the repo: **Settings → Pages → Build and deployment → Source: GitHub
   Actions** (one-time).
4. The `Deploy to GitHub Pages` workflow runs on each push; your site appears at
   `https://<you>.github.io/<repo>/`. Pages is HTTPS, so the camera works there.

## Tips for a good reading

- Face even, bright light; avoid backlight and harsh shadows.
- Hold still — motion is the biggest source of error.
- Allow ~10 seconds of warm-up before the BPM stabilizes.
