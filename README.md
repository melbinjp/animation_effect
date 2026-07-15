# Line Art Animator

Browser-based line-art animation tool for photos and videos, with client-side processing and no uploads.

Turn videos and photos of animals, people, plants — any subject — into eye-catching line-art animation, directly in the browser.

## Live on GitHub Pages

This tool is designed to run on [GitHub Pages](https://pages.github.com/) with zero server-side configuration.

To publish:
1. Push the repository to GitHub.
2. Go to **Settings → Pages**, set the source to the `main` branch, root folder.
3. Open the published URL — the app is fully self-contained.

All vendor assets (FFmpeg WASM, OpenCV) are either bundled locally or loaded from stable public CDNs, so no build step is needed.

## Style presets

| Preset | Best for |
|---|---|
| **Manga Contrast** | Bold, high-contrast — great for portraits and animals |
| **Neon Pop** | Dark background with glowing cyan lines — vivid social-media look |
| **Vivid Toon** | Clean white bg with bold indigo lines — cartoon / comic feel |
| **Warm Sketch** | Cream background, rich brown lines — pencil-sketch warmth |
| **Studio Ink** | Neutral warm paper look |
| **Blueprint Draft** | Technical blue-tone style |

## What it does

- Processes images entirely client-side and exports PNG files.
- Processes videos client-side with OpenCV (edge detection) and FFmpeg WASM (MP4 encoding).
- Keeps all media on the user's device — nothing is uploaded anywhere.
- Video export engine loads automatically in the background when a video is selected.

## Production operating guidance

- Best results come from short clips, reference footage, and moderate resolutions.
- For video, start with `75%` render size and `18 FPS` unless you know the browser can handle more.
- Use `50%` render size for long or high-resolution clips.
- Files above `250 MB` are intentionally rejected because browser memory usage becomes unreliable.
- Clips longer than roughly `20 seconds` can still work, but expect significantly longer rendering times.

## Runtime model

- OpenCV is loaded inside a Web Worker via `importScripts` from the official docs CDN.
- FFmpeg WASM is loaded on-demand when a video file is selected (auto-starts in the background).
- Video rendering samples frames from the browser video element, applies line-art via OpenCV, and encodes with FFmpeg WASM.

## Local use

```
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Testing

To test the OpenCV processing engine loading functionality:

1. Start a local server (see above)
2. Open `http://localhost:8000/test-loading.html`
3. The test page will automatically run tests for:
   - Worker creation
   - OpenCV initialization (with 30s timeout)
   - Error handling and logging
   - Message type validation

The test page provides real-time console output and test results. All tests should pass with OpenCV initializing in 10-30 seconds depending on your connection and browser.

## Performance & Testing

### Optimizations

The video processing engine includes three performance optimizations that together deliver a 40–50% throughput improvement over the baseline hybrid pipeline:

| Optimization | Improvement | Description |
|---|---|---|
| Complete GPU pipeline | ~25% | All filters (CLAHE, Auto-Normalize, Clean-Speckles, Color-Edges) run as WebGPU compute shaders, eliminating CPU/GPU context switches |
| Adaptive worker scaling | ~10% | Worker count adjusts every 50 frames based on median frame latency and available memory |
| Parallel video decode pool | ~5% | A pool of cloned video elements decodes multiple frames concurrently, keeping workers busy |

### Expected benchmarks

- **GPU (WebGPU-capable browser required):** 5–50 ms per frame at 4K resolution. A 2.5-hour 4K 60 fps video processes in roughly 1.5–2 hours instead of 3–3.5 hours.
- **CPU fallback (Node.js / no WebGPU):** A full pipeline run on a 320×180 test frame completes in under 5 seconds in the test suite. Full 4K rendering falls back to CPU-only and will be significantly slower.

> GPU benchmarks require a WebGPU-capable browser (Chrome 113+ on Windows/macOS/Linux). The app automatically falls back to CPU processing when WebGPU is unavailable, with no loss of output quality.

### Adaptive worker scaling UI indicator

During video export, the toolbar shows the current active worker count and whether auto-scaling is enabled (e.g., `Workers: 4 · Auto-scaling: ON`). If you manually adjust the worker slider, auto-scaling turns off and the indicator updates to `Auto-scaling: OFF`.

### "Reset to Human Defaults" button

In Custom/Experiment mode, a **Reset to Human Defaults** button restores the baseline starting values optimised for human-subject footage: Ink low 40, Ink high 100, Bilateral diameter 13, Sigma 90, Clean speckles enabled. It also clears any saved settings from localStorage.

### Running tests

```
npm test
```

The test suite runs entirely in Node (no browser required) using Vitest and fast-check. It covers:

- **Pure logic** — PerformanceMonitor, AdaptiveWorkerScaler, SettingsParser, SettingsPrettyPrinter (unit + property-based round-trip)
- **GPU shader equivalence** — CPU reference vs GPU pipeline output (MAE ≤ 1, SSIM > 0.99) across CLAHE, Auto-Normalize, Clean-Speckles, Color-Edges
- **Worker scaling invariants** — worker count stays within [1, maxWorkers], no frames lost under random scale-up/down sequences
- **Decode pool concurrency** — outstanding acquired elements never exceed pool size; FIFO release order preserved
- **Memory bounds** — simulated peak memory stays within budget for videos up to 100,000 frames
- **Frame order preservation** — output frame indices match input indices under adaptive scaling
- **Custom mode stability** — all 2^10 filter toggle combinations complete without crash or exception
- **Error handling** — GPU device-lost fallback, decode failure skip, memory pressure worker reduction, settings validation

Current status: **437 tests passing across 20 test files** (as of latest run, ~8.5 s).

## Current constraints

- Rendering very long clips is bounded by browser CPU and memory limits.
- Cancellation is best-effort; the browser may need a moment to release memory after a large job.
- The app is intentionally optimized for reliability over maximum throughput.
