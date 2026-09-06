# Line Art Animator

Client-side image and video line-art extraction web application. All processing executes locally in the browser using WebAssembly and WebGPU; no data is uploaded to external servers.

For high-throughput, multi-core, or CUDA GPU server-side rendering, refer to the [Native Python CLI and Server WebUI](native/README.md).

---

## Technical Stack

- **Computer Vision:** OpenCV.js (Canny edge detection, bilateral filtering, morphological transforms)
- **Stylization:** Custom XDoG (Extended Difference of Gaussians) inking pipeline
- **Segmentation:** MediaPipe Selfie Multiclass TFLite model running client-side (classifies background, hair, face skin, body skin, clothes)
- **Video Encoding:** FFmpeg WASM (`@ffmpeg/ffmpeg`) for frame demuxing and MP4 multiplexing
- **Compute Acceleration:** WebGPU compute shaders with automatic CPU/WASM fallback
- **Hosting:** Static HTML/CSS/JS (zero backend requirement)

---

## Style Presets

| Preset | Engine | Description |
| :--- | :--- | :--- |
| `ultimate` | XDoG + Canny | Default. Adaptive XDoG stroke extraction with Canny structural edges. |
| `portrait` / `human` | XDoG | Optimized for facial features, skin smoothing, and hair detail. |
| `subject` | XDoG + Mask | Isolates foreground subject onto pure white background. |
| `studio` | XDoG | Fine line weight with warm cream background and charcoal ink. |
| `manga` | XDoG | High-contrast black and white graphic styling. |
| `pencil` | XDoG | Low-contrast gradient strokes simulating graphite. |
| `neon` | XDoG | Inverted dark canvas with high-saturation cyan/magenta strokes. |
| `vivid` | XDoG | Comic-style bold indigo strokes on light canvas. |
| `warm` | XDoG | Warm cream tone `#fff7e8` background with dark umber `#5c2c12` ink strokes. |
| `blueprint` | XDoG | Architectural drafting style with pale cyan canvas and deep navy lines. |
| `classic` | Canny | Direct OpenCV Canny edge extraction without XDoG shading. |
| `body` | XDoG + Seg | Multi-class body segmentation map blended directly over ink layer. |

---

## Local Deployment

Serve directory via any HTTP server (required for Web Worker and WASM MIME headers):

```bash
# Python 3
python -m http.server 8000
```

Navigate to `http://localhost:8000`.

### Cross-Origin Isolation (COOP / COEP)

FFmpeg WASM and multithreaded features require SharedArrayBuffer. Production servers must send:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

---

## GitHub Pages Deployment

1. Push repository to GitHub.
2. In repository settings: **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to `Deploy from a branch`.
4. Select `main` branch and `/ (root)` folder, then save.

---

## Test Suite

Run unit and shader property tests:

```bash
npm install
npm test
```

Test coverage includes:
- Mathematical invariants for XDoG and normalization.
- CPU reference vs. WebGPU compute shader numerical equivalence (MAE $\le 1$, SSIM $> 0.99$).
- Worker pool queue concurrency and memory bounding under scale.

---

## Operational Limits (Browser)

- **Memory:** Video processing consumes browser heap memory. Files over 250 MB should be downscaled or processed via the native pipeline.
- **Resolution:** Recommended render size is $1280 \times 720$ or $1920 \times 1080$ at $\le 30$ FPS for browser stability.
- **Native Alternative:** For 4K resolution, long videos, or batch execution, use the [Native Pipeline](native/README.md).
