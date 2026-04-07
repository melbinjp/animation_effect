# Line Art Animator

Turn videos and photos of animals, people, plants — any subject — into eye-catching line-art animation, directly in the browser. No server, no uploads.

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

## Current constraints

- Rendering very long clips is bounded by browser CPU and memory limits.
- Cancellation is best-effort; the browser may need a moment to release memory after a large job.
- The app is intentionally optimized for reliability over maximum throughput.
