# Line Art Animator

Line Art Animator is a browser-only tool for turning images and short video clips into clean line-art styled exports.

## What it does

- Processes images fully client-side and exports PNG files.
- Processes videos fully client-side with OpenCV for frame styling and FFmpeg WASM for MP4 assembly.
- Keeps media on the user's machine. Nothing is uploaded to a server.

## Production operating guidance

- Best results come from short clips, reference footage, and moderate resolutions.
- For video, start with `75%` render size and `18 FPS` unless you know the browser can handle more.
- Use `50%` render size for long or high-resolution clips.
- Files above `250 MB` are intentionally rejected because browser memory usage becomes unreliable.
- Clips longer than roughly `20 seconds` can still work, but expect significantly longer rendering times.

## Runtime model

- OpenCV is used for edge extraction and line-art styling.
- FFmpeg WASM is loaded only when video export is needed.
- Video rendering works by sampling frames in the browser, converting them to PNG frames, and encoding them into MP4.

## Local use

Because the app loads browser assets and WASM files, run it from a local web server rather than opening the HTML file directly when possible.

Example:

```powershell
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Current constraints

- Rendering very long clips is still bounded by browser CPU and memory limits.
- Cancellation is best-effort. It stops frame generation and resets FFmpeg, but the browser may need a moment to release memory after a large job.
- The app is intentionally optimized for reliability over maximum throughput.