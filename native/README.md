# Linearty native CLI

A native Python port of the same ink pipeline the web app runs
(`presets.py`/`ink.py`/`human.py`/`pipeline.py` mirror `script.js`'s
`STYLE_PRESETS`, `studio-ink.js`, `human.js`, and `worker.js`'s `process()`
one-to-one). Built for dropping onto a server with more CPU or GPU than a
browser tab can use, to process a video of any size as fast as the hardware
allows.

## Why this isn't just "the website, but as a script"

The browser version carries several limits that exist only because of the
constraints of running inside a browser tab -- a shared, bounded WASM heap,
and the risk of losing hours of work to a crashed or reloaded tab. None of
that applies to a dedicated server process, so none of it is carried over
here:

| Browser limit | Why it existed there | Here |
|---|---|---|
| 4 encode workers, ~250MB/worker | One WASM heap per FFmpeg instance in a shared tab | `os.cpu_count()` workers by default (`--workers N` to override) |
| Default resolution/quality cap | Protect the WASM heap from large frames | No cap by default -- full source resolution. `--max-dimension` is opt-in |
| Streaming chunk encode, 400MB budget | Bound WASM filesystem usage | Frames stream through an `ffmpeg` pipe, never touch disk as intermediates |
| Retry / partial-salvage / checkpoint-resume | A browser tab can be killed mid-render | Not needed -- a crashed process here costs a rerun, not a lost session |

## Install

```
pip install -r requirements.txt
```

Do not additionally `pip install opencv-python` -- `mediapipe` pulls in its
own `opencv-contrib-python`, and both packages provide the `cv2` module.
Installing both risks a broken or order-dependent `cv2` import.

Requires `ffmpeg`/`ffprobe` on `PATH`.

## Usage

```
python cli.py input.mp4 -o output.mp4 --preset ultimate
```

Common flags:

- `--preset` -- one of `ultimate`, `manga`, `studio`, `neon`, `warm`, `vivid`,
  `blueprint`, `classic`, `human`, `subject`, `pencil`, `custom` (default `ultimate`)
- `--human-aware` / `--no-human-aware` -- defaults to on for every preset except
  `classic`, matching the website's own default
- `--pose-lines`, `--face-contours` -- optional landmark overlays (off by default)
- `--max-dimension N` -- opt-in resolution cap; omit for full source resolution
- `--fps N` -- override output fps (default: source fps)
- `--workers N` -- parallel worker processes (default: all CPU cores)
- `--encoder auto|nvenc|vaapi|qsv|libx264` -- see the GPU section below
- `--gpu-filter` -- opportunistic OpenCV CUDA use for the ink filter itself

While running, `cli.py` prints a real progress line every 5 seconds --
frame count, percentage, aggregate fps, elapsed time, and ETA -- parsed from
a shared-memory counter each worker updates per frame, not just a
per-segment "done" marker (which for a large `--workers` count could
otherwise mean only a handful of updates across a very long render). Each
update is its own newline-terminated line rather than an in-place `\r` bar,
specifically so it reads correctly when redirected to a log file and
tailed (`tail -f` / `Get-Content -Wait`), not just in an interactive
terminal.

**Never edit these files while a render is running.** On Windows,
`multiprocessing` always re-imports the whole script fresh from disk for
any new worker process it spawns (including replacing one that
unexpectedly died) -- editing `cli.py`/`human.py`/`pipeline.py` mid-run risks
a signature mismatch between the running main process and a freshly
spawned worker, which can spiral into an endless respawn-crash loop. This
happened during development; the fix was killing the whole process tree
and relaunching, not patching around it. `webui.py` doesn't have this
problem since it's a separate file the running job's workers never import.

## Web UI

`webui.py` is a local browser front end for the same CLI -- pick a file
path and settings, watch the same live progress in a page instead of a
terminal, cancel a running job. It shares the actual website's stylesheet
(`style.css`, one directory up) so it looks like the same product, not a
bolted-on dev tool.

```
pip install -r requirements.txt -r requirements-webui.txt
python webui.py
```

Then open http://127.0.0.1:8765/. It launches renders as plain `python
cli.py ...` subprocesses (the same command you'd run by hand), so
everything in the Usage/GPU sections above still applies -- the page is
just a form and a progress viewer on top. Local-only by design (binds to
127.0.0.1), no auth, no multi-user job isolation -- built for one person
watching their own renders, not a shared service.

## GPU: three independent, honestly-scoped levers

There is no single "use GPU" switch -- three separate things can each use
hardware acceleration or not, independently:

1. **Video encode** (`--encoder`, default `auto`): tries `h264_nvenc`,
   `h264_vaapi`, `h264_qsv` in order, verifying each with a real tiny test
   encode (not just checking `ffmpeg -encoders` -- confirmed during
   development that a hardware encoder can be compiled into an `ffmpeg`
   build and still fail to actually initialize on a machine with no
   matching GPU/driver present), then falls back to `libx264`. If a chosen
   hardware encoder dies mid-run on a specific segment, that segment
   automatically retries once with `libx264`. This is the most reliable
   lever -- ffmpeg's hardware encoders are mature.
2. **MediaPipe segmentation** (always on when `--human-aware` is active,
   automatic): tries the GPU delegate first, silently falls back to CPU on
   any failure. Same pattern the website's `human.js` uses.
3. **OpenCV CUDA for the ink filter** (`--gpu-filter`, off by default): only
   takes effect if the installed `opencv` build actually has CUDA support,
   which the standard `mediapipe`-provided `opencv-contrib-python` wheel does
   **not**. Checked via `cv2.cuda.getCudaEnabledDeviceCount()` and silently
   ignored if zero, since CPU parallelism is already the primary, always-
   available speed path (see below).

## How the parallelism works

The video is split into `--workers` contiguous frame segments. Each segment
is handed whole to one worker process (`multiprocessing.Pool`), which
decodes, processes, and encodes that entire segment sequentially before the
next segment is picked up. This is deliberate, not incidental: MediaPipe's
VIDEO running mode requires strictly increasing timestamps, which is only
guaranteed if one process handles one contiguous, ordered range of frames
from start to finish -- interleaving frames across workers (as a naive
round-robin frame-level parallelism would) would violate that. Segments are
encoded to temporary `.mp4` files and concatenated (stream copy, no
re-encode) at the end; audio is muxed back in from the original input in one
final plain `ffmpeg` pass.

Each worker calls `cv2.setNumThreads(1)` on startup. Without this, `N`
worker processes each also spawning OpenCV's own internal multi-threading
would oversubscribe the machine's cores (`N` processes x `M` internal
threads each, competing for far fewer than `N * M` actual cores) -- which
makes more cores *slower*, not faster. This is exactly the kind of
"limiting" behavior this whole port exists to avoid.

Frame-count/segment-boundary math assumes constant frame rate (CFR) input,
which covers the overwhelming majority of real sources including anything
this repo's own web app or `ffmpeg` itself produces. A genuinely
variable-frame-rate source may see minor frame-count drift at segment
boundaries.

## What's intentionally not ported

- The WebGPU/WGSL shaders in `gpu-worker.js` -- confirmed by reading that
  file's own header that they exist purely to compensate for WASM's slow
  single-threaded CPU execution in-browser, and are algorithmically
  identical to the CPU path already ported here. Native `opencv-python`
  already outperforms the WASM baseline without needing shader code.
- The live camera/screen-capture input path -- not applicable to a CLI that
  processes an existing file.
- The WASM streaming/retry/resume machinery -- see the limits table above.

## Tests

```
pip install pytest
python -m pytest tests/
```

Covers the pure numeric pieces that don't need a real video or a loaded
MediaPipe model: `compute_xdog_map`'s edge response, `compute_class_confidence`'s
local-homogeneity math, and `apply_human_ink`'s per-class blend behavior.

`human.py`'s MediaPipe integration and `cli.py`'s end-to-end pipeline (decode
-> segment -> encode -> concat -> audio mux) were verified manually against
the bundled model files and a real `ffmpeg`/`ffprobe` install during
development, across the `classic`, `ultimate` (human-aware), scaled-output,
and hardware-encoder-fallback paths -- not just unit-tested in isolation.
