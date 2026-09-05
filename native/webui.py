"""Minimal browser UI for cli.py -- local by default, remotely usable too.

Picks a video + settings, launches a render as a plain subprocess -- the
exact same `python cli.py ...` command you'd type by hand -- and shows live
progress parsed straight from cli.py's own "Progress: ..." log lines. No
separate progress-tracking mechanism to keep in sync with cli.py; if cli.py's
progress line format ever changes, update PROGRESS_RE to match.

Two ways to point it at an input video:
  - a server-local path (same machine/pod as this script) -- fastest, no
    copy needed
  - a browser file upload (POST /upload) -- for when the browser is on a
    different machine than the one running this script (e.g. driving a
    rented pod's webui from your own PC): the file is saved server-side and
    the returned path is used exactly like a typed-in path would be.

Local-only by design, same as before: binds to 127.0.0.1. To reach it from
your own PC when it's actually running on a rented pod, use an SSH local
port-forward (`ssh -L 8765:localhost:8765 ...`) rather than exposing this
on a public port -- SSH's own key auth secures the connection, so this
script itself still needs no auth of its own, same threat model as
"local-only" always had. Don't pass --host 0.0.0.0 to expose this
directly on a public interface; it accepts uploads and runs subprocesses
with no login of its own.

Job state lives in memory -- lost on restart, which is fine for a render
you're actively watching in the same browser tab.

Run with: python webui.py [--host 127.0.0.1] [--port 8765]
Then open http://127.0.0.1:8765/ in a browser (or, over an SSH tunnel from
another machine, whatever local port you forwarded).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response

from presets import STYLE_PRESETS

# Duplicated from cli.py rather than imported -- same reason as
# _real_core_count() below: importing cli.py as a module runs its whole
# top-level (sys.argv pre-scan, heavy mediapipe/tf imports) against
# webui.py's own argv, which makes no sense here.
QUALITY_PRESETS = {
    "indistinguishable": 18, "optimized": 21, "balanced": 24,
    "small": 28, "aggressive": 32, "maximum": 40,
}

NATIVE_DIR = Path(__file__).resolve().parent
CLI_PATH = NATIVE_DIR / "cli.py"
WEBSITE_DIR = NATIVE_DIR.parent  # the browser app's own index.html/style.css live here, when present
UPLOAD_DIR = NATIVE_DIR / "webui_uploads"


def _real_core_count():
    """Duplicated from cli.py rather than imported from it, deliberately --
    importing cli.py as a module would run its whole top-level (sys.argv
    scan, OMP/TF env var setup) using webui.py's OWN argv, which makes no
    sense (that argv is --host/--port, not --threads-per-worker). Same
    real bug this fixes as cli.py's own copy: os.cpu_count() reports a
    container's HOST-level logical CPU count, not what's actually
    allocated to it -- confirmed live, a pod with 16 real cores reported
    128 via os.cpu_count(), and this page's own default --workers value
    (populated from this function) fed that straight into a real render,
    spawning 128 MediaPipe workers that oversubscribed 16 real cores into
    a stall (load average 38-65)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4

app = FastAPI()
jobs = {}  # job_id -> dict(process, log_path, output_path, cmd, start_time, cancelled)

PROGRESS_RE = re.compile(
    r"Progress:\s*(\d+)/(\d+) frames \(\s*([\d.]+)%\) \|\s*([\d.]+) fps \| "
    r"elapsed\s*([\d.]+)m \| ETA\s*([\d.]+)m"
)


def _default_output(input_path):
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_linearty{p.suffix}"))


def _tail_progress(log_path):
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None
    done, total, pct, fps, elapsed, eta = matches[-1]
    return {
        "done": int(done), "total": int(total), "pct": float(pct),
        "fps": float(fps), "elapsed_min": float(elapsed), "eta_min": float(eta),
    }


def _last_error_line(log_path):
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        if "Error" in line or "Traceback" in line:
            return line.strip()
    return None


@app.get("/", response_class=HTMLResponse)
def index():
    preset_options = "".join(
        f'<option value="{key}"{" selected" if key == "ultimate" else ""}>{val["label"]}</option>'
        for key, val in STYLE_PRESETS.items()
    )
    quality_options = "".join(
        f'<option value="{key}"{" selected" if key == "balanced" else ""}>{key} (CRF {crf_val})</option>'
        for key, crf_val in QUALITY_PRESETS.items()
    )
    cores = _real_core_count()
    # Same dynamic formula as cli.py's own default -- see that file's
    # --threads-per-worker help text for the validated reasoning
    # (fewer, properly multi-threaded workers beats many single-threaded
    # ones for MediaPipe specifically).
    threads_per_worker = max(1, round(cores ** 0.5))
    default_workers = max(1, cores // threads_per_worker)
    return (
        PAGE_TEMPLATE.replace("__PRESET_OPTIONS__", preset_options)
        .replace("__QUALITY_OPTIONS__", quality_options)
        .replace("__DEFAULT_WORKERS__", str(default_workers))
        .replace("__DEFAULT_THREADS_PER_WORKER__", str(threads_per_worker))
    )


@app.get("/style.css")
def website_stylesheet():
    # Reuses the actual browser app's stylesheet (one directory up) so this
    # job-launcher page shares its exact color/type/component language
    # instead of drifting into its own separate look. Falls back to a
    # minimal inline-equivalent palette when running somewhere that doesn't
    # have the full repo checked out (e.g. a rented pod with just native/
    # uploaded) -- same class names, so the page still renders sensibly.
    css_path = WEBSITE_DIR / "style.css"
    if css_path.is_file():
        return Response(css_path.read_text(encoding="utf-8"), media_type="text/css")
    return Response(FALLBACK_CSS, media_type="text/css")


@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    """For when the browser driving this page is on a different machine
    than the one running it (e.g. this webui running on a rented pod,
    reached from your own PC over an SSH tunnel) -- your local file paths
    don't exist on the pod's filesystem, so the file has to actually be
    uploaded, not just referenced by path. Saves under webui_uploads/ and
    returns that server-side path, which the page then fills into the
    input path field exactly as if you'd typed it."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest)}


@app.post("/jobs")
def start_job(
    input_path: str = Form(...),
    output_path: str = Form(""),
    preset: str = Form("ultimate"),
    workers: int = Form(...),
    threads_per_worker: int = Form(...),
    max_dimension: str = Form(""),
    human_aware: str = Form("default"),
    pose_lines: str = Form(""),
    face_contours: str = Form(""),
    encoder: str = Form("auto"),
    quality: str = Form("balanced"),
    crf: str = Form(""),
    temporal_denoise: str = Form(""),
):
    input_path = input_path.strip().strip('"')
    if not os.path.isfile(input_path):
        return JSONResponse({"error": f"Input file not found: {input_path}"}, status_code=400)
    if preset not in STYLE_PRESETS:
        return JSONResponse({"error": f"Unknown preset: {preset}"}, status_code=400)
    if quality not in QUALITY_PRESETS:
        return JSONResponse({"error": f"Unknown quality tier: {quality}"}, status_code=400)

    out = output_path.strip().strip('"') or _default_output(input_path)
    job_id = uuid.uuid4().hex[:8]
    log_path = NATIVE_DIR / f"webui_job_{job_id}.log"

    cmd = [
        sys.executable, str(CLI_PATH), input_path, "-o", out,
        "--preset", preset, "--workers", str(workers),
        "--threads-per-worker", str(threads_per_worker), "--encoder", encoder,
        "--quality", quality,
    ]
    if crf.strip():
        cmd += ["--crf", crf.strip()]
    if temporal_denoise == "on":
        cmd.append("--temporal-denoise")
    if max_dimension.strip():
        cmd += ["--max-dimension", max_dimension.strip()]
    if human_aware == "on":
        cmd.append("--human-aware")
    elif human_aware == "off":
        cmd.append("--no-human-aware")
    if pose_lines == "on":
        cmd.append("--pose-lines")
    if face_contours == "on":
        cmd.append("--face-contours")

    with open(log_path, "w", encoding="utf-8") as log_f:
        # Popen duplicates the fd into the child; our handle can close right
        # away without affecting the child's writes -- otherwise it leaks
        # for the server's whole lifetime, since nothing else ever closes it.
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(NATIVE_DIR))

    jobs[job_id] = {
        "process": proc, "log_path": str(log_path),
        "output_path": out, "cmd": cmd, "start_time": time.time(), "cancelled": False,
    }
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)

    returncode = job["process"].poll()
    progress = _tail_progress(job["log_path"])

    if job["cancelled"]:
        state = "cancelled"
    elif returncode is None:
        state = "running"
    elif returncode == 0:
        state = "done"
    else:
        state = "failed"

    return {
        "state": state,
        "returncode": returncode,
        "progress": progress,
        "output_path": job["output_path"],
        "error": _last_error_line(job["log_path"]) if state == "failed" else None,
    }


@app.get("/jobs/{job_id}/log")
def job_log(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    try:
        text = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""
    return JSONResponse({"log": text[-4000:]})


@app.get("/jobs/{job_id}/download")
def job_download(job_id: str):
    """Streams the finished render back over the same connection the
    browser used to reach this page -- so when this page is opened through
    an SSH local port-forward (the documented way to drive a pod's webui
    from your own PC), the download travels through that same tunnel and
    lands whereever the browser is configured to save downloads, with no
    separate transfer step or script needed."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    if job["process"].poll() != 0:
        return JSONResponse({"error": "job not finished successfully"}, status_code=409)
    out_path = Path(job["output_path"])
    if not out_path.is_file():
        return JSONResponse({"error": f"output file missing: {out_path}"}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(str(out_path), filename=out_path.name, media_type="video/mp4")


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    job["cancelled"] = True
    pid = job["process"].pid
    # A plain .terminate() only kills the immediate subprocess, leaving its
    # multiprocessing.Pool workers as orphans on Windows -- kill the whole
    # process tree instead (same fix as the manual cleanup this came from).
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    return {"ok": True}


FALLBACK_CSS = """
:root {
  --bg: #f4efe6; --bg-strong: #e7ddd0; --surface: #faf6ee; --surface-strong: #fffaf2;
  --ink: #16212f; --ink-soft: #6e655b; --accent: #b85c38; --accent-strong: #8b3e1f;
  --line: color-mix(in oklab, #16212f 12%, transparent);
  --success: #1f6b4f; --warn: #a35a1c; --radius: 16px; --radius-sm: 12px;
  --font-sans: "Outfit", "Avenir Next", "Segoe UI", system-ui, sans-serif;
  --font-display: "Fraunces", "Iowan Old Style", Palatino, Georgia, serif;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: var(--font-sans); color: var(--ink); background: var(--bg); }
h1, h2 { font-family: var(--font-display); font-weight: 500; }
.page-shell { width: min(80rem, calc(100% - 2rem)); margin: 0 auto; padding: 0 0 2.5rem; }
.site-header { display: flex; flex-direction: column; gap: 0.75rem; padding: 0.85rem 0 1rem; border-bottom: 1px solid var(--line); }
.brand-row { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
.brand-mark { display: flex; align-items: center; gap: 0.75rem; }
.brand-icon { display: grid; place-items: center; width: 2.5rem; height: 2.5rem; border-radius: 12px; background: var(--accent); color: #fff8f2; }
.eyebrow { margin: 0; font-size: 0.7rem; font-weight: 600; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.04em; }
.brand-name { margin: 0; font-family: var(--font-display); font-size: 1.1rem; }
.hero h1 { font-size: 1.4rem; margin: 0.75rem 0 0.4rem; }
.hero-copy { color: var(--ink-soft); line-height: 1.5; margin: 0; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); }
.controls-panel { padding: 1rem; }
.panel-heading h2 { margin: 0; font-size: 0.95rem; }
.panel-heading p { margin: 0.35rem 0 0; color: var(--ink-soft); }
.control-group, .control-grid { margin-top: 1rem; }
.control-group label, .control-grid label { display: block; margin-bottom: 8px; font-weight: 600; }
.control-group input, .control-group select, .control-grid select { width: 100%; min-height: 44px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 12px; background: #fff; color: var(--ink); }
.control-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.action-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1.15rem; }
.primary-button, .secondary-button { border: none; border-radius: 12px; padding: 11px 16px; min-height: 44px; cursor: pointer; font-weight: 500; }
.primary-button { background: var(--ink); color: #fff; }
.secondary-button { background: transparent; border: 1px solid var(--line); color: var(--ink); }
"""

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linearty native — server render</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Outfit:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/style.css">
<style>
  /* Job-launcher-specific bits only -- everything else (colors, type,
     .panel/.control-group/.primary-button) comes from the real site's own
     style.css, linked above, so this stays visually the same app. */
  .page-shell { padding-top: 1.5rem; }
  .job-list { display: flex; flex-direction: column; gap: 0.9rem; margin-top: 1rem; }
  .job-card { padding: 0.9rem 1rem; }
  .job-card .job-head { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; }
  .job-card .job-id { font-family: monospace; color: var(--ink-soft); font-size: 0.85rem; }
  .job-card .job-state { font-weight: 600; font-size: 0.9rem; }
  .job-card.state-done .job-state { color: var(--success); }
  .job-card.state-failed .job-state { color: var(--warn); }
  .job-card progress { width: 100%; height: 0.6rem; border-radius: 6px; accent-color: var(--accent); margin-top: 0.6rem; }
  .job-card .job-meta { font-size: 0.8rem; color: var(--ink-soft); margin-top: 0.4rem; }
  .job-card details { margin-top: 0.5rem; }
  .job-card summary { cursor: pointer; font-size: 0.8rem; color: var(--ink-soft); }
  .job-card pre.log { background: var(--ink); color: #e7ddd0; padding: 0.6rem; border-radius: 10px; font-size: 0.72rem; max-height: 220px; overflow: auto; white-space: pre-wrap; margin-top: 0.4rem; }
  .overlay-row { display: flex; align-items: center; gap: 1.2rem; min-height: 44px; }
  .overlay-row label { display: flex; align-items: center; gap: 0.4rem; font-weight: 500; margin-bottom: 0; }
  .overlay-row input[type=checkbox] { width: auto; min-height: 0; }
</style>
</head>
<body>
<main class="page-shell">
  <header class="site-header">
    <div class="brand-row">
      <div class="brand-mark">
        <span class="brand-icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>
        </span>
        <div>
          <p class="eyebrow">Server-side render queue</p>
          <p class="brand-name">Linearty native</p>
        </div>
      </div>
    </div>
    <section class="hero">
      <h1>Drive the native CLI from a browser tab.</h1>
      <p class="hero-copy">
        Same ink pipeline as the website, running as a local process on this machine's CPU/GPU
        instead of a browser tab -- point it at a file path, watch live progress, no upload needed.
      </p>
    </section>
  </header>

  <section class="panel controls-panel">
    <div class="panel-heading">
      <h2>New render</h2>
      <p>Runs the exact same <code>cli.py</code> command you'd type by hand.</p>
    </div>
    <form id="renderForm">
      <div class="control-group">
        <label for="input_path">Input video path (server-local)</label>
        <input type="text" id="input_path" name="input_path" placeholder="C:\\path\\to\\input.mp4" required>
      </div>
      <div class="control-group">
        <label for="upload_file">...or upload from this browser (if it's on a different machine than this server, e.g. driving a rented pod over an SSH tunnel)</label>
        <input type="file" id="upload_file" accept="video/*">
        <div id="uploadStatus" style="margin-top:0.4rem;font-size:0.85rem;color:var(--ink-soft);"></div>
      </div>
      <div class="control-group">
        <label for="output_path">Output path (optional — defaults to same folder, "_linearty" suffix)</label>
        <input type="text" id="output_path" name="output_path" placeholder="">
      </div>
      <div class="control-grid">
        <div class="control-group">
          <label for="preset">Preset</label>
          <select id="preset" name="preset">__PRESET_OPTIONS__</select>
        </div>
        <div class="control-group">
          <label for="workers">Workers</label>
          <input type="number" id="workers" name="workers" value="__DEFAULT_WORKERS__" min="1">
        </div>
        <div class="control-group">
          <label for="threads_per_worker">Threads per worker</label>
          <input type="number" id="threads_per_worker" name="threads_per_worker" value="__DEFAULT_THREADS_PER_WORKER__" min="1">
        </div>
        <div class="control-group">
          <label for="human_aware">Human-aware</label>
          <select id="human_aware" name="human_aware">
            <option value="default">Default (on unless classic)</option>
            <option value="on">Force on</option>
            <option value="off">Force off</option>
          </select>
        </div>
        <div class="control-group">
          <label for="encoder">Encoder</label>
          <select id="encoder" name="encoder">
            <option value="auto">Auto</option>
            <option value="nvenc">NVENC</option>
            <option value="vaapi">VAAPI</option>
            <option value="qsv">QSV</option>
            <option value="libx264">libx264 (software)</option>
          </select>
        </div>
        <div class="control-group">
          <label for="max_dimension">Max dimension (blank = full source resolution)</label>
          <input type="text" id="max_dimension" name="max_dimension" placeholder="">
        </div>
        <div class="control-group">
          <label for="quality">Quality tier (output size)</label>
          <select id="quality" name="quality">__QUALITY_OPTIONS__</select>
        </div>
        <div class="control-group">
          <label for="crf">Custom CRF (blank = use quality tier above)</label>
          <input type="number" id="crf" name="crf" placeholder="" min="0" max="51">
        </div>
        <div class="control-group">
          <label>Overlays</label>
          <div class="overlay-row">
            <label><input type="checkbox" name="pose_lines"> Pose lines</label>
            <label><input type="checkbox" name="face_contours"> Face contours</label>
            <label><input type="checkbox" name="temporal_denoise" checked> Temporal denoise</label>
          </div>
        </div>
      </div>
      <div class="action-row">
        <button type="submit" class="primary-button">Start render</button>
      </div>
    </form>
  </section>

  <section class="panel controls-panel" style="margin-top: 1rem;">
    <div class="panel-heading">
      <h2>Jobs</h2>
      <p>Refreshes every couple of seconds while a render is running.</p>
    </div>
    <div class="job-list" id="jobs"></div>
  </section>
</main>

<script>
const jobsDiv = document.getElementById('jobs');
const knownJobs = JSON.parse(localStorage.getItem('linearty_jobs') || '[]');
// Tracks which finished jobs already triggered a download -- persisted so a
// page reload after a job finished doesn't re-download it, but a genuinely
// new completion always does.
const downloadedJobs = new Set(JSON.parse(localStorage.getItem('linearty_downloaded') || '[]'));

function markDownloaded(jobId) {
  downloadedJobs.add(jobId);
  localStorage.setItem('linearty_downloaded', JSON.stringify([...downloadedJobs]));
}

function saveKnownJobs() {
  localStorage.setItem('linearty_jobs', JSON.stringify(knownJobs));
}

document.getElementById('upload_file').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const statusEl = document.getElementById('uploadStatus');
  const sizeMb = (file.size / 1e6).toFixed(1);
  const startTime = Date.now();
  const formData = new FormData();
  formData.append('file', file);

  // fetch() has no visibility into upload (request body) progress -- only
  // XMLHttpRequest exposes upload.onprogress, which is why this isn't a
  // fetch() call. For a large file (real videos are often 1GB+) with no
  // progress shown, it's indistinguishable from a hung/failed upload.
  const xhr = new XMLHttpRequest();
  xhr.upload.addEventListener('progress', (ev) => {
    if (!ev.lengthComputable) return;
    const pct = (ev.loaded / ev.total * 100).toFixed(1);
    const elapsedS = (Date.now() - startTime) / 1000;
    const mbps = (ev.loaded / 1e6 / elapsedS).toFixed(1);
    const doneMb = (ev.loaded / 1e6).toFixed(1);
    statusEl.textContent = `Uploading ${file.name}: ${doneMb} / ${sizeMb} MB (${pct}%) at ${mbps} MB/s`;
  });
  xhr.addEventListener('load', () => {
    try {
      const data = JSON.parse(xhr.responseText);
      if (data.path) {
        document.getElementById('input_path').value = data.path;
        statusEl.textContent = `Uploaded -- server path: ${data.path}`;
      } else {
        statusEl.textContent = `Upload failed: ${xhr.responseText}`;
      }
    } catch (err) {
      statusEl.textContent = `Upload failed: bad response (${err})`;
    }
  });
  xhr.addEventListener('error', () => { statusEl.textContent = 'Upload failed: network error.'; });
  xhr.open('POST', '/upload');
  xhr.send(formData);
  statusEl.textContent = `Uploading ${file.name} (${sizeMb} MB)... 0%`;
});

function jobCard(jobId) {
  let el = document.getElementById('job-' + jobId);
  if (!el) {
    el = document.createElement('div');
    el.id = 'job-' + jobId;
    el.className = 'panel job-card';
    el.innerHTML = `
      <div class="job-head">
        <span class="job-id">Job ${jobId}</span>
        <span class="job-state">starting…</span>
        <button type="button" class="secondary-button cancel-btn" data-job="${jobId}">Cancel</button>
      </div>
      <progress value="0" max="100"></progress>
      <div class="job-meta"></div>
      <details><summary>Log</summary><pre class="log"></pre></details>
    `;
    jobsDiv.prepend(el);
    el.querySelector('.cancel-btn').addEventListener('click', () => {
      fetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
    });
  }
  return el;
}

async function pollJob(jobId) {
  const el = jobCard(jobId);
  try {
    const res = await fetch(`/jobs/${jobId}/status`);
    if (!res.ok) return;
    const data = await res.json();
    el.classList.remove('state-failed', 'state-done');
    el.querySelector('.job-state').textContent = data.state;
    if (data.progress) {
      const p = data.progress;
      el.querySelector('progress').value = p.pct;
      el.querySelector('.job-meta').textContent =
        `${p.done}/${p.total} frames (${p.pct.toFixed(1)}%) | ${p.fps.toFixed(1)} fps | elapsed ${p.elapsed_min.toFixed(1)}m | ETA ${p.eta_min.toFixed(1)}m`;
    }
    if (data.state === 'done') {
      el.classList.add('state-done');
      el.querySelector('.job-meta').textContent += ` — output: ${data.output_path}`;
      if (!downloadedJobs.has(jobId)) {
        // Auto-trigger the browser's own download -- lands wherever the
        // browser is configured to save files, no manual click and no
        // separate transfer step needed. Marked downloaded immediately so
        // a later poll tick (every 2s, before this download even starts)
        // can't fire it twice.
        markDownloaded(jobId);
        const a = document.createElement('a');
        a.href = `/jobs/${jobId}/download`;
        a.download = '';
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    } else if (data.state === 'failed') {
      el.classList.add('state-failed');
      el.querySelector('.job-meta').textContent = data.error || 'Failed — see log';
    }
    const logRes = await fetch(`/jobs/${jobId}/log`);
    if (logRes.ok) {
      const logData = await logRes.json();
      el.querySelector('.log').textContent = logData.log;
    }
  } catch (e) {
    // transient fetch error; next poll will retry
  }
}

function pollAll() {
  knownJobs.forEach(pollJob);
}

setInterval(pollAll, 2000);
pollAll();

document.getElementById('renderForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const params = new URLSearchParams();
  for (const [k, v] of formData.entries()) {
    if (k === 'pose_lines' || k === 'face_contours' || k === 'temporal_denoise') {
      params.set(k, 'on');
    } else {
      params.set(k, v);
    }
  }
  const res = await fetch('/jobs', { method: 'POST', body: params });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  knownJobs.unshift(data.job_id);
  saveKnownJobs();
  jobCard(data.job_id);
  pollJob(data.job_id);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Local browser UI for the Linearty native CLI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Linearty web UI at http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)
