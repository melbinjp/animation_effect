"""Minimal local browser UI for cli.py.

Picks a video + settings, launches a render as a plain subprocess -- the
exact same `python cli.py ...` command you'd type by hand -- and shows live
progress parsed straight from cli.py's own "Progress: ..." log lines. No
separate progress-tracking mechanism to keep in sync with cli.py; if cli.py's
progress line format ever changes, update PROGRESS_RE to match.

Local-only by design: binds to 127.0.0.1. This is a one-person tool for
driving a script on your own machine/server, not a multi-user service -- no
auth, and job state lives in memory (lost on restart, which is fine for a
render you're actively watching in the same browser tab).

Run with: python webui.py [--host 127.0.0.1] [--port 8765]
Then open http://127.0.0.1:8765/ in a browser.
"""

import argparse
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response

from presets import STYLE_PRESETS

NATIVE_DIR = Path(__file__).resolve().parent
CLI_PATH = NATIVE_DIR / "cli.py"
WEBSITE_DIR = NATIVE_DIR.parent  # the browser app's own index.html/style.css live here

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
    return PAGE_TEMPLATE.replace("__PRESET_OPTIONS__", preset_options).replace(
        "__DEFAULT_WORKERS__", str(os.cpu_count() or 4)
    )


@app.get("/style.css")
def website_stylesheet():
    # Reuses the actual browser app's stylesheet (one directory up) so this
    # job-launcher page shares its exact color/type/component language
    # instead of drifting into its own separate look.
    css_path = WEBSITE_DIR / "style.css"
    return Response(css_path.read_text(encoding="utf-8"), media_type="text/css")


@app.post("/jobs")
def start_job(
    input_path: str = Form(...),
    output_path: str = Form(""),
    preset: str = Form("ultimate"),
    workers: int = Form(...),
    max_dimension: str = Form(""),
    human_aware: str = Form("default"),
    pose_lines: str = Form(""),
    face_contours: str = Form(""),
    encoder: str = Form("auto"),
):
    input_path = input_path.strip().strip('"')
    if not os.path.isfile(input_path):
        return JSONResponse({"error": f"Input file not found: {input_path}"}, status_code=400)
    if preset not in STYLE_PRESETS:
        return JSONResponse({"error": f"Unknown preset: {preset}"}, status_code=400)

    out = output_path.strip().strip('"') or _default_output(input_path)
    job_id = uuid.uuid4().hex[:8]
    log_path = NATIVE_DIR / f"webui_job_{job_id}.log"

    cmd = [
        sys.executable, str(CLI_PATH), input_path, "-o", out,
        "--preset", preset, "--workers", str(workers), "--encoder", encoder,
    ]
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
        <label for="input_path">Input video path</label>
        <input type="text" id="input_path" name="input_path" placeholder="C:\\path\\to\\input.mp4" required>
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
          <label>Overlays</label>
          <div class="overlay-row">
            <label><input type="checkbox" name="pose_lines"> Pose lines</label>
            <label><input type="checkbox" name="face_contours"> Face contours</label>
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

function saveKnownJobs() {
  localStorage.setItem('linearty_jobs', JSON.stringify(knownJobs));
}

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
    if (k === 'pose_lines' || k === 'face_contours') {
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
