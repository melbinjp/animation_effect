"""Linearty Native Studio Web UI -- Interactive preview and server-side render queue.

Features:
- Material Design 3 UI with responsive 2-column studio layout.
- Instant single-frame preview on any slider, preset, or toggle change.
- Multi-view comparison: Ink, Original Photo, and Split wipe with interactive reveal slider & animated wipe.
- One-click sample loading (Portrait, Figure, Still Life).
- Direct single-image processing (PNG download) and multi-worker video render queue.
- Complete CLI parameter forwarding (--detail, --line-weight, --fps, --white-balance, etc.).
"""

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

import human as human_mod
import pipeline
from presets import HUMAN_TUNING, STYLE_PRESETS, get_human_tuning, get_preset

QUALITY_PRESETS = {
    "indistinguishable": 18,
    "optimized": 21,
    "balanced": 24,
    "small": 28,
    "aggressive": 32,
    "maximum": 40,
}

NATIVE_DIR = Path(__file__).resolve().parent
CLI_PATH = NATIVE_DIR / "cli.py"
WEBSITE_DIR = NATIVE_DIR.parent
UPLOAD_DIR = NATIVE_DIR / "webui_uploads"
OUTPUT_DIR = NATIVE_DIR / "webui_outputs"
SAMPLES_DIR = WEBSITE_DIR / "samples"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _real_core_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


app = FastAPI(title="Linearty Native Studio")
jobs = {}  # job_id -> dict(process, log_path, output_path, cmd, start_time, cancelled)

PROGRESS_RE = re.compile(
    r"Progress:\s*(\d+)/(\d+) frames \(\s*([\d.]+)%\) \|\s*([\d.]+) fps \| "
    r"elapsed\s*([\d.]+)m \| ETA\s*([\d.]+)m"
)


def _default_output(input_path: str, is_image: bool = False) -> str:
    p = Path(input_path)
    ext = ".png" if is_image else p.suffix or ".mp4"
    return str(OUTPUT_DIR / f"{p.stem}_linearty{ext}")


def _tail_progress(log_path: str):
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None
    done, total, pct, fps, elapsed, eta = matches[-1]
    return {
        "done": int(done),
        "total": int(total),
        "pct": float(pct),
        "fps": float(fps),
        "elapsed_min": float(elapsed),
        "eta_min": float(eta),
    }


def _last_error_line(log_path: str):
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        if "Error" in line or "Traceback" in line:
            return line.strip()
    return None


def is_image_file(path_str: str) -> bool:
    ext = Path(path_str).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def probe_media_file(path_str: str):
    p = Path(path_str)
    if not p.is_file():
        return None
    if is_image_file(path_str):
        img = cv2.imread(str(p))
        if img is None:
            return None
        h, w = img.shape[:2]
        return {
            "type": "image",
            "width": w,
            "height": h,
            "duration": 0.0,
            "fps": 0.0,
            "total_frames": 1,
            "has_audio": False,
        }
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(p),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / float(den or 1)
        duration = float(data.get("format", {}).get("duration") or 0.0)

        nb_frames = stream.get("nb_frames")
        if nb_frames and nb_frames.isdigit() and int(nb_frames) > 0:
            total_frames = int(nb_frames)
        elif duration > 0:
            total_frames = max(1, round(duration * fps))
        else:
            total_frames = 1

        has_audio_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(p),
        ]
        has_audio = bool(
            subprocess.run(has_audio_cmd, capture_output=True, text=True).stdout.strip()
        )
        return {
            "type": "video",
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "duration": round(duration, 2),
            "total_frames": total_frames,
            "has_audio": has_audio,
        }
    except Exception:
        return None


def extract_frame_raw(input_path: str, timestamp: float = 0.0, max_dimension: Optional[int] = 1280):
    p = Path(input_path)
    if not p.is_file():
        raise FileNotFoundError(f"Media file not found: {input_path}")

    if is_image_file(input_path):
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {input_path}")
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(p),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not res.stdout:
            cmd_fb = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(p),
                "-ss",
                f"{timestamp:.3f}",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-q:v",
                "2",
                "-",
            ]
            res = subprocess.run(cmd_fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0 or not res.stdout:
                raise RuntimeError("Failed to extract video frame at timestamp")
        img_bgr = cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError("Could not decode frame from ffmpeg stream")

    if max_dimension and max_dimension > 0:
        h, w = img_bgr.shape[:2]
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            nw = max(2, round(w * scale) // 2 * 2)
            nh = max(2, round(h * scale) // 2 * 2)
            img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    return rgba, img_bgr


@app.get("/samples/{filename}")
def get_sample(filename: str):
    target = (SAMPLES_DIR / filename).resolve()
    if not target.is_file() or not str(target).startswith(str(SAMPLES_DIR.resolve())):
        return JSONResponse({"error": "Sample not found"}, status_code=404)
    return FileResponse(str(target), media_type="image/jpeg")


@app.get("/probe")
def probe_media(path: str = Query(...)):
    clean_path = path.strip().strip('"')
    meta = probe_media_file(clean_path)
    if not meta:
        return JSONResponse({"error": "Unable to inspect media file"}, status_code=400)
    return meta


@app.get("/preview/source")
def preview_source(path: str = Query(...), t: float = Query(0.0), max_dim: int = Query(1280)):
    clean_path = path.strip().strip('"')
    try:
        _, img_bgr = extract_frame_raw(clean_path, timestamp=t, max_dimension=max_dim)
        success, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success:
            return JSONResponse({"error": "Encode failed"}, status_code=500)
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/preview")
async def preview_filter(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    input_path = str(data.get("input_path", "")).strip().strip('"')
    if not input_path or not os.path.isfile(input_path):
        return JSONResponse({"error": f"File not found: {input_path}"}, status_code=400)

    timestamp = float(data.get("timestamp", 0.0))
    preset_name = data.get("preset", "ultimate")
    if preset_name not in STYLE_PRESETS:
        preset_name = "ultimate"
    preset = get_preset(preset_name)

    detail = int(data.get("detail", 62))
    line_weight = int(data.get("line_weight", 1))
    human_aware = bool(data.get("human_aware", preset_name != "classic"))
    pose_lines = bool(data.get("pose_lines", False))
    face_contours = bool(data.get("face_contours", False))
    temporal_denoise = bool(data.get("temporal_denoise", True))
    white_balance = bool(data.get("white_balance", False))
    auto_normalize = bool(data.get("auto_normalize", True))
    dark_boost = bool(data.get("dark_boost", False))

    custom_mode = preset_name == "custom"
    max_dim = int(data.get("max_dimension", 1280) or 1280)

    settings = {
        "preset": dict(preset),
        "engine": preset["engine"],
        "detail": detail,
        "custom_mode": custom_mode,
        "white_balance": white_balance,
        "auto_normalize": auto_normalize,
        "dark_boost": dark_boost,
        "dark_boost_clip": float(data.get("dark_boost_clip", 2.5)),
        "clean_speckles": bool(data.get("clean_speckles", preset.get("clean_speckles", False))),
        "clean_speckles_intensity": int(data.get("clean_speckles_intensity", 1)),
        "merge_double_edge": bool(data.get("merge_double_edge", preset.get("merge_double_edge", False))),
        "merge_double_edge_intensity": int(data.get("merge_double_edge_intensity", 2)),
        "line_weight": line_weight,
        "color_edges": bool(data.get("color_edges", False)),
        "human_aware": human_aware,
        "pose_lines": pose_lines,
        "face_contours": face_contours,
        "temporal_denoise": temporal_denoise,
    }

    if custom_mode:
        settings.update({
            "use_bilateral": bool(data.get("use_bilateral", True)),
            "bilateral_passes": int(data.get("bilateral_passes", 2)),
            "use_gaussian": bool(data.get("use_gaussian", False)),
            "gaussian_passes": int(data.get("gaussian_passes", 1)),
            "use_median": bool(data.get("use_median", False)),
            "median_passes": int(data.get("median_passes", 1)),
        })
        if "custom_bg" in data:
            bg_hex = str(data["custom_bg"]).lstrip("#")
            if len(bg_hex) == 6:
                settings["preset"]["background"] = tuple(int(bg_hex[i : i + 2], 16) for i in (0, 2, 4))
        if "custom_ink" in data:
            ink_hex = str(data["custom_ink"]).lstrip("#")
            if len(ink_hex) == 6:
                settings["preset"]["ink"] = tuple(int(ink_hex[i : i + 2], 16) for i in (0, 2, 4))
        if "custom_low_thresh" in data:
            settings["preset"]["low_threshold"] = int(data["custom_low_thresh"])
        if "custom_high_thresh" in data:
            settings["preset"]["high_threshold"] = int(data["custom_high_thresh"])
        if "custom_bilateral" in data:
            settings["preset"]["bilateral_diameter"] = int(data["custom_bilateral"])
        if "custom_sigma" in data:
            settings["preset"]["sigma"] = int(data["custom_sigma"])
        if "body_map_overlay" in data:
            settings["body_map_overlay"] = bool(data["body_map_overlay"])
    elif preset.get("body_map_overlay"):
        settings["body_map_overlay"] = True

    tuning = dict(get_human_tuning(preset_name))
    for k in ("skin_smooth", "hair_boost", "silhouette_boost", "subject_isolation"):
        if k in data and data[k] is not None:
            try:
                tuning[k] = float(data[k])
            except ValueError:
                pass
    settings.update(tuning)

    t0 = time.perf_counter()
    try:
        rgba, img_bgr = extract_frame_raw(input_path, timestamp=timestamp, max_dimension=max_dim)
        rgb_for_human = None
        frame_settings = dict(settings)

        if human_aware:
            src_h, src_w = rgba.shape[:2]
            rgb_for_human = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
            want_landmarks = pose_lines or face_contours
            if human_mod.ensure_human(landmarks=want_landmarks, mode="IMAGE"):
                human_res = human_mod.infer_human(rgb_for_human, src_w, src_h, settings, use_video_mode=False)
                if human_res:
                    frame_settings["class_mask"] = human_res["class_mask"]
                    frame_settings["extra_lines"] = human_res["extra_lines"]
                else:
                    frame_settings["human_aware"] = False
            else:
                frame_settings["human_aware"] = False

        out_rgb, _ = pipeline.process_frame(rgba, frame_settings, rgb=rgb_for_human)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            return JSONResponse({"error": "Failed to encode preview image"}, status_code=500)

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return Response(
            content=buf.tobytes(),
            media_type="image/jpeg",
            headers={"X-Process-Time-Ms": str(elapsed_ms), "Cache-Control": "no-cache"},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/process-image")
async def process_single_image(request: Request):
    """Processes a single image directly using pipeline.py and saves as high-res PNG."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request JSON"}, status_code=400)

    input_path = str(data.get("input_path", "")).strip().strip('"')
    if not input_path or not os.path.isfile(input_path):
        return JSONResponse({"error": "Input image file not found"}, status_code=400)

    output_path = str(data.get("output_path", "")).strip().strip('"') or _default_output(input_path, is_image=True)
    preset_name = data.get("preset", "ultimate")
    preset = get_preset(preset_name)

    detail = int(data.get("detail", 62))
    line_weight = int(data.get("line_weight", 1))
    human_aware = bool(data.get("human_aware", preset_name != "classic"))
    pose_lines = bool(data.get("pose_lines", False))
    face_contours = bool(data.get("face_contours", False))
    white_balance = bool(data.get("white_balance", False))
    auto_normalize = bool(data.get("auto_normalize", True))
    dark_boost = bool(data.get("dark_boost", False))

    settings = {
        "preset": dict(preset),
        "engine": preset["engine"],
        "detail": detail,
        "custom_mode": preset_name == "custom",
        "white_balance": white_balance,
        "auto_normalize": auto_normalize,
        "dark_boost": dark_boost,
        "clean_speckles": bool(data.get("clean_speckles", preset.get("clean_speckles", False))),
        "clean_speckles_intensity": int(data.get("clean_speckles_intensity", 1)),
        "merge_double_edge": bool(data.get("merge_double_edge", preset.get("merge_double_edge", False))),
        "merge_double_edge_intensity": int(data.get("merge_double_edge_intensity", 2)),
        "line_weight": line_weight,
        "human_aware": human_aware,
        "pose_lines": pose_lines,
        "face_contours": face_contours,
        "temporal_denoise": False,
    }
    if preset_name == "custom":
        if "custom_bg" in data:
            bg_hex = str(data["custom_bg"]).lstrip("#")
            if len(bg_hex) == 6:
                settings["preset"]["background"] = tuple(int(bg_hex[i : i + 2], 16) for i in (0, 2, 4))
        if "custom_ink" in data:
            ink_hex = str(data["custom_ink"]).lstrip("#")
            if len(ink_hex) == 6:
                settings["preset"]["ink"] = tuple(int(ink_hex[i : i + 2], 16) for i in (0, 2, 4))
        if "custom_low_thresh" in data:
            settings["preset"]["low_threshold"] = int(data["custom_low_thresh"])
        if "custom_high_thresh" in data:
            settings["preset"]["high_threshold"] = int(data["custom_high_thresh"])
        if "custom_bilateral" in data:
            settings["preset"]["bilateral_diameter"] = int(data["custom_bilateral"])
        if "custom_sigma" in data:
            settings["preset"]["sigma"] = int(data["custom_sigma"])
        if "body_map_overlay" in data:
            settings["body_map_overlay"] = bool(data["body_map_overlay"])
    elif preset.get("body_map_overlay"):
        settings["body_map_overlay"] = True

    settings.update(get_human_tuning(preset_name))

    max_dim_str = data.get("max_dimension")
    max_dim = int(max_dim_str) if max_dim_str and str(max_dim_str).isdigit() else None

    try:
        rgba, _ = extract_frame_raw(input_path, timestamp=0.0, max_dimension=max_dim)
        rgb_for_human = None
        frame_settings = dict(settings)

        if human_aware:
            src_h, src_w = rgba.shape[:2]
            rgb_for_human = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
            want_landmarks = pose_lines or face_contours
            if human_mod.ensure_human(landmarks=want_landmarks, mode="IMAGE"):
                human_res = human_mod.infer_human(rgb_for_human, src_w, src_h, settings, use_video_mode=False)
                if human_res:
                    frame_settings["class_mask"] = human_res["class_mask"]
                    frame_settings["extra_lines"] = human_res["extra_lines"]
                else:
                    frame_settings["human_aware"] = False
            else:
                frame_settings["human_aware"] = False

        out_rgb, _ = pipeline.process_frame(rgba, frame_settings, rgb=rgb_for_human)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), out_bgr)

        # Register as a completed job so the UI shows it nicely
        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = {
            "process": None,
            "log_path": "",
            "output_path": str(out_p),
            "start_time": time.time(),
            "cancelled": False,
            "is_image": True,
        }
        return {"job_id": job_id, "output_path": str(out_p)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
    detail: int = Form(62),
    line_weight: int = Form(1),
    fps: str = Form(""),
    white_balance: str = Form(""),
    auto_normalize: str = Form("on"),
    dark_boost: str = Form(""),
    gpu_filter: str = Form(""),
    body_map_overlay: str = Form(""),
    settings_json: str = Form(""),
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
        sys.executable,
        str(CLI_PATH),
        input_path,
        "-o",
        out,
        "--preset",
        preset,
        "--workers",
        str(workers),
        "--threads-per-worker",
        str(threads_per_worker),
        "--encoder",
        encoder,
        "--quality",
        quality,
        "--detail",
        str(detail),
        "--line-weight",
        str(line_weight),
    ]

    if crf.strip():
        cmd += ["--crf", crf.strip()]
    if fps.strip():
        cmd += ["--fps", fps.strip()]
    if temporal_denoise == "on":
        cmd.append("--temporal-denoise")
    if white_balance == "on":
        cmd.append("--white-balance")
    if auto_normalize != "on":
        cmd.append("--no-auto-normalize")
    if dark_boost == "on":
        cmd.append("--dark-boost")
    if gpu_filter == "on":
        cmd.append("--gpu-filter")
    if body_map_overlay in ("on", "true"):
        cmd.append("--body-map-overlay")
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

    if settings_json.strip():
        try:
            raw_s = json.loads(settings_json.strip())
            custom_payload = {}
            if preset == "custom" or raw_s.get("preset") == "custom":
                preset_overrides = {}
                if "custom_bg" in raw_s:
                    bg_hex = str(raw_s["custom_bg"]).lstrip("#")
                    if len(bg_hex) == 6:
                        preset_overrides["background"] = [int(bg_hex[i : i + 2], 16) for i in (0, 2, 4)]
                if "custom_ink" in raw_s:
                    ink_hex = str(raw_s["custom_ink"]).lstrip("#")
                    if len(ink_hex) == 6:
                        preset_overrides["ink"] = [int(ink_hex[i : i + 2], 16) for i in (0, 2, 4)]
                if "custom_low_thresh" in raw_s:
                    preset_overrides["low_threshold"] = int(raw_s["custom_low_thresh"])
                if "custom_high_thresh" in raw_s:
                    preset_overrides["high_threshold"] = int(raw_s["custom_high_thresh"])
                if "custom_bilateral" in raw_s:
                    preset_overrides["bilateral_diameter"] = int(raw_s["custom_bilateral"])
                if "custom_sigma" in raw_s:
                    preset_overrides["sigma"] = int(raw_s["custom_sigma"])
                if preset_overrides:
                    custom_payload["preset"] = preset_overrides

            for k in ("skin_smooth", "hair_boost", "silhouette_boost", "subject_isolation"):
                if k in raw_s and raw_s[k] is not None:
                    try:
                        custom_payload[k] = float(raw_s[k])
                    except ValueError:
                        pass

            if custom_payload:
                cmd += ["--settings-json", json.dumps(custom_payload)]
        except Exception:
            pass

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(NATIVE_DIR))

    jobs[job_id] = {
        "process": proc,
        "log_path": str(log_path),
        "output_path": out,
        "cmd": cmd,
        "start_time": time.time(),
        "cancelled": False,
        "is_image": False,
    }
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)

    if job.get("is_image"):
        return {
            "state": "done",
            "returncode": 0,
            "progress": {"pct": 100.0, "done": 1, "total": 1, "fps": 0, "elapsed_min": 0, "eta_min": 0},
            "output_path": job["output_path"],
            "error": None,
        }

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
    if job.get("is_image"):
        return JSONResponse({"log": "Single image processed successfully."})
    try:
        text = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""
    return JSONResponse({"log": text[-4000:]})


@app.get("/jobs/{job_id}/download")
def job_download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    if not job.get("is_image") and job["process"].poll() != 0:
        return JSONResponse({"error": "job not finished successfully"}, status_code=409)
    out_path = Path(job["output_path"])
    if not out_path.is_file():
        return JSONResponse({"error": f"output file missing: {out_path}"}, status_code=404)
    media_type = "image/png" if is_image_file(str(out_path)) else "video/mp4"
    return FileResponse(str(out_path), filename=out_path.name, media_type=media_type)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    job["cancelled"] = True
    if job.get("process"):
        pid = job["process"].pid
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            import signal
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
    return {"ok": True}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    meta = probe_media_file(str(dest)) or {}
    return {"path": str(dest), "filename": file.filename, "meta": meta}


@app.get("/", response_class=HTMLResponse)
def index():
    preset_chips = []
    preset_options = []
    for key, val in STYLE_PRESETS.items():
        is_sel = key == "ultimate"
        preset_chips.append(
            f'<button type="button" class="md-chip{" active" if is_sel else ""}" data-preset="{key}">{val["label"]}</button>'
        )
        preset_options.append(
            f'<option value="{key}"{" selected" if is_sel else ""}>{val["label"]}</option>'
        )

    quality_options = "".join(
        f'<option value="{key}"{" selected" if key == "balanced" else ""}>{key.capitalize()} (CRF {crf_val})</option>'
        for key, crf_val in QUALITY_PRESETS.items()
    )

    cores = _real_core_count()
    threads_per_worker = max(1, round(cores**0.5))
    default_workers = max(1, cores // threads_per_worker)

    preset_tuning_json = json.dumps(HUMAN_TUNING)
    presets_data_json = json.dumps(
        {
            k: {
                "label": v["label"],
                "engine": v["engine"],
                "low_threshold": v.get("low_threshold", 32),
                "high_threshold": v.get("high_threshold", 108),
                "bilateral_diameter": v.get("bilateral_diameter", 7),
                "sigma": v.get("sigma", 48),
                "background": "#%02x%02x%02x" % v.get("background", (255, 255, 255)),
                "ink": "#%02x%02x%02x" % v.get("ink", (0, 0, 0)),
                "clean_speckles": v.get("clean_speckles", True),
                "merge_double_edge": v.get("merge_double_edge", False),
            }
            for k, v in STYLE_PRESETS.items()
        }
    )

    html = (
        PAGE_TEMPLATE.replace("__PRESET_CHIPS__", "".join(preset_chips))
        .replace("__PRESET_OPTIONS__", "".join(preset_options))
        .replace("__QUALITY_OPTIONS__", quality_options)
        .replace("__DEFAULT_WORKERS__", str(default_workers))
        .replace("__DEFAULT_THREADS_PER_WORKER__", str(threads_per_worker))
        .replace("__PRESET_TUNING_JSON__", preset_tuning_json)
        .replace("__PRESETS_DATA_JSON__", presets_data_json)
    )
    return html


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linearty Studio -- Interactive Native Render</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --md-sys-color-primary: #8b3e1f;
    --md-sys-color-on-primary: #ffffff;
    --md-sys-color-primary-container: #ffdbd0;
    --md-sys-color-on-primary-container: #3a0b00;
    --md-sys-color-surface: #faf6ee;
    --md-sys-color-surface-container: #f4ede3;
    --md-sys-color-surface-container-high: #eee5d8;
    --md-sys-color-surface-container-highest: #e7ddcd;
    --md-sys-color-on-surface: #1d1b16;
    --md-sys-color-on-surface-variant: #53433f;
    --md-sys-color-outline: #85736e;
    --md-sys-color-outline-variant: #d8c2bc;
    --md-sys-color-secondary: #77574e;
    --md-sys-color-success: #1f6b4f;
    --md-sys-color-warn: #a35a1c;
    --md-shape-corner-sm: 8px;
    --md-shape-corner-md: 12px;
    --md-shape-corner-lg: 16px;
    --md-shape-corner-full: 9999px;
    --md-elevation-1: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --md-elevation-2: 0 3px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.08);
    --md-elevation-3: 0 10px 20px rgba(0,0,0,0.12), 0 3px 6px rgba(0,0,0,0.08);
    --font-sans: "Outfit", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-display: "Fraunces", Georgia, serif;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: var(--font-sans);
    color: var(--md-sys-color-on-surface);
    background-color: var(--md-sys-color-surface);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  /* Top App Bar */
  .top-app-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: var(--md-sys-color-surface-container);
    border-bottom: 1px solid var(--md-sys-color-outline-variant);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .brand-block { display: flex; align-items: center; gap: 0.75rem; }
  .brand-icon {
    width: 2.25rem;
    height: 2.25rem;
    border-radius: var(--md-shape-corner-md);
    background: var(--md-sys-color-primary);
    color: var(--md-sys-color-on-primary);
    display: grid;
    place-items: center;
    box-shadow: var(--md-elevation-1);
  }
  .brand-text h1 {
    font-family: var(--font-display);
    font-size: 1.25rem;
    margin: 0;
    line-height: 1.2;
    color: var(--md-sys-color-on-surface);
  }
  .brand-text p {
    margin: 0;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--md-sys-color-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .engine-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--md-sys-color-primary-container);
    color: var(--md-sys-color-on-primary-container);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.35rem 0.8rem;
    border-radius: var(--md-shape-corner-full);
  }

  /* Studio Layout */
  .studio-shell {
    max-width: 1440px;
    margin: 0 auto;
    padding: 1.25rem;
    display: grid;
    grid-template-columns: minmax(360px, 460px) 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  @media (max-width: 1024px) {
    .studio-shell { grid-template-columns: 1fr; }
  }

  /* Material Card / Panel */
  .md-card {
    background: var(--md-sys-color-surface-container);
    border: 1px solid var(--md-sys-color-outline-variant);
    border-radius: var(--md-shape-corner-lg);
    box-shadow: var(--md-elevation-1);
    overflow: hidden;
    margin-bottom: 1.25rem;
  }
  .md-card-header {
    padding: 1rem 1.25rem 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .md-card-title {
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
    color: var(--md-sys-color-on-surface);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .md-card-content { padding: 0.75rem 1.25rem 1.25rem; }

  /* Collapsible Card Details */
  details.md-card {
    border: 1px solid var(--md-sys-color-outline-variant);
    transition: box-shadow 0.2s;
  }
  details.md-card summary {
    cursor: pointer;
    list-style: none;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
  }
  details.md-card summary::-webkit-details-marker { display: none; }
  details.md-card summary::after {
    content: "▼";
    font-family: inherit;
    font-size: 0.8rem;
    color: var(--md-sys-color-outline);
    transition: transform 0.2s ease;
  }
  details.md-card[open] summary::after { transform: rotate(180deg); }

  /* Input fields & controls */
  .form-row { margin-bottom: 1rem; }
  .form-row:last-child { margin-bottom: 0; }
  .form-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--md-sys-color-on-surface);
    margin-bottom: 0.4rem;
  }
  .form-label .val-bubble {
    font-family: monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--md-sys-color-primary);
    background: var(--md-sys-color-surface-container-highest);
    padding: 0.15rem 0.45rem;
    border-radius: var(--md-shape-corner-sm);
  }

  .md-input, .md-select {
    width: 100%;
    min-height: 42px;
    padding: 0.5rem 0.85rem;
    border: 1px solid var(--md-sys-color-outline-variant);
    border-radius: var(--md-shape-corner-md);
    background: #fff;
    color: var(--md-sys-color-on-surface);
    font-family: inherit;
    font-size: 0.9rem;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .md-input:focus, .md-select:focus {
    outline: none;
    border-color: var(--md-sys-color-primary);
    box-shadow: 0 0 0 3px var(--md-sys-color-primary-container);
  }

  /* Material Range Slider */
  .md-slider {
    width: 100%;
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    border-radius: 3px;
    background: var(--md-sys-color-outline-variant);
    outline: none;
    margin: 0.6rem 0;
  }
  .md-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--md-sys-color-primary);
    cursor: pointer;
    box-shadow: var(--md-elevation-1);
    transition: transform 0.15s;
  }
  .md-slider::-webkit-slider-thumb:hover { transform: scale(1.15); }
  .md-slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--md-sys-color-primary);
    border: none;
    cursor: pointer;
    box-shadow: var(--md-elevation-1);
  }

  /* Preset Chips Container */
  .preset-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.4rem;
  }
  .md-chip {
    background: var(--md-sys-color-surface-container-high);
    border: 1px solid var(--md-sys-color-outline-variant);
    color: var(--md-sys-color-on-surface);
    border-radius: var(--md-shape-corner-full);
    padding: 0.4rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.18s ease;
    user-select: none;
  }
  .md-chip:hover {
    background: var(--md-sys-color-surface-container-highest);
    border-color: var(--md-sys-color-outline);
  }
  .md-chip.active {
    background: var(--md-sys-color-primary);
    color: var(--md-sys-color-on-primary);
    border-color: var(--md-sys-color-primary);
    box-shadow: var(--md-elevation-1);
  }

  /* Samples row */
  .sample-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }
  .sample-btn {
    flex: 1;
    background: var(--md-sys-color-surface-container-high);
    border: 1px dashed var(--md-sys-color-outline-variant);
    border-radius: var(--md-shape-corner-md);
    padding: 0.5rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--md-sys-color-secondary);
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
  }
  .sample-btn:hover {
    border-color: var(--md-sys-color-primary);
    color: var(--md-sys-color-primary);
    background: #fff;
  }

  /* Toggles / Checkboxes */
  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0;
  }
  .toggle-label {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--md-sys-color-on-surface);
    cursor: pointer;
  }
  .md-switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }
  .md-switch input { opacity: 0; width: 0; height: 0; }
  .slider-round {
    position: absolute;
    cursor: pointer;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: var(--md-sys-color-outline-variant);
    transition: 0.25s;
    border-radius: 24px;
  }
  .slider-round:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: 0.25s;
    border-radius: 50%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
  }
  input:checked + .slider-round { background-color: var(--md-sys-color-primary); }
  input:checked + .slider-round:before { transform: translateX(20px); }

  /* 2-column micro-grid */
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }

  /* Color picker control */
  .color-control {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .color-control input[type="color"] {
    width: 42px;
    height: 42px;
    border: none;
    border-radius: var(--md-shape-corner-md);
    cursor: pointer;
    background: none;
    padding: 0;
  }

  /* Action Buttons */
  .btn-primary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    min-height: 46px;
    padding: 0.75rem 1.5rem;
    background: var(--md-sys-color-primary);
    color: var(--md-sys-color-on-primary);
    border: none;
    border-radius: var(--md-shape-corner-md);
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    box-shadow: var(--md-elevation-1);
    transition: background-color 0.2s, box-shadow 0.2s, transform 0.1s;
  }
  .btn-primary:hover {
    background: #733217;
    box-shadow: var(--md-elevation-2);
  }
  .btn-primary:active { transform: scale(0.99); }
  .btn-secondary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    min-height: 36px;
    padding: 0.4rem 0.9rem;
    background: transparent;
    color: var(--md-sys-color-primary);
    border: 1px solid var(--md-sys-color-primary);
    border-radius: var(--md-shape-corner-md);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-secondary:hover {
    background: var(--md-sys-color-primary-container);
  }

  /* Right column: Preview Dock */
  .preview-dock {
    position: sticky;
    top: 5rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  .preview-card {
    background: var(--md-sys-color-surface-container-high);
    border: 1px solid var(--md-sys-color-outline-variant);
    border-radius: var(--md-shape-corner-lg);
    box-shadow: var(--md-elevation-2);
    overflow: hidden;
  }
  .preview-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    background: var(--md-sys-color-surface-container-highest);
    border-bottom: 1px solid var(--md-sys-color-outline-variant);
  }

  /* Segmented Button */
  .segmented-button {
    display: inline-flex;
    background: var(--md-sys-color-surface-container);
    border: 1px solid var(--md-sys-color-outline-variant);
    border-radius: var(--md-shape-corner-full);
    padding: 2px;
  }
  .segmented-item {
    background: transparent;
    border: none;
    border-radius: var(--md-shape-corner-full);
    padding: 0.35rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--md-sys-color-on-surface-variant);
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .segmented-item.active {
    background: var(--md-sys-color-primary);
    color: var(--md-sys-color-on-primary);
    box-shadow: var(--md-elevation-1);
  }

  /* Preview Stage */
  .preview-stage {
    position: relative;
    width: 100%;
    min-height: 380px;
    max-height: 600px;
    background: #111;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .preview-img-container {
    position: relative;
    display: inline-block;
    max-width: 100%;
    max-height: 600px;
  }
  .preview-stage img {
    display: block;
    max-width: 100%;
    max-height: 600px;
    object-fit: contain;
    user-select: none;
  }
  #sourceImg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  #inkImg { position: relative; z-index: 2; }

  .preview-overlay-info {
    position: absolute;
    bottom: 10px;
    right: 10px;
    background: rgba(0,0,0,0.65);
    color: #fff;
    padding: 0.25rem 0.6rem;
    border-radius: var(--md-shape-corner-sm);
    font-size: 0.72rem;
    font-family: monospace;
    z-index: 10;
    backdrop-filter: blur(4px);
  }

  .preview-spinner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 20;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: var(--md-shape-corner-full);
    font-size: 0.82rem;
    display: none;
    align-items: center;
    gap: 0.5rem;
    backdrop-filter: blur(4px);
  }
  .preview-spinner.visible { display: flex; }
  .spinner-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--md-sys-color-primary-container);
    animation: pulse 1s infinite alternate;
  }
  @keyframes pulse { to { transform: scale(1.4); opacity: 0.5; } }

  /* Split scrubber */
  .split-control-bar {
    display: none;
    padding: 0.6rem 1rem;
    background: var(--md-sys-color-surface-container-highest);
    border-top: 1px solid var(--md-sys-color-outline-variant);
    align-items: center;
    gap: 0.75rem;
  }
  .split-control-bar.visible { display: flex; }

  /* Timeline Scrubber */
  .timeline-bar {
    display: none;
    padding: 0.6rem 1rem;
    background: var(--md-sys-color-surface-container);
    border-top: 1px solid var(--md-sys-color-outline-variant);
    align-items: center;
    gap: 0.75rem;
  }
  .timeline-bar.visible { display: flex; }

  /* Job Cards */
  .job-card {
    padding: 1rem;
    margin-bottom: 0.75rem;
  }
  .job-head { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
  .job-title { font-weight: 600; font-size: 0.88rem; }
  .job-state { font-size: 0.8rem; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: var(--md-shape-corner-sm); }
  .state-running .job-state { background: var(--md-sys-color-primary-container); color: var(--md-sys-color-on-primary-container); }
  .state-done .job-state { background: #d1e7dd; color: #0f5132; }
  .state-failed .job-state { background: #f8d7da; color: #842029; }
  .job-progress-track {
    width: 100%;
    height: 6px;
    background: var(--md-sys-color-outline-variant);
    border-radius: 3px;
    overflow: hidden;
    margin: 0.6rem 0;
  }
  .job-progress-fill {
    height: 100%;
    background: var(--md-sys-color-primary);
    width: 0%;
    transition: width 0.3s ease;
  }
  .job-meta { font-size: 0.78rem; color: var(--md-sys-color-secondary); }
  .job-log {
    margin-top: 0.5rem;
    background: #181c20;
    color: #e2e2e6;
    padding: 0.6rem;
    border-radius: var(--md-shape-corner-sm);
    font-size: 0.72rem;
    max-height: 160px;
    overflow: auto;
    white-space: pre-wrap;
  }
</style>
</head>
<body>

<header class="top-app-bar">
  <div class="brand-block">
    <div class="brand-icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/>
      </svg>
    </div>
    <div class="brand-text">
      <h1>Linearty Native Studio</h1>
      <p>Instant Preview & Server Render</p>
    </div>
  </div>
  <div class="engine-badge">
    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#1f6b4f"></span>
    <span id="coresBadge">Detected cores: __DEFAULT_WORKERS__ workers</span>
  </div>
</header>

<main class="studio-shell">

  <!-- LEFT COLUMN: CONTROLS -->
  <div class="controls-col">
    <form id="renderForm">

      <!-- Card 1: Input Source -->
      <div class="md-card">
        <div class="md-card-header">
          <h2 class="md-card-title">1. Input Media</h2>
          <span id="mediaBadge" style="font-size:0.75rem;font-weight:600;color:var(--md-sys-color-secondary)">No media</span>
        </div>
        <div class="md-card-content">
          <div class="form-row">
            <input type="file" id="uploadFile" accept="video/*,image/*" style="display:none">
            <button type="button" class="btn-secondary" style="width:100%;margin-bottom:0.5rem;" onclick="document.getElementById('uploadFile').click()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              Upload Image or Video
            </button>
            <div id="uploadProgressText" style="font-size:0.78rem;color:var(--md-sys-color-secondary);margin-bottom:0.4rem;"></div>
          </div>
          <div class="form-row">
            <div class="form-label"><span>Server file path</span></div>
            <input type="text" id="inputPath" name="input_path" class="md-input" placeholder="/workspace/... or C:/path/to/file.mp4" required>
          </div>
          <div class="form-row">
            <div class="form-label"><span>Or try a sample still</span></div>
            <div class="sample-row">
              <button type="button" class="sample-btn" onclick="loadSample('portrait.jpg')">Portrait</button>
              <button type="button" class="sample-btn" onclick="loadSample('fullbody.jpg')">Figure</button>
              <button type="button" class="sample-btn" onclick="loadSample('still-life.jpg')">Still Life</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Card 2: Preset selector -->
      <div class="md-card">
        <div class="md-card-header">
          <h2 class="md-card-title">2. Style Presets</h2>
        </div>
        <div class="md-card-content">
          <div class="preset-chips" id="presetChipsContainer">
            __PRESET_CHIPS__
          </div>
          <select id="presetSelect" name="preset" style="display:none">
            __PRESET_OPTIONS__
          </select>
        </div>
      </div>

      <!-- Card 3: Style Adjustments -->
      <div class="md-card">
        <div class="md-card-header">
          <h2 class="md-card-title">3. Style Sliders</h2>
        </div>
        <div class="md-card-content">
          <div class="form-row">
            <div class="form-label">
              <span>Edge Detail (Threshold scaling)</span>
              <span class="val-bubble" id="detailVal">62</span>
            </div>
            <input type="range" id="detail" name="detail" class="md-slider" min="35" max="90" value="62">
          </div>

          <div class="grid-2">
            <div class="form-row">
              <div class="form-label"><span>Line Weight</span></div>
              <select id="lineWeight" name="line_weight" class="md-select">
                <option value="1" selected>Fine (1px)</option>
                <option value="2">Balanced (2px)</option>
                <option value="3">Bold (3px)</option>
                <option value="4">Heavy (4px)</option>
              </select>
            </div>
            <div class="form-row">
              <div class="form-label"><span>Video FPS</span></div>
              <select id="fpsSelect" name="fps" class="md-select">
                <option value="" selected>Original FPS</option>
                <option value="12">12 fps (Anime)</option>
                <option value="18">18 fps (Hand-drawn)</option>
                <option value="24">24 fps (Cinema)</option>
                <option value="30">30 fps (Smooth)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <!-- Card 4: Body Maps (MediaPipe) -->
      <details class="md-card" id="bodyMapsCard" open>
        <summary>4. Human-Aware Body Maps</summary>
        <div class="md-card-content">
          <div class="toggle-row">
            <label class="toggle-label" for="humanAware">Human-aware line art</label>
            <label class="md-switch">
              <input type="checkbox" id="humanAware" name="human_aware" value="on" checked>
              <span class="slider-round"></span>
            </label>
          </div>
          <p style="margin:0 0 0.75rem;font-size:0.78rem;color:var(--md-sys-color-secondary);">Quiets skin texture, emphasizes hair, outlines silhouettes, and softens busy backgrounds.</p>

          <div class="form-row">
            <div class="form-label"><span>Quiet skin</span><span class="val-bubble" id="skinSmoothVal">0.80</span></div>
            <input type="range" id="skinSmooth" class="md-slider" min="0" max="1" step="0.01" value="0.80">
          </div>
          <div class="form-row">
            <div class="form-label"><span>Hair emphasis</span><span class="val-bubble" id="hairBoostVal">1.32</span></div>
            <input type="range" id="hairBoost" class="md-slider" min="0.5" max="2" step="0.05" value="1.32">
          </div>
          <div class="form-row">
            <div class="form-label"><span>Silhouette outline</span><span class="val-bubble" id="silhouetteBoostVal">0.72</span></div>
            <input type="range" id="silhouetteBoost" class="md-slider" min="0" max="1.5" step="0.05" value="0.72">
          </div>
          <div class="form-row">
            <div class="form-label"><span>Background hush</span><span class="val-bubble" id="subjectIsolationVal">0.46</span></div>
            <input type="range" id="subjectIsolation" class="md-slider" min="0" max="1" step="0.01" value="0.46">
          </div>

          <div class="grid-2" style="margin-top:0.75rem;">
            <div class="toggle-row">
              <label class="toggle-label" for="poseLines">Pose lines</label>
              <label class="md-switch">
                <input type="checkbox" id="poseLines" name="pose_lines">
                <span class="slider-round"></span>
              </label>
            </div>
            <div class="toggle-row">
              <label class="toggle-label" for="faceContours">Face mesh</label>
              <label class="md-switch">
                <input type="checkbox" id="faceContours" name="face_contours">
                <span class="slider-round"></span>
              </label>
            </div>
          </div>
        </div>
      </details>

      <!-- Card 5: Custom / Experiment (Expands when Preset == custom) -->
      <details class="md-card" id="customControlsCard" style="display:none">
        <summary>5. Custom / Experiment Settings</summary>
        <div class="md-card-content">
          <div class="grid-2">
            <div class="form-row">
              <div class="form-label"><span>Background</span></div>
              <div class="color-control">
                <input type="color" id="customBgColor" value="#fff7e8">
                <input type="text" id="customBgHex" class="md-input" value="#fff7e8" style="min-height:36px;font-size:0.8rem">
              </div>
            </div>
            <div class="form-row">
              <div class="form-label"><span>Ink Color</span></div>
              <div class="color-control">
                <input type="color" id="customInkColor" value="#5c2c12">
                <input type="text" id="customInkHex" class="md-input" value="#5c2c12" style="min-height:36px;font-size:0.8rem">
              </div>
            </div>
          </div>

          <div class="form-row">
            <div class="form-label"><span>Low Threshold</span><span class="val-bubble" id="lowThreshVal">32</span></div>
            <input type="range" id="customLowThresh" class="md-slider" min="5" max="150" value="32">
          </div>
          <div class="form-row">
            <div class="form-label"><span>High Threshold</span><span class="val-bubble" id="highThreshVal">104</span></div>
            <input type="range" id="customHighThresh" class="md-slider" min="20" max="255" value="104">
          </div>
          <div class="form-row">
            <div class="form-label"><span>Bilateral Diameter</span><span class="val-bubble" id="bilateralDiaVal">7</span></div>
            <input type="range" id="customBilateral" class="md-slider" min="1" max="15" step="2" value="7">
          </div>
          <div class="form-row">
            <div class="form-label"><span>Bilateral Sigma</span><span class="val-bubble" id="bilateralSigmaVal">58</span></div>
            <input type="range" id="customSigma" class="md-slider" min="10" max="200" value="58">
          </div>

          <div class="toggle-row" style="margin-top:0.75rem;">
            <label class="toggle-label" for="customBodyMapOverlay">Body Map Overlay (Multi-class blend)</label>
            <label class="md-switch">
              <input type="checkbox" id="customBodyMapOverlay" name="custom_body_map_overlay">
              <span class="slider-round"></span>
            </label>
          </div>
        </div>
      </details>

      <!-- Card 6: Output & Quality Settings -->
      <details class="md-card">
        <summary>6. Output, Hardware & Advanced</summary>
        <div class="md-card-content">
          <div class="form-row">
            <div class="form-label"><span>Output path (optional)</span></div>
            <input type="text" id="outputPath" name="output_path" class="md-input" placeholder="Defaults to webui_outputs/<filename>_linearty.mp4">
          </div>
          <div class="grid-2">
            <div class="form-row">
              <div class="form-label"><span>Quality Tier</span></div>
              <select id="quality" name="quality" class="md-select">__QUALITY_OPTIONS__</select>
            </div>
            <div class="form-row">
              <div class="form-label"><span>Custom CRF</span></div>
              <input type="number" id="crf" name="crf" class="md-input" placeholder="e.g. 24" min="0" max="51">
            </div>
          </div>
          <div class="grid-2">
            <div class="form-row">
              <div class="form-label"><span>Encoder</span></div>
              <select id="encoder" name="encoder" class="md-select">
                <option value="auto" selected>Auto probe</option>
                <option value="nvenc">NVIDIA NVENC</option>
                <option value="vaapi">VAAPI</option>
                <option value="qsv">Intel QSV</option>
                <option value="libx264">libx264 (CPU)</option>
              </select>
            </div>
            <div class="form-row">
              <div class="form-label"><span>Max Dimension</span></div>
              <input type="number" id="maxDimension" name="max_dimension" class="md-input" placeholder="Full (e.g. 1920)">
            </div>
          </div>
          <div class="grid-2">
            <div class="form-row">
              <div class="form-label"><span>Workers</span></div>
              <input type="number" id="workers" name="workers" class="md-input" value="__DEFAULT_WORKERS__" min="1">
            </div>
            <div class="form-row">
              <div class="form-label"><span>Threads / Worker</span></div>
              <input type="number" id="threadsPerWorker" name="threads_per_worker" class="md-input" value="__DEFAULT_THREADS_PER_WORKER__" min="1">
            </div>
          </div>

          <div class="toggle-row" style="margin-top:0.75rem;">
            <label class="toggle-label" for="autoNormalize">Auto-normalize contrast</label>
            <label class="md-switch">
              <input type="checkbox" id="autoNormalize" name="auto_normalize" value="on" checked>
              <span class="slider-round"></span>
            </label>
          </div>
          <div class="toggle-row">
            <label class="toggle-label" for="temporalDenoise">Temporal denoise (reduces jitter)</label>
            <label class="md-switch">
              <input type="checkbox" id="temporalDenoise" name="temporal_denoise" value="on" checked>
              <span class="slider-round"></span>
            </label>
          </div>
          <div class="toggle-row">
            <label class="toggle-label" for="whiteBalance">Auto white-balance (Gray World)</label>
            <label class="md-switch">
              <input type="checkbox" id="whiteBalance" name="white_balance" value="on">
              <span class="slider-round"></span>
            </label>
          </div>
          <div class="toggle-row">
            <label class="toggle-label" for="darkBoost">Extra shadow lift</label>
            <label class="md-switch">
              <input type="checkbox" id="darkBoost" name="dark_boost" value="on">
              <span class="slider-round"></span>
            </label>
          </div>
          <div class="toggle-row">
            <label class="toggle-label" for="gpuFilter">Opportunistic CUDA filter</label>
            <label class="md-switch">
              <input type="checkbox" id="gpuFilter" name="gpu_filter" value="on">
              <span class="slider-round"></span>
            </label>
          </div>
        </div>
      </details>

      <div style="margin-top:1.25rem;">
        <button type="submit" id="actionBtn" class="btn-primary">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Start Full Render
        </button>
      </div>
    </form>
  </div>

  <!-- RIGHT COLUMN: PREVIEW DOCK & RENDER QUEUE -->
  <div class="preview-col">
    <div class="preview-dock">

      <!-- Interactive Preview Card -->
      <div class="preview-card">
        <div class="preview-toolbar">
          <div class="segmented-button">
            <button type="button" class="segmented-item active" data-view="ink" onclick="setViewMode('ink')">Ink Art</button>
            <button type="button" class="segmented-item" data-view="photo" onclick="setViewMode('photo')">Original Photo</button>
            <button type="button" class="segmented-item" data-view="split" onclick="setViewMode('split')">Split Wipe</button>
            <button type="button" class="segmented-item" data-view="body" onclick="setViewMode('body')">Body</button>
          </div>
          <button type="button" class="btn-secondary" id="revealBtn" onclick="animateReveal()" style="font-size:0.75rem;">
            ✨ Ink Reveal Wipe
          </button>
        </div>

        <div class="preview-stage" id="previewStage">
          <div class="preview-spinner" id="previewSpinner">
            <div class="spinner-dot"></div>
            <span>Rendering preview...</span>
          </div>

          <div class="preview-img-container">
            <img id="sourceImg" src="" alt="Source" style="display:none">
            <img id="inkImg" src="" alt="Ink Result">
          </div>
          <div class="preview-overlay-info" id="previewMeta">Ready</div>
        </div>

        <!-- Split View Scrubber -->
        <div class="split-control-bar" id="splitControlBar">
          <span style="font-size:0.8rem;font-weight:600;min-width:70px">Wipe: <span id="splitVal">50%</span></span>
          <input type="range" id="splitSlider" class="md-slider" min="0" max="100" value="50" oninput="updateSplit(this.value)">
        </div>

        <!-- Video Timeline Scrubber -->
        <div class="timeline-bar" id="timelineBar">
          <span style="font-size:0.8rem;font-weight:600;min-width:60px" id="timelineCurrent">0.0s</span>
          <input type="range" id="timelineSlider" class="md-slider" min="0" max="10" step="0.1" value="0" oninput="onTimelineScrub(this.value)">
          <span style="font-size:0.8rem;color:var(--md-sys-color-secondary);min-width:50px" id="timelineTotal">/ 0.0s</span>
        </div>
      </div>

      <!-- Render Jobs Card -->
      <div class="md-card">
        <div class="md-card-header">
          <h2 class="md-card-title">Active & Recent Jobs</h2>
          <span id="jobCountBadge" style="font-size:0.75rem;color:var(--md-sys-color-secondary);">0 jobs</span>
        </div>
        <div class="md-card-content" id="jobsList">
          <p id="noJobsMsg" style="margin:0;font-size:0.85rem;color:var(--md-sys-color-secondary)">No render jobs yet. Tweak settings above and click Start Render.</p>
        </div>
      </div>

    </div>
  </div>

</main>

<script>
const PRESET_TUNING = __PRESET_TUNING_JSON__;
const PRESET_DATA = __PRESETS_DATA_JSON__;

let currentMediaMeta = null;
let currentTimestamp = 0.0;
let currentViewMode = 'ink';
let previewDebounceTimer = null;
let activeSplitPercent = 50;

const inputPathEl = document.getElementById('inputPath');
const uploadFileEl = document.getElementById('uploadFile');
const presetSelectEl = document.getElementById('presetSelect');
const presetChips = document.querySelectorAll('.md-chip');
const detailSlider = document.getElementById('detail');
const detailVal = document.getElementById('detailVal');
const actionBtn = document.getElementById('actionBtn');

// View elements
const inkImg = document.getElementById('inkImg');
const sourceImg = document.getElementById('sourceImg');
const previewSpinner = document.getElementById('previewSpinner');
const previewMeta = document.getElementById('previewMeta');
const splitControlBar = document.getElementById('splitControlBar');
const splitSlider = document.getElementById('splitSlider');
const splitVal = document.getElementById('splitVal');
const timelineBar = document.getElementById('timelineBar');
const timelineSlider = document.getElementById('timelineSlider');
const timelineCurrent = document.getElementById('timelineCurrent');
const timelineTotal = document.getElementById('timelineTotal');
const mediaBadge = document.getElementById('mediaBadge');

// Preset chip interactions
presetChips.forEach(chip => {
  chip.addEventListener('click', () => {
    presetChips.forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    const pKey = chip.dataset.preset;
    presetSelectEl.value = pKey;

    // Show/hide custom controls card
    const customCard = document.getElementById('customControlsCard');
    if (pKey === 'custom') {
      customCard.style.display = 'block';
      customCard.open = true;
    } else {
      customCard.style.display = 'none';
    }

    // Populate human tuning sliders for this preset
    const tuning = PRESET_TUNING[pKey] || PRESET_TUNING.ultimate;
    if (tuning) {
      document.getElementById('skinSmooth').value = tuning.skin_smooth;
      document.getElementById('skinSmoothVal').textContent = Number(tuning.skin_smooth).toFixed(2);
      document.getElementById('hairBoost').value = tuning.hair_boost;
      document.getElementById('hairBoostVal').textContent = Number(tuning.hair_boost).toFixed(2);
      document.getElementById('silhouetteBoost').value = tuning.silhouette_boost;
      document.getElementById('silhouetteBoostVal').textContent = Number(tuning.silhouette_boost).toFixed(2);
      document.getElementById('subjectIsolation').value = tuning.subject_isolation;
      document.getElementById('subjectIsolationVal').textContent = Number(tuning.subject_isolation).toFixed(2);
    }

    // Classic canny disables human_aware by default
    const humanAwareEl = document.getElementById('humanAware');
    if (pKey === 'classic') {
      humanAwareEl.checked = false;
    } else if (!humanAwareEl.checked && pKey !== 'custom') {
      humanAwareEl.checked = true;
    }

    schedulePreview();
  });
});

// Slider value bubbles sync
function bindSliderBubble(id, bubbleId, decimals = 0) {
  const el = document.getElementById(id);
  const bubble = document.getElementById(bubbleId);
  if (!el || !bubble) return;
  el.addEventListener('input', () => {
    const val = parseFloat(el.value);
    bubble.textContent = decimals > 0 ? val.toFixed(decimals) : val;
    schedulePreview();
  });
}
bindSliderBubble('detail', 'detailVal', 0);
bindSliderBubble('skinSmooth', 'skinSmoothVal', 2);
bindSliderBubble('hairBoost', 'hairBoostVal', 2);
bindSliderBubble('silhouetteBoost', 'silhouetteBoostVal', 2);
bindSliderBubble('subjectIsolation', 'subjectIsolationVal', 2);
bindSliderBubble('customLowThresh', 'lowThreshVal', 0);
bindSliderBubble('customHighThresh', 'highThreshVal', 0);
bindSliderBubble('customBilateral', 'bilateralDiaVal', 0);
bindSliderBubble('customSigma', 'bilateralSigmaVal', 0);

// Color inputs sync
const customBgColor = document.getElementById('customBgColor');
const customBgHex = document.getElementById('customBgHex');
if (customBgColor && customBgHex) {
  customBgColor.addEventListener('input', () => { customBgHex.value = customBgColor.value; schedulePreview(); });
  customBgHex.addEventListener('change', () => { customBgColor.value = customBgHex.value; schedulePreview(); });
}
const customInkColor = document.getElementById('customInkColor');
const customInkHex = document.getElementById('customInkHex');
if (customInkColor && customInkHex) {
  customInkColor.addEventListener('input', () => { customInkHex.value = customInkColor.value; schedulePreview(); });
  customInkHex.addEventListener('change', () => { customInkColor.value = customInkHex.value; schedulePreview(); });
}

// All toggles and selects trigger preview
document.querySelectorAll('#renderForm input[type="checkbox"], #renderForm select').forEach(el => {
  el.addEventListener('change', schedulePreview);
});

// Debounced preview scheduler
function schedulePreview() {
  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(fetchPreview, 260);
}

// Inspect media path on blur/enter
inputPathEl.addEventListener('change', () => inspectMediaPath(inputPathEl.value));

async function inspectMediaPath(pathStr) {
  pathStr = (pathStr || '').trim().replace(/^"(.*)"$/, '$1');
  if (!pathStr) return;
  try {
    const res = await fetch(`/probe?path=${encodeURIComponent(pathStr)}`);
    if (!res.ok) {
      mediaBadge.textContent = "File not found";
      mediaBadge.style.color = "var(--md-sys-color-warn)";
      return;
    }
    currentMediaMeta = await res.json();
    mediaBadge.textContent = `${currentMediaMeta.type.toUpperCase()} -- ${currentMediaMeta.width}x${currentMediaMeta.height}` +
      (currentMediaMeta.type === 'video' ? ` (${currentMediaMeta.duration}s @ ${currentMediaMeta.fps}fps)` : '');
    mediaBadge.style.color = "var(--md-sys-color-success)";

    if (currentMediaMeta.type === 'video') {
      timelineBar.classList.add('visible');
      timelineSlider.max = currentMediaMeta.duration;
      timelineSlider.value = 0;
      currentTimestamp = 0;
      timelineCurrent.textContent = "0.0s";
      timelineTotal.textContent = `/ ${currentMediaMeta.duration}s`;
      actionBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Full Render`;
    } else {
      timelineBar.classList.remove('visible');
      currentTimestamp = 0;
      actionBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg> Process & Download Image`;
    }

    fetchSourceFrame();
    fetchPreview();
  } catch (err) {
    console.error("Probe failed:", err);
  }
}

// Sample Still loader
function loadSample(sampleName) {
  // Let the server know the path
  const samplePath = `../samples/${sampleName}`;
  inputPathEl.value = samplePath;
  inspectMediaPath(samplePath);
}

// Fetch source frame for Photo and Split views
async function fetchSourceFrame() {
  const p = inputPathEl.value.trim();
  if (!p) return;
  sourceImg.src = `/preview/source?path=${encodeURIComponent(p)}&t=${currentTimestamp}&max_dim=1280`;
}

// Timeline scrub for videos
function onTimelineScrub(val) {
  currentTimestamp = parseFloat(val);
  timelineCurrent.textContent = `${currentTimestamp.toFixed(1)}s`;
  fetchSourceFrame();
  schedulePreview();
}

function gatherAllSettings() {
  return {
    input_path: inputPathEl.value.trim(),
    timestamp: currentTimestamp,
    preset: presetSelectEl.value,
    detail: parseInt(detailSlider.value, 10),
    line_weight: parseInt(document.getElementById('lineWeight').value, 10),
    human_aware: document.getElementById('humanAware').checked,
    skin_smooth: parseFloat(document.getElementById('skinSmooth').value),
    hair_boost: parseFloat(document.getElementById('hairBoost').value),
    silhouette_boost: parseFloat(document.getElementById('silhouetteBoost').value),
    subject_isolation: parseFloat(document.getElementById('subjectIsolation').value),
    pose_lines: document.getElementById('poseLines').checked,
    face_contours: document.getElementById('faceContours').checked,
    white_balance: document.getElementById('whiteBalance').checked,
    auto_normalize: document.getElementById('autoNormalize').checked,
    temporal_denoise: document.getElementById('temporalDenoise').checked,
    dark_boost: document.getElementById('darkBoost').checked,
    custom_bg: customBgHex ? customBgHex.value : '#fff7e8',
    custom_ink: customInkHex ? customInkHex.value : '#5c2c12',
    custom_low_thresh: parseInt(document.getElementById('customLowThresh').value, 10),
    custom_high_thresh: parseInt(document.getElementById('customHighThresh').value, 10),
    custom_bilateral: parseInt(document.getElementById('customBilateral').value, 10),
    custom_sigma: parseInt(document.getElementById('customSigma').value, 10),
    body_map_overlay: (currentViewMode === 'body') || (document.getElementById('customBodyMapOverlay') && document.getElementById('customBodyMapOverlay').checked),
    max_dimension: 1280
  };
}

async function fetchPreview() {
  const p = inputPathEl.value.trim();
  if (!p) return;
  previewSpinner.classList.add('visible');

  try {
    const payload = gatherAllSettings();
    const res = await fetch('/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      previewMeta.textContent = `Preview error: ${err.error || res.statusText}`;
      previewSpinner.classList.remove('visible');
      return;
    }
    const blob = await res.blob();
    const objUrl = URL.createObjectURL(blob);
    inkImg.src = objUrl;
    const procTime = res.headers.get('X-Process-Time-Ms');
    previewMeta.textContent = `Preview rendered in ${procTime || '~'}ms`;
    applyViewModeStyles();
  } catch (err) {
    console.error("Preview fetch error:", err);
  } finally {
    previewSpinner.classList.remove('visible');
  }
}

// View Mode switcher (Ink | Photo | Split | Body)
function setViewMode(mode) {
  currentViewMode = mode;
  document.querySelectorAll('.segmented-item').forEach(b => {
    b.classList.toggle('active', b.dataset.view === mode);
  });
  if (mode === 'split') {
    splitControlBar.classList.add('visible');
  } else {
    splitControlBar.classList.remove('visible');
  }
  applyViewModeStyles();
  // If switching to or from Body view mode, re-render preview with or without body overlay
  schedulePreview();
}

function updateSplit(pct) {
  activeSplitPercent = pct;
  splitVal.textContent = `${pct}%`;
  applyViewModeStyles();
}

function applyViewModeStyles() {
  if (currentViewMode === 'photo') {
    sourceImg.style.display = 'block';
    sourceImg.style.clipPath = 'none';
    inkImg.style.display = 'none';
  } else if (currentViewMode === 'ink' || currentViewMode === 'body') {
    sourceImg.style.display = 'none';
    inkImg.style.display = 'block';
    inkImg.style.clipPath = 'none';
  } else if (currentViewMode === 'split') {
    sourceImg.style.display = 'block';
    sourceImg.style.clipPath = 'none';
    inkImg.style.display = 'block';
    inkImg.style.clipPath = `inset(0 calc(100% - ${activeSplitPercent}%) 0 0)`;
  }
}

// Animated reveal wipe (1.6s ease)
function animateReveal() {
  setViewMode('split');
  const start = performance.now();
  const duration = 1600;
  function step(now) {
    const t = Math.min((now - start) / duration, 1.0);
    // Cubic ease out
    const ease = 1 - Math.pow(1 - t, 3);
    const pct = Math.round(ease * 100);
    splitSlider.value = pct;
    updateSplit(pct);
    if (t < 1.0) {
      requestAnimationFrame(step);
    }
  }
  requestAnimationFrame(step);
}

// File upload handler
uploadFileEl.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const statusEl = document.getElementById('uploadProgressText');
  const sizeMb = (file.size / 1e6).toFixed(1);
  const startTime = Date.now();
  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();
  xhr.upload.addEventListener('progress', (ev) => {
    if (!ev.lengthComputable) return;
    const pct = (ev.loaded / ev.total * 100).toFixed(1);
    const elapsedS = (Date.now() - startTime) / 1000;
    const mbps = (ev.loaded / 1e6 / (elapsedS || 1)).toFixed(1);
    statusEl.textContent = `Uploading ${file.name}: ${(ev.loaded / 1e6).toFixed(1)} / ${sizeMb} MB (${pct}%) @ ${mbps} MB/s`;
  });
  xhr.addEventListener('load', () => {
    try {
      const data = JSON.parse(xhr.responseText);
      if (data.path) {
        inputPathEl.value = data.path;
        statusEl.textContent = `Uploaded: ${file.name} (${sizeMb} MB)`;
        inspectMediaPath(data.path);
      } else {
        statusEl.textContent = `Upload failed: ${xhr.responseText}`;
      }
    } catch (err) {
      statusEl.textContent = `Upload error (${err})`;
    }
  });
  xhr.addEventListener('error', () => { statusEl.textContent = 'Upload failed: network error'; });
  xhr.open('POST', '/upload');
  xhr.send(formData);
});

// Render Jobs Manager
const knownJobs = JSON.parse(localStorage.getItem('linearty_jobs') || '[]');
const downloadedJobs = new Set(JSON.parse(localStorage.getItem('linearty_downloaded') || '[]'));
const jobsListEl = document.getElementById('jobsList');
const noJobsMsg = document.getElementById('noJobsMsg');
const jobCountBadge = document.getElementById('jobCountBadge');

function markDownloaded(id) {
  downloadedJobs.add(id);
  localStorage.setItem('linearty_downloaded', JSON.stringify([...downloadedJobs]));
}

function saveKnownJobs() {
  localStorage.setItem('linearty_jobs', JSON.stringify(knownJobs));
  jobCountBadge.textContent = `${knownJobs.length} jobs`;
}

function jobCard(jobId) {
  let el = document.getElementById('job-' + jobId);
  if (!el) {
    noJobsMsg.style.display = 'none';
    el = document.createElement('div');
    el.id = 'job-' + jobId;
    el.className = 'md-card job-card state-running';
    el.innerHTML = `
      <div class="job-head">
        <span class="job-title">Job ${jobId}</span>
        <span class="job-state">Starting...</span>
        <button type="button" class="btn-secondary cancel-btn" style="padding:0.2rem 0.6rem;min-height:28px">Cancel</button>
      </div>
      <div class="job-progress-track"><div class="job-progress-fill"></div></div>
      <div class="job-meta">Initializing process...</div>
      <details style="margin-top:0.5rem">
        <summary style="font-size:0.75rem;cursor:pointer;color:var(--md-sys-color-secondary)">View Log</summary>
        <pre class="job-log"></pre>
      </details>
    `;
    jobsListEl.prepend(el);
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
    el.classList.remove('state-running', 'state-done', 'state-failed');
    el.classList.add(`state-${data.state}`);
    el.querySelector('.job-state').textContent = data.state.toUpperCase();

    if (data.progress) {
      const p = data.progress;
      el.querySelector('.job-progress-fill').style.width = `${Math.min(100, Math.max(0, p.pct))}%`;
      el.querySelector('.job-meta').textContent =
        `${p.done}/${p.total} frames (${p.pct.toFixed(1)}%) | ${p.fps.toFixed(1)} fps | elapsed ${p.elapsed_min.toFixed(1)}m | ETA ${p.eta_min.toFixed(1)}m`;
    }
    if (data.state === 'done') {
      el.querySelector('.job-progress-fill').style.width = '100%';
      el.querySelector('.job-meta').textContent = `Done -- Output: ${data.output_path}`;
      if (!downloadedJobs.has(jobId)) {
        markDownloaded(jobId);
        const a = document.createElement('a');
        a.href = `/jobs/${jobId}/download`;
        a.download = '';
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    } else if (data.state === 'failed') {
      el.querySelector('.job-meta').textContent = data.error || 'Render failed -- check log';
    }
    const logRes = await fetch(`/jobs/${jobId}/log`);
    if (logRes.ok) {
      const logData = await logRes.json();
      el.querySelector('.job-log').textContent = logData.log;
    }
  } catch (err) {
    // Retry on next cycle
  }
}

function pollAll() {
  knownJobs.forEach(pollJob);
}
setInterval(pollAll, 2000);
if (knownJobs.length > 0) {
  saveKnownJobs();
  pollAll();
}

// Form submit: triggers either single-image process or video job
document.getElementById('renderForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const inputP = inputPathEl.value.trim();
  if (!inputP) {
    alert("Please select or upload an input file first.");
    return;
  }

  // Single-image direct rendering
  if (currentMediaMeta && currentMediaMeta.type === 'image') {
    const payload = gatherAllSettings();
    payload.output_path = document.getElementById('outputPath').value.trim();
    payload.max_dimension = document.getElementById('maxDimension').value.trim();

    actionBtn.disabled = true;
    actionBtn.textContent = "Processing image...";
    try {
      const res = await fetch('/process-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.error) {
        alert("Image processing failed: " + data.error);
      } else {
        knownJobs.unshift(data.job_id);
        saveKnownJobs();
        jobCard(data.job_id);
        pollJob(data.job_id);
      }
    } catch (err) {
      alert("Error: " + err);
    } finally {
      actionBtn.disabled = false;
      actionBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg> Process & Download Image`;
    }
    return;
  }

  // Video render job
  const formData = new FormData(e.target);
  const params = new URLSearchParams();
  for (const [k, v] of formData.entries()) {
    params.set(k, v);
  }
  // Explicitly handle checkboxes
  const checkboxes = ['human_aware', 'pose_lines', 'face_contours', 'temporal_denoise', 'white_balance', 'auto_normalize', 'dark_boost', 'gpu_filter'];
  checkboxes.forEach(c => {
    const el = document.getElementById(c);
    if (el) params.set(c, el.checked ? 'on' : '');
  });
  const isBodyMode = (currentViewMode === 'body') || (document.getElementById('customBodyMapOverlay') && document.getElementById('customBodyMapOverlay').checked);
  params.set('body_map_overlay', isBodyMode ? 'on' : '');
  params.set('settings_json', JSON.stringify(gatherAllSettings()));

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

// Auto-load portrait sample on initial launch if input is blank
window.addEventListener('DOMContentLoaded', () => {
  if (!inputPathEl.value.trim()) {
    loadSample('portrait.jpg');
  }
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

    print(f"Linearty Studio web UI at http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)
