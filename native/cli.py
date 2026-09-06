"""Command-line interface for the Linearty video inking pipeline.
Decodes video into contiguous frame segments, applies stylistic and human-aware
filters via multiprocessing workers, and re-encodes with audio muxing.
"""

import math
import os
import sys


def _real_core_count():
    """Returns available CPU core count, respecting Linux cgroup/affinity limits when present."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


# Extract --threads-per-worker early to configure OpenMP/TFLite threading before library initialization.
_cores = _real_core_count()
_threads_arg = None
for _i, _a in enumerate(sys.argv):
    if _a == "--threads-per-worker" and _i + 1 < len(sys.argv):
        _threads_arg = sys.argv[_i + 1]
    elif _a.startswith("--threads-per-worker="):
        _threads_arg = _a.split("=", 1)[1]
_default_threads_per_worker = max(1, round(math.sqrt(_cores)))
_threads_per_worker = int(_threads_arg) if _threads_arg else _default_threads_per_worker

# Configure thread pool limits for OpenMP and TensorFlow Lite runtimes.
os.environ.setdefault("OMP_NUM_THREADS", str(_threads_per_worker))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_threads_per_worker))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(_threads_per_worker))

import argparse
import json
import shutil
import subprocess
import tempfile
import time

import human as human_mod
import pipeline
from presets import get_preset, get_human_tuning

# Set by _init_worker in each worker process: a shared-memory int array, one
# slot per segment, so the main process can render a live progress bar
# without waiting for a whole (thousands-of-frames) segment to finish.
_progress = None


def probe_video(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    duration = float(data.get("format", {}).get("duration") or 0)

    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames.isdigit() and int(nb_frames) > 0:
        total_frames = int(nb_frames)
    elif duration > 0:
        total_frames = max(1, round(duration * fps))
    else:
        raise RuntimeError(f"Could not determine frame count or duration for {path}")

    has_audio_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path]
    has_audio = bool(subprocess.run(has_audio_cmd, capture_output=True, text=True).stdout.strip())

    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames, "has_audio": has_audio}


# Named quality presets mapped to constant rate factor (CRF) values.
QUALITY_PRESETS = {
    "indistinguishable": 18,
    "optimized": 21,
    "balanced": 24,
    "small": 28,
    "aggressive": 32,
    "maximum": 40,
}


def _encoder_extra_args(encoder, crf):
    if encoder == "h264_nvenc":
        return ["-preset", "p4", "-rc", "vbr", "-cq", str(crf)]
    if encoder == "h264_vaapi":
        return ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=nv12,hwupload"]
    if encoder == "h264_qsv":
        return ["-preset", "medium"]
    return ["-preset", "medium", "-crf", str(crf)]


def _encoder_actually_works(encoder, extra_args):
    """Executes a 2-frame test encode to verify that the encoder initializes on current hardware."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1:r=5",
        "-frames:v", "2",
        "-c:v", encoder, *extra_args,
        "-pix_fmt", "yuv420p",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def detect_encoder(preferred, crf):
    """Detects and returns an operational encoder and its parameter list, falling back to libx264."""
    if preferred == "libx264":
        return "libx264", _encoder_extra_args("libx264", crf)

    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    available = result.stdout

    name_map = {"nvenc": "h264_nvenc", "vaapi": "h264_vaapi", "qsv": "h264_qsv"}
    if preferred in name_map:
        candidates = [name_map[preferred]]
    elif preferred == "auto":
        candidates = ["h264_nvenc", "h264_vaapi", "h264_qsv"]
    else:
        candidates = [preferred]  # a raw ffmpeg encoder name passed directly

    for enc in candidates:
        if enc in available:
            extra = _encoder_extra_args(enc, crf)
            if _encoder_actually_works(enc, extra):
                return enc, extra
            print(f"Encoder '{enc}' is compiled into ffmpeg but failed a live test encode; skipping.", file=sys.stderr)

    if preferred not in ("auto",) and preferred not in name_map:
        raise RuntimeError(f"Requested encoder '{preferred}' not found in ffmpeg -encoders output")

    return "libx264", _encoder_extra_args("libx264", crf)


def _resolve_dims(width, height, max_dimension):
    if not max_dimension or max(width, height) <= max_dimension:
        return width, height
    scale = max_dimension / max(width, height)
    # even dimensions: required by yuv420p and most hardware encoders
    return (max(2, round(width * scale) // 2 * 2), max(2, round(height * scale) // 2 * 2))


def _build_settings(args):
    preset_name = args.preset
    preset = dict(get_preset(preset_name))
    settings = {
        "preset": preset,
        "engine": preset["engine"],
        "detail": args.detail,
        "custom_mode": preset_name == "custom",
        "white_balance": args.white_balance,
        "auto_normalize": not args.no_auto_normalize,
        "dark_boost": args.dark_boost,
        "dark_boost_clip": 2.5,
        "clean_speckles": preset.get("clean_speckles", False),
        "clean_speckles_intensity": 1,
        "merge_double_edge": preset.get("merge_double_edge", False),
        "merge_double_edge_intensity": 2,
        "line_weight": args.line_weight,
        "color_edges": False,
        "human_aware": args.human_aware,
        "pose_lines": args.pose_lines,
        "face_contours": args.face_contours,
        "temporal_denoise": args.temporal_denoise,
        "body_map_overlay": args.body_map_overlay or preset.get("body_map_overlay", False),
        # Not an ink-pipeline setting -- bundled into settings purely so
        # _process_segment's hardware-encoder-failed fallback (which only
        # has settings/encoder/encoder_args in scope, not args) can rebuild
        # libx264 args with the right CRF instead of a hardcoded one.
        "crf": args.crf,
    }
    if preset_name == "custom":
        settings.update({"use_bilateral": True, "use_gaussian": False, "use_median": False})
    if args.human_aware:
        settings.update(get_human_tuning(preset_name))

    if getattr(args, "settings_json", None):
        try:
            if os.path.isfile(args.settings_json):
                with open(args.settings_json, "r", encoding="utf-8") as f:
                    extra = json.load(f)
            else:
                extra = json.loads(args.settings_json)
            if isinstance(extra, dict):
                if "preset" in extra and isinstance(extra["preset"], dict):
                    settings["preset"].update(extra["preset"])
                    del extra["preset"]
                settings.update(extra)
        except Exception as e:
            print(f"Warning: Failed to parse --settings-json ({e})", file=sys.stderr)

    return settings


def _init_worker(progress_array, worker_slot_counter, stagger_seconds, threads_per_worker, total_workers):
    # Stagger worker initialization to avoid resource exhaustion from concurrent process creation bursts.
    with worker_slot_counter.get_lock():
        slot = worker_slot_counter.value
        worker_slot_counter.value += 1
    if stagger_seconds > 0 and slot > 0:
        time.sleep(slot * stagger_seconds)

    # Bind worker process to a dedicated core slice when affinity control is available.
    try:
        full_affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        full_affinity = None  # not available on Windows; no-op there
    if full_affinity and total_workers > 0:
        chunk = max(1, len(full_affinity) // total_workers)
        start = (slot % total_workers) * chunk
        my_cores = full_affinity[start:start + chunk] or full_affinity
        try:
            os.sched_setaffinity(0, set(my_cores))
        except (AttributeError, OSError):
            pass

    import cv2
    cv2.setNumThreads(threads_per_worker)
    global _progress
    _progress = progress_array


def _process_segment(seg_idx, input_path, start_frame, end_frame, fps, src_w, src_h, out_w, out_h,
                      settings, encoder, encoder_args, tmp_dir):
    import cv2
    import numpy as np
    from hw_detect import HAS_GPU

    human_aware = settings.get("human_aware")
    want_landmarks = bool(settings.get("pose_lines") or settings.get("face_contours"))
    if human_aware:
        human_mod.ensure_human(landmarks=want_landmarks, mode="VIDEO")

    start_time = start_frame / fps
    num_frames = end_frame - start_frame
    frame_size = src_w * src_h * 4  # RGBA at native decode resolution

    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error"
    ]
    if HAS_GPU:
        decode_cmd.extend(["-hwaccel", "cuda"])
    
    decode_cmd.extend([
        "-ss", f"{start_time:.6f}", "-i", input_path,
        "-frames:v", str(num_frames),
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-",
    ])
    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    seg_path = os.path.join(tmp_dir, f"seg{seg_idx:05d}.mp4")
    
    if HAS_GPU and encoder == "libx264":
        encoder = "h264_nvenc"
        encoder_args = ["-preset", "p6", "-b:v", "0", "-cq", "20"]

    encode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.6f}",
        "-i", "-",
        "-c:v", encoder, *encoder_args,
        "-pix_fmt", "yuv420p",
        seg_path,
    ]
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frames_written = 0
    prev_gray = None  # carried sequentially across this segment's frames -- see temporal_denoise below
    try:
        for _ in range(num_frames):
            raw = decode_proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 4))

            frame_settings = settings
            rgb_for_human = None
            if human_aware:
                rgb_for_human = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
                human_result = human_mod.infer_human(rgb_for_human, src_w, src_h, settings, use_video_mode=True)
                frame_settings = dict(settings)
                if human_result:
                    frame_settings["class_mask"] = human_result["class_mask"]
                    frame_settings["extra_lines"] = human_result["extra_lines"]
                else:
                    frame_settings["human_aware"] = False

            if settings.get("temporal_denoise"):
                if frame_settings is settings:
                    frame_settings = dict(settings)
                frame_settings["prev_gray"] = prev_gray

            # Reuse pre-computed RGB buffer to avoid redundant color conversion.
            out_rgb, prev_gray = pipeline.process_frame(rgba, frame_settings, rgb=rgb_for_human)
            if (out_w, out_h) != (src_w, src_h):
                out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

            try:
                encode_proc.stdin.write(np.ascontiguousarray(out_rgb).tobytes())
            except (BrokenPipeError, OSError):
                # Encoder terminated prematurely; stop streaming frames.
                break
            frames_written += 1
            if _progress is not None:
                _progress[seg_idx] = frames_written
    finally:
        decode_proc.stdout.close()
        decode_proc.wait()
        encode_proc.stdin.close()
        stderr = encode_proc.stderr.read()
        encode_proc.wait()

    if encode_proc.returncode != 0:
        if encoder != "libx264":
            fallback_args = _encoder_extra_args("libx264", settings["crf"])
            return _process_segment(seg_idx, input_path, start_frame, end_frame, fps, src_w, src_h,
                                     out_w, out_h, settings, "libx264", fallback_args, tmp_dir)
        raise RuntimeError(f"Segment {seg_idx} encode failed ({frames_written} frames written): "
                            f"{stderr.decode(errors='replace')}")

    return seg_path


def _process_segment_star(args_tuple):
    return _process_segment(*args_tuple)


def build_segments(total_frames, workers):
    frames_per_seg = math.ceil(total_frames / workers)
    segments = []
    start = 0
    while start < total_frames:
        end = min(start + frames_per_seg, total_frames)
        segments.append((start, end))
        start = end
    return segments


def concat_segments(seg_paths, tmp_dir, out_path):
    list_path = os.path.join(tmp_dir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in seg_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", out_path,
    ]
    subprocess.run(cmd, check=True)


def mux_audio(video_only_path, original_input, out_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_only_path, "-i", original_input,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Command-line video inking and stylization tool.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--preset", default="ultimate", help="Style preset (default: ultimate)")
    parser.add_argument("--detail", type=int, default=62, help="Edge detail level (default: 62)")
    parser.add_argument("--line-weight", type=int, default=1, dest="line_weight", help="Line stroke dilation weight")
    parser.add_argument("--white-balance", action="store_true", dest="white_balance", help="Apply gray-world white balance")
    parser.add_argument("--no-auto-normalize", action="store_true", dest="no_auto_normalize", help="Disable adaptive luminance normalization")
    parser.add_argument("--dark-boost", action="store_true", dest="dark_boost", help="Enhance contrast in shadow regions via CLAHE")
    parser.add_argument("--human-aware", dest="human_aware", action="store_true", default=None, help="Enable MediaPipe human segmentation")
    parser.add_argument("--no-human-aware", dest="human_aware", action="store_false", help="Disable MediaPipe human segmentation")
    parser.add_argument("--pose-lines", action="store_true", dest="pose_lines", help="Render pose landmark lines")
    parser.add_argument("--face-contours", action="store_true", dest="face_contours", help="Render face contour landmarks")
    parser.add_argument("--temporal-denoise", action="store_true", dest="temporal_denoise",
                         help="Motion-adaptive temporal smoothing on pre-Canny luminance to reduce edge jitter.")
    parser.add_argument("--max-dimension", type=int, default=None,
                         help="Maximum output dimension (width or height) in pixels; preserves aspect ratio.")
    parser.add_argument("--fps", type=float, default=None, help="Output frame rate (default: source frame rate)")
    default_workers = max(1, _cores // _threads_per_worker)
    parser.add_argument("--workers", type=int, default=default_workers,
                         help=f"Number of parallel worker processes (default: {default_workers}).")
    parser.add_argument("--threads-per-worker", type=int, default=_threads_per_worker,
                         dest="threads_per_worker",
                         help=f"Internal thread count per worker process (default: {_default_threads_per_worker}).")
    parser.add_argument("--worker-stagger", type=float, default=1.0, dest="worker_stagger",
                         help="Delay in seconds between starting consecutive worker processes (default: 1.0).")
    parser.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "vaapi", "qsv", "libx264"],
                         help="Video encoder: 'auto', 'nvenc', 'vaapi', 'qsv', or 'libx264' (default: auto).")
    quality_choices = ", ".join(f"{name} (CRF {crf})" for name, crf in QUALITY_PRESETS.items())
    parser.add_argument("--quality", choices=list(QUALITY_PRESETS), default="balanced",
                         help=f"Output quality tier: {quality_choices} (default: balanced).")
    parser.add_argument("--crf", type=int, default=None,
                         help="Explicit Constant Rate Factor (CRF) override; lower values increase quality.")
    parser.add_argument("--body-map-overlay", action="store_true", dest="body_map_overlay",
                         help="Bake colorized body map overlay into output.")
    parser.add_argument("--settings-json", dest="settings_json", default=None,
                         help="JSON string or file path containing pipeline settings overrides.")
    parser.add_argument("--gpu-filter", action="store_true", dest="gpu_filter",
                         help="Enable CUDA acceleration for OpenCV filters if supported by build.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.human_aware is None:
        args.human_aware = args.preset != "classic"
    if args.crf is None:
        args.crf = QUALITY_PRESETS[args.quality]

    if args.gpu_filter:
        import cv2
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            print("--gpu-filter requested but no CUDA-enabled OpenCV build/device found; continuing on CPU.", file=sys.stderr)
            args.gpu_filter = False

    info = probe_video(args.input)
    fps = args.fps or info["fps"]
    out_w, out_h = _resolve_dims(info["width"], info["height"], args.max_dimension)
    settings = _build_settings(args)

    from hw_detect import HAS_GPU, log_hardware_status, get_vram_info
    log_hardware_status()
    if HAS_GPU:
        free_gb, total_gb = get_vram_info()
        if total_gb > 0:
            gpu_max_workers = max(1, int(total_gb // 2.8))
            if args.workers > gpu_max_workers:
                print(f"[INFO] GPU VRAM: {total_gb:.1f} GB ({free_gb:.1f} GB free). Scaling worker pool from {args.workers} to {gpu_max_workers} to maximize throughput safely.", flush=True)
                args.workers = gpu_max_workers
            else:
                print(f"[INFO] GPU VRAM: {total_gb:.1f} GB ({free_gb:.1f} GB free). Running with {args.workers} worker(s).", flush=True)
        else:
            if args.workers > 4:
                print(f"[INFO] Hardware Acceleration active: capping workers at 4 (was {args.workers}) to prevent GPU VRAM exhaustion.", flush=True)
                args.workers = 4

    encoder, encoder_args = detect_encoder(args.encoder, args.crf)
    print(f"Input: {info['width']}x{info['height']} @ {info['fps']:.3f}fps, "
          f"{info['total_frames']} frames, audio={info['has_audio']}")
    print(f"Output: {out_w}x{out_h} @ {fps:.3f}fps, preset={args.preset}, human_aware={args.human_aware}, "
          f"encoder={encoder}, crf={args.crf}, workers={args.workers}")

    segments = build_segments(info["total_frames"], args.workers)
    print(f"Split into {len(segments)} segments across {args.workers} worker(s)")

    tmp_dir = tempfile.mkdtemp(prefix="linearty_native_")
    try:
        job_args = [
            (i, args.input, start, end, fps, info["width"], info["height"], out_w, out_h,
             settings, encoder, encoder_args, tmp_dir)
            for i, (start, end) in enumerate(segments)
        ]

        import multiprocessing as mp
        progress_array = mp.Array("i", len(segments))
        worker_slot_counter = mp.Value("i", 0)
        total_frames = info["total_frames"]
        with mp.Pool(processes=args.workers, initializer=_init_worker,
                     initargs=(progress_array, worker_slot_counter, args.worker_stagger,
                               args.threads_per_worker, args.workers)) as pool:
            async_result = pool.map_async(_process_segment_star, job_args)
            start_wall = time.time()
            # Print periodic progress updates.
            while not async_result.ready():
                async_result.wait(5)
                done = sum(progress_array)
                elapsed = time.time() - start_wall
                pct = 100 * done / max(1, total_frames)
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (total_frames - done) / rate if rate > 0 else 0
                print(f"Progress: {done}/{total_frames} frames ({pct:5.1f}%) | "
                      f"{rate:6.1f} fps | elapsed {elapsed / 60:5.1f}m | ETA {eta_s / 60:5.1f}m",
                      flush=True)
            seg_paths = async_result.get()
        print(f"All {len(segments)} segments done")

        video_only_path = os.path.join(tmp_dir, "video_only.mp4")
        concat_segments(seg_paths, tmp_dir, video_only_path)

        if info["has_audio"]:
            mux_audio(video_only_path, args.input, args.output)
        else:
            shutil.copyfile(video_only_path, args.output)

        print(f"Done: {args.output}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
