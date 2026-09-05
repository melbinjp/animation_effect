"""Native CLI for Linearty's ink pipeline: decode a video, run every frame
through pipeline.py (plus optional human.py segmentation), re-encode, remux
audio. Built for a dedicated server process, not a browser tab, so none of
worker.js's WASM-heap-driven limits apply — see the plan doc's "explicitly
not limiting" table. No default resolution/quality cap, no fixed worker
ceiling, no checkpoint/resume machinery (a crashed process here costs a
rerun, not an unrecoverable multi-hour browser session).

Parallelism model: the source is split into `--workers` contiguous frame
segments (default os.cpu_count()), each handed to one multiprocessing.Pool
worker as a single starmap call. Because one worker processes its entire
segment sequentially inside one function call, MediaPipe VIDEO mode's
strictly-increasing-timestamp requirement is satisfied automatically — no
cross-worker coordination needed. Segments are encoded to temp .mp4 files
and concatenated (stream copy, no re-encode) at the end, then audio is
muxed back in from the original input in one final plain ffmpeg pass.

On Windows, multiprocessing always uses "spawn" (never "fork"): every
worker re-imports this module fresh in a new process, so the __main__
guard below is mandatory and the worker function must be a plain
top-level, picklable callable — not a closure or a bound method.
"""

import os

# Must run before `import human`/`cv2`/mediapipe pull in TFLite's XNNPACK
# CPU delegate, which auto-sizes its own internal thread pool independent
# of cv2.setNumThreads(1) below (that call only affects OpenCV's threading,
# not TFLite's). Confirmed live: going from 8 to 14 worker PROCESSES made
# throughput WORSE (5.5fps -> ~2.2fps) on this 18-core machine, consistent
# with each MediaPipe-based worker already running its own multi-threaded
# CPU inference -- more processes multiplies real OS thread count far past
# what 18 cores can serve, causing oversubscription/context-switch thrash
# rather than more throughput. These env vars are the only lever available
# since the Python Tasks API (mediapipe.tasks.python.BaseOptions) exposes
# no num_threads parameter to set this directly (checked live: its
# constructor only takes model_asset_path/model_asset_buffer/delegate).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import argparse
import json
import math
import shutil
import subprocess
import sys
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


def _encoder_extra_args(encoder):
    # A 90s 1080p ink-line render came out to ~380MB against a 43MB source,
    # which first looked like a missing-quality-flag bug on h264_qsv (ink
    # output looks visually simpler than the source footage). Tested against
    # the actual real ink pixel content (not a synthetic proxy) before
    # committing to that theory: h264_qsv's plain default (no quality flag)
    # produced output roughly the SAME size as libx264 -crf 20 on identical
    # frames, and adding -global_quality 20 made it ~40% LARGER, not
    # smaller -- the opposite of what a "fix" should do. So the real
    # explanation is that ink/line-art content (hard binary edges, plus
    # frame-to-frame edge jitter from source sensor noise amplified through
    # Canny) is inherently harder for H.264's block-DCT+motion-prediction
    # model to compress than smooth natural video, regardless of encoder --
    # the source's small size reflects how well natural video compresses,
    # not a bitrate target the ink output should also hit. h264_qsv is left
    # at its plain default rather than an unverified "fix" that measurably
    # made real content worse.
    if encoder == "h264_nvenc":
        # -rc vbr alongside -cq is the standard, well-documented pairing for
        # reliable nvenc constant-quality behavior (unlike qsv's
        # -global_quality above, this one isn't just an untested guess).
        return ["-preset", "p4", "-rc", "vbr", "-cq", "20"]
    if encoder == "h264_vaapi":
        return ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=nv12,hwupload"]
    if encoder == "h264_qsv":
        return ["-preset", "medium"]
    return ["-preset", "medium", "-crf", "20"]


def _encoder_actually_works(encoder, extra_args):
    """ffmpeg -encoders only lists what the binary was compiled with, not
    what actually initializes on this machine (e.g. h264_nvenc can be
    compiled in on a box with no NVIDIA GPU or driver present at all).
    Confirmed live: a compiled-in-but-unusable nvenc entry made 'auto' pick
    it, and every segment's encode then failed at runtime. Verify instead
    with one real tiny test encode."""
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


def detect_encoder(preferred):
    """preferred: 'auto' | 'nvenc' | 'vaapi' | 'qsv' | 'libx264' | any raw
    ffmpeg encoder name. 'auto' probes candidates with a real test encode
    (see _encoder_actually_works) and picks the first that actually
    initializes, falling back to libx264 if none do."""
    if preferred == "libx264":
        return "libx264", _encoder_extra_args("libx264")

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
            extra = _encoder_extra_args(enc)
            if _encoder_actually_works(enc, extra):
                return enc, extra
            print(f"Encoder '{enc}' is compiled into ffmpeg but failed a live test encode; skipping.", file=sys.stderr)

    if preferred not in ("auto",) and preferred not in name_map:
        raise RuntimeError(f"Requested encoder '{preferred}' not found in ffmpeg -encoders output")

    return "libx264", _encoder_extra_args("libx264")


def _resolve_dims(width, height, max_dimension):
    if not max_dimension or max(width, height) <= max_dimension:
        return width, height
    scale = max_dimension / max(width, height)
    # even dimensions: required by yuv420p and most hardware encoders
    return (max(2, round(width * scale) // 2 * 2), max(2, round(height * scale) // 2 * 2))


def _build_settings(args):
    preset_name = args.preset
    preset = get_preset(preset_name)
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
    }
    if preset_name == "custom":
        settings.update({"use_bilateral": True, "use_gaussian": False, "use_median": False})
    if args.human_aware:
        settings.update(get_human_tuning(preset_name))
    return settings


def _init_worker(progress_array, worker_slot_counter, stagger_seconds):
    # Claim a slot number and, if staggering, sleep proportionally to it
    # BEFORE this worker's _process_segment call starts spawning its own
    # ffmpeg decode/encode subprocesses. Root cause this addresses: spawning
    # --workers processes each of which also spawns 2 ffmpeg children means
    # up to 3x --workers process-creation calls landing in a tight burst;
    # confirmed live on this machine that a large burst (18 workers, ~54
    # process creations) can hit a Windows-specific ceiling
    # (`OSError: [WinError 1450] Insufficient system resources`) well before
    # actual RAM or CPU are exhausted. Spreading worker startup out removes
    # the burst without touching any per-frame processing -- same pixels,
    # same settings, just not all-at-once.
    with worker_slot_counter.get_lock():
        slot = worker_slot_counter.value
        worker_slot_counter.value += 1
    if stagger_seconds > 0 and slot > 0:
        time.sleep(slot * stagger_seconds)

    import cv2
    cv2.setNumThreads(1)  # avoid N processes x M internal cv2 threads oversubscribing cores
    global _progress
    _progress = progress_array


def _process_segment(seg_idx, input_path, start_frame, end_frame, fps, src_w, src_h, out_w, out_h,
                      settings, encoder, encoder_args, tmp_dir):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)

    human_aware = settings.get("human_aware")
    want_landmarks = bool(settings.get("pose_lines") or settings.get("face_contours"))
    if human_aware:
        human_mod.ensure_human(landmarks=want_landmarks, mode="VIDEO")

    start_time = start_frame / fps
    num_frames = end_frame - start_frame
    frame_size = src_w * src_h * 4  # RGBA at native decode resolution

    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_time:.6f}", "-i", input_path,
        "-frames:v", str(num_frames),
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-",
    ]
    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    seg_path = os.path.join(tmp_dir, f"seg{seg_idx:05d}.mp4")
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

            # Pass the RGB conversion already computed above (when
            # human-aware) so process_frame doesn't redo the identical
            # cv2.cvtColor call a second time -- same bytes either way.
            out_rgb = pipeline.process_frame(rgba, frame_settings, rgb=rgb_for_human)
            if (out_w, out_h) != (src_w, src_h):
                out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

            try:
                encode_proc.stdin.write(np.ascontiguousarray(out_rgb).tobytes())
            except (BrokenPipeError, OSError):
                # The encoder process died mid-stream (e.g. a hardware encoder
                # ffmpeg lists as compiled-in but can't actually init on this
                # machine -- no GPU present). Stop feeding it; the returncode
                # check below drives the same libx264 fallback/raise path as
                # a clean non-zero exit.
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
            fallback_args = _encoder_extra_args("libx264")
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
    parser = argparse.ArgumentParser(description="Linearty native ink-pipeline CLI: process a video at full hardware speed, no browser limits.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--preset", default="ultimate", help="Style preset (default: ultimate)")
    parser.add_argument("--detail", type=int, default=62, help="Edge detail slider, matches the site's default of 62")
    parser.add_argument("--line-weight", type=int, default=1, dest="line_weight")
    parser.add_argument("--white-balance", action="store_true", dest="white_balance")
    parser.add_argument("--no-auto-normalize", action="store_true", dest="no_auto_normalize")
    parser.add_argument("--dark-boost", action="store_true", dest="dark_boost")
    parser.add_argument("--human-aware", dest="human_aware", action="store_true", default=None)
    parser.add_argument("--no-human-aware", dest="human_aware", action="store_false")
    parser.add_argument("--pose-lines", action="store_true", dest="pose_lines")
    parser.add_argument("--face-contours", action="store_true", dest="face_contours")
    parser.add_argument("--max-dimension", type=int, default=None,
                         help="Opt-in resolution cap. Omit for full source resolution (the default) -- "
                              "unlike the browser version, nothing is capped unless you ask for it.")
    parser.add_argument("--fps", type=float, default=None, help="Override output fps (default: source fps)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                         help="Parallel worker processes (default: all CPU cores)")
    parser.add_argument("--worker-stagger", type=float, default=1.0, dest="worker_stagger",
                         help="Seconds to stagger each worker's startup by (worker N waits N * this "
                              "long before spawning its own ffmpeg subprocesses). Confirmed on real "
                              "hardware that launching many workers (each of which also spawns 2 "
                              "ffmpeg children) in one burst can hit a Windows process-creation limit "
                              "well before RAM/CPU are actually exhausted; spreading startup out avoids "
                              "it at the cost of a few extra seconds before full parallelism kicks in "
                              "-- negligible next to render time. Set to 0 to disable.")
    parser.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "vaapi", "qsv", "libx264"],
                         help="Video encoder. 'auto' tries hardware encoders in order and falls back to libx264.")
    parser.add_argument("--gpu-filter", action="store_true", dest="gpu_filter",
                         help="Opportunistic OpenCV CUDA use for the ink filter itself. Only takes effect if the "
                              "installed opencv build has CUDA support (the standard pip wheel does not); "
                              "silently ignored otherwise.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.human_aware is None:
        args.human_aware = args.preset != "classic"  # mirrors script.js's own default

    if args.gpu_filter:
        import cv2
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            print("--gpu-filter requested but no CUDA-enabled OpenCV build/device found; continuing on CPU.", file=sys.stderr)
            args.gpu_filter = False

    info = probe_video(args.input)
    fps = args.fps or info["fps"]
    out_w, out_h = _resolve_dims(info["width"], info["height"], args.max_dimension)
    settings = _build_settings(args)

    encoder, encoder_args = detect_encoder(args.encoder)
    print(f"Input: {info['width']}x{info['height']} @ {info['fps']:.3f}fps, "
          f"{info['total_frames']} frames, audio={info['has_audio']}")
    print(f"Output: {out_w}x{out_h} @ {fps:.3f}fps, preset={args.preset}, human_aware={args.human_aware}, "
          f"encoder={encoder}, workers={args.workers}")

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
                     initargs=(progress_array, worker_slot_counter, args.worker_stagger)) as pool:
            async_result = pool.map_async(_process_segment_star, job_args)
            start_wall = time.time()
            # One newline-terminated line per update (not an in-place \r bar):
            # this is meant to be watched via `tail -f` / `Get-Content -Wait`
            # on a redirected log file as much as in an interactive terminal,
            # and \r updates render as one unreadable run-on line in a file.
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
