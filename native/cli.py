"""Native CLI for Linearty's ink pipeline: decode a video, run every frame
through pipeline.py (plus optional human.py segmentation), re-encode, remux
audio. Built for a dedicated server process, not a browser tab, so none of
worker.js's WASM-heap-driven limits apply — see the plan doc's "explicitly
not limiting" table. No default resolution/quality cap, no fixed worker
ceiling, no checkpoint/resume machinery (a crashed process here costs a
rerun, not an unrecoverable multi-hour browser session).

Parallelism model: the source is split into `--workers` contiguous frame
segments (default: real cores / --threads-per-worker, both dynamically
detected -- see _real_core_count()'s docstring for why this isn't just
os.cpu_count()), each handed to one multiprocessing.Pool worker as a single
starmap call, and each worker's cv2/MediaPipe use --threads-per-worker
threads internally rather than being restricted to one (validated live:
fewer, properly multi-threaded workers substantially beats more
single-threaded ones -- see the --threads-per-worker help text). Because
one worker processes its entire segment sequentially inside one function
call, MediaPipe VIDEO mode's strictly-increasing-timestamp requirement is
satisfied automatically — no cross-worker coordination needed. Segments are
encoded to temp .mp4 files and concatenated (stream copy, no re-encode) at
the end, then audio is muxed back in from the original input in one final
plain ffmpeg pass.

On Windows, multiprocessing always uses "spawn" (never "fork"): every
worker re-imports this module fresh in a new process, so the __main__
guard below is mandatory and the worker function must be a plain
top-level, picklable callable — not a closure or a bound method.
"""

import math
import os
import sys


def _real_core_count():
    """os.cpu_count() reports the HOST's total logical CPUs inside a
    container, not what's actually allocated to it -- confirmed live on a
    rented pod where os.cpu_count() said 128 (the host) while the container
    was only given 16, and a --workers default trusting the former spawned
    128 MediaPipe worker processes fighting over 16 real cores (severe
    oversubscription -- load average 38-65 against a 16-core budget,
    progress stalled outright). os.sched_getaffinity(0) respects the
    container's actual cgroup/affinity-limited allocation on Linux; it
    doesn't exist on Windows/macOS, where os.cpu_count() is already correct
    since those platforms aren't usually run inside a shared-host container
    the way this script's target rented-Linux-pod use case is."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


# Read --threads-per-worker (or fall back to the same dynamic default main()
# computes below) here, before any heavy import, since OMP/TF thread-count
# env vars only take effect if set before mediapipe/TFLite's XNNPACK CPU
# delegate initializes -- too late once `import human` below has run. This
# is a plain sys.argv scan rather than argparse specifically so the real
# argparse pass in parse_args() (which also validates everything else)
# doesn't have to run twice.
_cores = _real_core_count()
_threads_arg = None
for _i, _a in enumerate(sys.argv):
    if _a == "--threads-per-worker" and _i + 1 < len(sys.argv):
        _threads_arg = sys.argv[_i + 1]
    elif _a.startswith("--threads-per-worker="):
        _threads_arg = _a.split("=", 1)[1]
_default_threads_per_worker = max(1, round(math.sqrt(_cores)))
_threads_per_worker = int(_threads_arg) if _threads_arg else _default_threads_per_worker

# Must run before `import human`/`cv2`/mediapipe pull in TFLite's XNNPACK
# CPU delegate, which auto-sizes its own internal thread pool independent of
# cv2.setNumThreads() below (that call only affects OpenCV's own threading,
# not TFLite's) -- these env vars are the only lever for TFLite specifically,
# since the Python Tasks API (mediapipe.tasks.python.BaseOptions) exposes no
# num_threads parameter directly (checked live: its constructor only takes
# model_asset_path/model_asset_buffer/delegate). Previously hardcoded to "1"
# to avoid N-single-threaded-processes oversubscription; confirmed live that
# this was actively counterproductive once the architecture moved to fewer,
# properly multi-threaded workers -- a standalone benchmark script with NO
# thread cap hit ~51fps aggregate (4 workers x 4 threads on 16 cores) versus
# this file's own 10-single-threaded-workers-capped-at-1 getting ~6fps on
# the same class of hardware. Scale the cap to match --threads-per-worker
# instead of always forcing it to 1.
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


# Named quality tiers, same idea as the Linearty Android app's
# encoder_quality_mode setting (SAME_AS_INPUT/INDISTINGUISHABLE/BALANCED/
# SMALL_SIZE -> CRF 21/18/24/28) -- CRF is a quality TARGET, not a size
# target, so naming tiers by what they look like (not an arbitrary number)
# is the right UI, and reusing that app's exact tier boundaries where they
# overlap keeps the two products consistent rather than inventing new
# numbers for the same idea. Extended past that app's ceiling of 28 with an
# "aggressive" tier, since ink/line-art content (flat color, hard edges, no
# photographic gradient/texture) tolerates far more compression before
# looking different -- confirmed live on real ink output: CRF 20 (this
# file's old hardcoded default) -> 33.0MB for a 90s 1080p clip; CRF 28 (the
# Android app's own max) -> 15.8MB (52% smaller); CRF 32 -> 9.1MB (72%
# smaller). Higher than that (36/40 measured 86-93% smaller) risks visibly
# softening the actual lines -- the one thing that must stay crisp for this
# style -- so isn't offered as a named tier; use --crf directly if you want
# to push further after checking output quality yourself.
QUALITY_PRESETS = {
    "indistinguishable": 18,
    "optimized": 21,
    "balanced": 24,
    "small": 28,
    "aggressive": 32,
    "maximum": 40,  # user-confirmed by eye against real output: still looks fine at CRF 40 (93% smaller than CRF 20)
}


def _encoder_extra_args(encoder, crf):
    # h264_qsv and h264_vaapi have no validated quality-target flag (see
    # detect_encoder's docstring / this file's git history: h264_qsv's
    # -global_quality measured 40% LARGER than its plain default on real
    # ink content, the opposite of what a quality flag should do, and
    # h264_vaapi's -qp was never validated at all) -- --quality/--crf only
    # actually changes output size on libx264 and h264_nvenc, the two
    # encoders with a real, tested quality control. This matters less in
    # practice than it sounds: a CPU-only pod (this project's actual
    # deployment target so far) always falls back to libx264 anyway.
    if encoder == "h264_nvenc":
        # -rc vbr alongside -cq is the standard, well-documented pairing for
        # reliable nvenc constant-quality behavior (unlike qsv's
        # -global_quality above, this one isn't just an untested guess).
        # nvenc's cq scale isn't numerically identical to x264's crf, but
        # tracks the same direction closely enough to reuse the same value.
        return ["-preset", "p4", "-rc", "vbr", "-cq", str(crf)]
    if encoder == "h264_vaapi":
        return ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=nv12,hwupload"]
    if encoder == "h264_qsv":
        return ["-preset", "medium"]
    return ["-preset", "medium", "-crf", str(crf)]


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


def detect_encoder(preferred, crf):
    """preferred: 'auto' | 'nvenc' | 'vaapi' | 'qsv' | 'libx264' | any raw
    ffmpeg encoder name. 'auto' probes candidates with a real test encode
    (see _encoder_actually_works) and picks the first that actually
    initializes, falling back to libx264 if none do."""
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
        "temporal_denoise": args.temporal_denoise,
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
    return settings


def _init_worker(progress_array, worker_slot_counter, stagger_seconds, threads_per_worker, total_workers):
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

    # Pin this worker to its own share of the real, allowed core set.
    # Necessary because MediaPipe Tasks' Python API exposes no way to cap
    # its own internal (XNNPACK) thread pool -- confirmed by inspecting
    # BaseOptions/ImageSegmenterOptions, neither takes a num_threads
    # argument -- and measured live that each worker process was actually
    # running ~48 OS threads regardless of --threads-per-worker or the
    # OMP_NUM_THREADS/TF_NUM_INTRAOP_THREADS env vars (one threadpool per
    # MediaPipe model -- segmenter+pose+face -- each apparently sized off
    # the real core count), driving load average to ~6x the real core
    # count on a 16-core pod. Those env vars still get set (module top) in
    # case a future MediaPipe version starts honoring them, but they can't
    # be relied on. Restricting this process's own CPU affinity bounds the
    # threads IT spawns to only the cores assigned to it, regardless of how
    # many MediaPipe creates -- oversubscription then stays local to this
    # worker's own share instead of spilling across every other worker's
    # cores too, which is what was actually driving load average to ~96 on
    # a 16-core box (all 4 workers' ~48 threads each fighting over all 16
    # cores at once, not just their intended 4).
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
    # Match cv2's own thread pool to the same --threads-per-worker value the
    # OMP_NUM_THREADS/TF_NUM_INTRAOP_THREADS env vars at module top used for
    # MediaPipe's TFLite delegate, so both actually agree on how many
    # threads this worker process is meant to use -- was hardcoded to 1,
    # which was fine for the old many-single-threaded-workers architecture
    # but actively wastes cores now that fewer, properly multi-threaded
    # workers is the validated-faster approach (see module-top comment).
    cv2.setNumThreads(threads_per_worker)
    global _progress
    _progress = progress_array


def _process_segment(seg_idx, input_path, start_frame, end_frame, fps, src_w, src_h, out_w, out_h,
                      settings, encoder, encoder_args, tmp_dir):
    import cv2
    import numpy as np

    # cv2's thread count for this process was already set once in
    # _init_worker -- no need to redo it per segment (this function only
    # ever runs once per worker in the current one-segment-per-worker
    # design anyway).

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

            # Pass the RGB conversion already computed above (when
            # human-aware) so process_frame doesn't redo the identical
            # cv2.cvtColor call a second time -- same bytes either way.
            out_rgb, prev_gray = pipeline.process_frame(rgba, frame_settings, rgb=rgb_for_human)
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
    parser.add_argument("--temporal-denoise", action="store_true", dest="temporal_denoise",
                         help="Experimental: motion-adaptive smoothing of the pre-Canny gray image "
                              "to reduce sensor-noise-driven edge jitter (and the file-size bloat it "
                              "causes), without blurring real motion -- off by default, opt in to test.")
    parser.add_argument("--max-dimension", type=int, default=None,
                         help="Opt-in resolution cap. Omit for full source resolution (the default) -- "
                              "unlike the browser version, nothing is capped unless you ask for it.")
    parser.add_argument("--fps", type=float, default=None, help="Override output fps (default: source fps)")
    default_workers = max(1, _cores // _threads_per_worker)
    parser.add_argument("--workers", type=int, default=default_workers,
                         help=f"Parallel worker processes (default: real cores / threads-per-worker "
                              f"= {_cores} / {_threads_per_worker} = {default_workers} on this machine). "
                              f"Uses the container's actual allocation (os.sched_getaffinity), not "
                              f"os.cpu_count() -- the latter reports the HOST's total inside a container, "
                              f"which caused a real 128-workers-on-16-real-cores oversubscription bug.")
    parser.add_argument("--threads-per-worker", type=int, default=_threads_per_worker,
                         dest="threads_per_worker",
                         help=f"Threads each worker's cv2/MediaPipe use internally (default: "
                              f"round(sqrt(real cores)) = {_default_threads_per_worker} on this machine, "
                              f"balancing process-count overhead against per-process thread-scaling "
                              f"efficiency). Validated live: 4 workers x 4 threads on 16 real cores hit "
                              f"~51fps aggregate vs. this same file's old always-1-thread default getting "
                              f"~6fps on the same class of hardware -- MediaPipe's CPU delegate sizes its "
                              f"own thread pool independent of cv2.setNumThreads() and scales well with "
                              f"more threads per process, unlike naively adding more single-threaded "
                              f"processes, which just oversubscribes past the real core count instead of "
                              f"adding throughput. NOTE: changing this after cli.py has already started "
                              f"has no effect on OMP_NUM_THREADS/TF_NUM_INTRAOP_THREADS (module-top comment "
                              f"explains why those are read from sys.argv directly, before this parser "
                              f"exists) -- this argument exists so --help/introspection show the value "
                              f"actually in effect, and so cv2.setNumThreads() inside each worker agrees "
                              f"with it; to actually change the thread count, pass this flag on every run.")
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
    quality_choices = ", ".join(f"{name} (CRF {crf})" for name, crf in QUALITY_PRESETS.items())
    parser.add_argument("--quality", choices=list(QUALITY_PRESETS), default="balanced",
                         help=f"Named output-size/quality tier ({quality_choices}). Default 'balanced' (CRF "
                              f"24) -- meaningfully smaller than this file's old hardcoded CRF 20 with real "
                              f"margin before the point real ink content was tested to start losing crispness "
                              f"(see --crf's help for the actual measured numbers). Same tier names/CRF "
                              f"values as the Linearty Android app's encoder_quality_mode setting where they "
                              f"overlap (up to 'small'); 'aggressive' goes further since ink/line-art content "
                              f"specifically (flat color, no photographic gradient) tolerates more compression "
                              f"than the general video that app's tiers were designed for. Ignored if --crf "
                              f"is also given.")
    parser.add_argument("--crf", type=int, default=None,
                         help="Explicit CRF, overriding --quality, for any value not covered by a named "
                              "tier (or just a preference for a raw number). Lower = higher quality/larger "
                              "file. Measured live on a real 90s 1080p ink render: CRF 20 -> 33.0MB, "
                              "24 -> 24.5MB, 28 -> 15.8MB (52%% smaller than 20), 32 -> 9.1MB (72%% smaller), "
                              "40 -> 2.4MB (93%% smaller, user-confirmed by eye to still look right on real "
                              "output). Nothing stops you from going even further than 40, but that's past "
                              "what's actually been checked for line crispness -- worth a look before trusting it.")
    parser.add_argument("--gpu-filter", action="store_true", dest="gpu_filter",
                         help="Opportunistic OpenCV CUDA use for the ink filter itself. Only takes effect if the "
                              "installed opencv build has CUDA support (the standard pip wheel does not); "
                              "silently ignored otherwise.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.human_aware is None:
        args.human_aware = args.preset != "classic"  # mirrors script.js's own default
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
