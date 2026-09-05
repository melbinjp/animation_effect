"""Port of human.js: MediaPipe segmentation + optional pose/face landmark
overlays, via the official Python `mediapipe` package instead of the JS
Tasks Vision bundle. Loads the exact same bundled model files already in
this repo (mediapipe/models/*), no re-fetching.

Process model: this module's engine cache is per-process (plain module-level
globals), not per-thread or global-to-the-run. cli.py's multiprocessing.Pool
workers each import this module fresh in their own process and lazily build
their own engines on first use — MediaPipe task objects aren't safely
shareable across a process boundary. VIDEO running mode's sequential-
timestamp requirement is satisfied as long as each worker processes its
assigned frames in increasing time order, which is how cli.py assigns work
(contiguous frame ranges per worker, not interleaved).

The pose/face landmark overlay path (off by default, opt-in via
--pose-lines/--face-contours) is a cosmetic extra, not the point of this
port. Verified live against an installed mediapipe==1.0.1: unlike the JS
tasks-vision bundle, the Python Tasks API exposes no PoseLandmarker.
POSE_CONNECTIONS-style class constants, and the legacy mediapipe.solutions
API that used to have them is gone from the Tasks-only 1.x package line
(confirmed by import failure, not assumed) - so connection topology comes
from mp_connections.py instead, a static table fetched from mediapipe's own
solutions source and range-checked against the known landmark counts. The
try/except here stays anyway as a defensive backstop: if a future mediapipe
release changes the landmark scheme, this degrades to "no overlay, a logged
warning" rather than failing the whole frame - segmentation (the actual
body-aware quieting) never depends on it.
"""

import os
import warnings

import numpy as np
import cv2

HUMAN_BG = 0
HUMAN_HAIR = 1
HUMAN_BODY = 2
HUMAN_FACE = 3
HUMAN_CLOTHES = 4
HUMAN_OTHER = 5

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mediapipe", "models")
SEG_MODEL = os.path.join(_MODELS_DIR, "selfie_multiclass_256x256.tflite")
POSE_MODEL = os.path.join(_MODELS_DIR, "pose_landmarker_lite.task")
FACE_MODEL = os.path.join(_MODELS_DIR, "face_landmarker.task")

# Per-process state — see module docstring.
_engines = {"IMAGE": None, "VIDEO": None}
_want_landmarks = False
_video_timestamp_ms = 0


def _mp_modules():
    # Imported lazily so a process that never needs human-aware processing
    # (e.g. --no-human-aware) doesn't pay MediaPipe's import cost at all.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    return mp, mp_python, mp_vision


def _build_engines(delegate, mode, want_landmarks):
    mp, mp_python, mp_vision = _mp_modules()
    running_mode = mp_vision.RunningMode.VIDEO if mode == "VIDEO" else mp_vision.RunningMode.IMAGE
    delegate_enum = (
        mp_python.BaseOptions.Delegate.GPU if delegate == "GPU" else mp_python.BaseOptions.Delegate.CPU
    )

    seg_options = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=SEG_MODEL, delegate=delegate_enum),
        running_mode=running_mode,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    seg = mp_vision.ImageSegmenter.create_from_options(seg_options)

    pose = None
    face = None
    if want_landmarks:
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL, delegate=delegate_enum),
            running_mode=running_mode,
            num_poses=2,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.4,
        )
        pose = mp_vision.PoseLandmarker.create_from_options(pose_options)

        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL, delegate=delegate_enum),
            running_mode=running_mode,
            num_faces=2,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        face = mp_vision.FaceLandmarker.create_from_options(face_options)

    return {"seg": seg, "pose": pose, "face": face, "has_landmarks": want_landmarks}


def ensure_human(landmarks=False, mode="IMAGE"):
    """Returns True if segmentation is available for this mode. Mirrors
    human.js's ensureHuman: GPU delegate tried first, falls back to CPU on
    any failure (GPU delegate success is more environment-dependent in
    Python than in-browser — this fallback is not optional)."""
    global _want_landmarks
    if landmarks:
        _want_landmarks = True

    cached = _engines.get(mode)
    if cached and (not _want_landmarks or cached["has_landmarks"]):
        return True

    try:
        _engines[mode] = _build_engines("GPU", mode, _want_landmarks)
        return True
    except Exception:
        try:
            _engines[mode] = _build_engines("CPU", mode, _want_landmarks)
            return True
        except Exception as err:
            warnings.warn(f"Linearty: body maps unavailable ({err})")
            _engines[mode] = None
            return False


def upsample_classes_bilinear(channels, out_w, out_h):
    """channels: list of 2D float32 arrays (one per class, model-native
    resolution). Bilinear-resizes each class's own confidence channel with
    cv2's native (fast, well-tested) resize, then argmax across classes —
    the numpy/cv2-native equivalent of human.js's upsampleClassesBilinear;
    see that function's comment for why per-channel bilinear-then-argmax is
    the mathematically correct way to upsample a categorical mask."""
    resized = [cv2.resize(ch, (out_w, out_h), interpolation=cv2.INTER_LINEAR) for ch in channels]
    stacked = np.stack(resized, axis=-1)
    return np.argmax(stacked, axis=-1).astype(np.uint8)


def _draw_connections(buf, w, h, landmarks, connections, radius, val, min_vis=0.35):
    if not connections:
        return
    thickness = max(1, int(round(radius * 2)))
    for start, end in connections:
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        a, b = landmarks[start], landmarks[end]
        if getattr(a, "visibility", 1.0) < min_vis or getattr(b, "visibility", 1.0) < min_vis:
            continue
        pt1 = (int(a.x * w), int(a.y * h))
        pt2 = (int(b.x * w), int(b.y * h))
        cv2.line(buf, pt1, pt2, int(val), thickness=thickness, lineType=cv2.LINE_AA)


def _pose_face_connections():
    """Returns (pose_connections, face_contours, face_lips, face_left,
    face_right) from the verified static tables in mp_connections.py — see
    module docstring. Any may be None if that import ever fails."""
    try:
        import mp_connections as topo
        return (
            topo.POSE_CONNECTIONS, topo.FACEMESH_CONTOURS, topo.FACEMESH_LIPS,
            topo.FACEMESH_LEFT_EYE, topo.FACEMESH_RIGHT_EYE,
        )
    except Exception as err:
        warnings.warn(f"Linearty: pose/face overlay connections unavailable, overlay disabled ({err})")
        return None, None, None, None, None


_connections_cache = None


def infer_human(image_rgb, width, height, settings, use_video_mode=False):
    """image_rgb: (H, W, 3) uint8 RGB numpy array. Returns a dict matching
    human.js's inferHuman return shape, or None."""
    global _video_timestamp_ms, _connections_cache

    mode = "VIDEO" if use_video_mode else "IMAGE"
    engines = _engines.get(mode)
    if not engines or not settings.get("human_aware"):
        return None

    try:
        mp, _, _ = _mp_modules()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        seg = engines["seg"]
        if use_video_mode:
            _video_timestamp_ms += 1
            result = seg.segment_for_video(mp_image, _video_timestamp_ms)
        else:
            result = seg.segment(mp_image)

        masks = result.confidence_masks
        if not masks:
            return None
        channels = [m.numpy_view() for m in masks]
        class_mask = upsample_classes_bilinear(channels, width, height)

        person_ratio = float(np.count_nonzero(class_mask != HUMAN_BG)) / max(1, class_mask.size)
        extra_lines = np.zeros((height, width), dtype=np.uint8)

        if (settings.get("pose_lines") or settings.get("face_contours")) and engines["has_landmarks"]:
            if _connections_cache is None:
                _connections_cache = _pose_face_connections()
            pose_conn, face_contours, face_lips, face_left, face_right = _connections_cache

            if settings.get("pose_lines") and engines["pose"] and pose_conn:
                poses = (
                    engines["pose"].detect_for_video(mp_image, _video_timestamp_ms)
                    if use_video_mode
                    else engines["pose"].detect(mp_image)
                )
                for lm in poses.pose_landmarks or []:
                    _draw_connections(extra_lines, width, height, lm, pose_conn, 1.35, 220, 0.4)

            if settings.get("face_contours") and engines["face"] and face_contours:
                faces = (
                    engines["face"].detect_for_video(mp_image, _video_timestamp_ms)
                    if use_video_mode
                    else engines["face"].detect(mp_image)
                )
                for lm in faces.face_landmarks or []:
                    _draw_connections(extra_lines, width, height, lm, face_contours, 0.9, 255, 0)
                    _draw_connections(extra_lines, width, height, lm, face_lips, 0.7, 200, 0)
                    _draw_connections(extra_lines, width, height, lm, face_left, 0.7, 240, 0)
                    _draw_connections(extra_lines, width, height, lm, face_right, 0.7, 240, 0)

        return {
            "class_mask": class_mask,
            "extra_lines": extra_lines,
            "person_ratio": person_ratio,
            "has_person": person_ratio > 0.012,
        }
    except Exception as err:
        warnings.warn(f"Linearty: human inference failed ({err})")
        return None
