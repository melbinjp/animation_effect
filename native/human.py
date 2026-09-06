"""MediaPipe selfie multiclass segmentation and optional pose/face landmark overlays.
Maintains per-process engine instances for multiprocessing worker isolation.
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

# Per-process engine cache.
_engines = {"IMAGE": None, "VIDEO": None}
_want_landmarks = False
_video_timestamp_ms = 0


def _mp_modules():
    # Lazy import to avoid loading MediaPipe in processes where human-aware mode is disabled.
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
    """Initializes segmentation and landmark engines, attempting GPU delegate first with CPU fallback."""
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
    """Upsamples class confidence channels via bilinear interpolation and returns the argmax class mask."""
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
    """Loads landmark connection topology tables from mp_connections."""
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
    """Executes multiclass segmentation and optional landmark detection on an RGB frame.

    Returns:
        dict: Keys 'class_mask', 'extra_lines', 'person_ratio', and 'has_person', or None on failure.
    """
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
