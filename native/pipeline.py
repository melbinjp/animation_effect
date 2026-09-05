"""Port of worker.js's process() method — the per-frame CPU pipeline: white
balance, smoothing, 3-stage adaptive normalize, Canny, morphology, then
either the "ultimate" (XDoG, via ink.py) or "classic" paint stage.

Unlike worker.js, this file never manually allocates/frees kernels or Mats
— that caching existed specifically to reduce WASM-heap churn across a
browser-tab-shared FFmpeg/OpenCV.js memory budget (see worker.js's
_getRectKernel comment). Native OpenCV kernels are tiny numpy arrays,
garbage-collected normally; there's no equivalent constraint to work around
here, so this file just creates them inline where used.

settings is a plain dict, keys matching the JS settings object but
snake_case (detail, preset, engine, custom_mode, white_balance,
auto_normalize, dark_boost, dark_boost_clip, clean_speckles,
clean_speckles_intensity, merge_double_edge, merge_double_edge_intensity,
line_weight, color_edges, color_soft_ness, color_low_thresh,
color_high_thresh, color_line_weight, color_opacity, class_mask,
extra_lines, human_aware, skin_smooth, hair_boost, silhouette_boost,
subject_isolation, pose_lines, face_contours) plus settings["preset"], the
already-selected dict from presets.py (not just its name).
"""

import numpy as np
import cv2

from ink import apply_gray_world, paint_ultimate_ink


def _rect_kernel(size):
    return np.ones((size, size), np.uint8)


def _ellipse_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _remove_small_components(edges, min_area):
    """Vectorized port of the connected-components speckle-cleanup loop in
    worker.js: labels every contiguous white region, zeroes out any whose
    area is below min_area. Continuous lines, however thin, always
    accumulate enough pixels to survive; isolated specks don't."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8, ltype=cv2.CV_32S)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    keep[0] = True  # background label — irrelevant, never a foreground pixel below
    small_mask = (~keep)[labels] & (labels != 0)
    edges[small_mask] = 0
    return edges


def process_frame(rgba, settings):
    """rgba: (H, W, 4) uint8 array (RGBA, matching the browser's ImageData
    layout — the alpha channel is dropped immediately, same as worker.js's
    cv.COLOR_RGBA2RGB). Returns an (H, W, 3) uint8 RGB image."""
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

    if settings.get("white_balance"):
        rgb = apply_gray_world(rgb)

    gray_raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    preset = settings["preset"]
    detail = settings.get("detail", 62)
    detail_factor = detail / 62
    low_threshold = max(12, round(preset["low_threshold"] / detail_factor))
    high_threshold = max(low_threshold + 24, round(preset["high_threshold"] / detail_factor))
    sigma = max(20, round(preset["sigma"] * (0.75 + (detail - 35) / 100)))
    diameter = preset["bilateral_diameter"]

    custom_mode = settings.get("custom_mode", False)

    if not custom_mode:
        if settings.get("engine") == "ultimate":
            # XDoG carries the stroke; skip heavy bilateral so it stays fast.
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            smoothed = cv2.bilateralFilter(rgb, diameter, sigma, sigma, borderType=cv2.BORDER_DEFAULT)
            if preset.get("smooth_passes", 1) >= 2:
                refine_sigma = max(15, round(sigma * 0.5))
                smoothed = cv2.bilateralFilter(smoothed, diameter, refine_sigma, refine_sigma, borderType=cv2.BORDER_DEFAULT)
                smoothed = cv2.bilateralFilter(smoothed, diameter, refine_sigma, refine_sigma, borderType=cv2.BORDER_DEFAULT)
            gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
    else:
        if settings.get("use_bilateral"):
            bilateral_passes = max(1, min(5, settings.get("bilateral_passes", 2)))
            result = cv2.bilateralFilter(rgb, diameter, sigma, sigma, borderType=cv2.BORDER_DEFAULT)
            if bilateral_passes > 1:
                refine_sigma = max(15, round(sigma * 0.5))
                for _ in range(bilateral_passes - 1):
                    result = cv2.bilateralFilter(result, diameter, refine_sigma, refine_sigma, borderType=cv2.BORDER_DEFAULT)
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if settings.get("use_gaussian"):
            for _ in range(max(1, min(5, settings.get("gaussian_passes", 1)))):
                gray = cv2.GaussianBlur(gray, (5, 5), 0, 0, borderType=cv2.BORDER_DEFAULT)

        if settings.get("use_median"):
            for _ in range(max(1, min(3, settings.get("median_passes", 1)))):
                gray = cv2.medianBlur(gray, 3)

    # Three-stage adaptive normalize — see worker.js's comment for the full
    # rationale (gamma lift / histogram stretch / adaptive CLAHE cascade).
    last_mean = None
    if settings.get("auto_normalize"):
        mean_arr, std_arr = cv2.meanStdDev(gray_raw)
        mean = float(mean_arr[0][0])
        std = float(std_arr[0][0])
        last_mean = mean

        if mean < 80:
            gamma = max(1.5, min(3.0, 1.0 + (80 - mean) / 40))
            lut = np.array([round(((i / 255) ** (1 / gamma)) * 255) for i in range(256)], dtype=np.uint8)
            gray_raw = cv2.LUT(gray_raw, lut)
            gray = cv2.LUT(gray, lut)

        if std < 45:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        adaptive_clip = max(1.5, min(4.5, 150 / max(mean, 1)))
        gray = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(8, 8)).apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0, 0, borderType=cv2.BORDER_DEFAULT)

    if settings.get("dark_boost"):
        clip_limit = max(1.0, min(6.0, settings.get("dark_boost_clip", 2.5)))
        gray = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(gray)

    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _rect_kernel(3))

    if settings.get("clean_speckles"):
        intensity = max(1, min(3, settings.get("clean_speckles_intensity", 1)))
        min_area = {1: 4, 2: 12, 3: 30}[intensity]
        edges = _remove_small_components(edges, min_area)

    if settings.get("merge_double_edge"):
        intensity = max(1, min(5, settings.get("merge_double_edge_intensity", 2)))
        merge_size = 3 + intensity * 2
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _rect_kernel(merge_size))
        edges = cv2.erode(edges, _rect_kernel(3))

    if settings.get("line_weight", 1) > 1:
        edges = cv2.dilate(edges, _ellipse_kernel(settings["line_weight"] + 1))

    if settings.get("engine") == "ultimate":
        return paint_ultimate_ink(
            gray, edges, settings,
            class_mask=settings.get("class_mask"),
            extra_lines=settings.get("extra_lines"),
        )

    return _paint_classic(rgb, gray_raw, edges, settings, last_mean)


def _paint_classic(rgb, gray_raw, edges, settings, last_mean):
    """The non-ultimate ("classic") paint stage: binary ink/background, with
    an optional color-edges overlay in Custom mode. Vectorized with numpy
    boolean indexing instead of worker.js's per-pixel loop."""
    edges_inv = cv2.bitwise_not(edges)
    h, w = edges_inv.shape
    preset = settings["preset"]
    bg = np.array(preset.get("background", (255, 255, 255)), dtype=np.uint8)
    ink_color = np.array(preset.get("ink", (0, 0, 0)), dtype=np.uint8)

    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = bg
    ink_pixels = edges_inv < 127
    out[ink_pixels] = ink_color

    if settings.get("color_edges"):
        color_src = gray_raw
        if settings.get("color_softness", 0) > 0:
            ksize = settings["color_softness"] * 2 + 1
            color_src = cv2.GaussianBlur(gray_raw, (ksize, ksize), 0, 0, borderType=cv2.BORDER_DEFAULT)
            color_src = cv2.normalize(color_src, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            rgb = cv2.GaussianBlur(rgb, (ksize, ksize), 0, 0, borderType=cv2.BORDER_DEFAULT)

        color_edges = cv2.Canny(
            color_src, settings.get("color_low_thresh", 40), settings.get("color_high_thresh", 100),
            apertureSize=3, L2gradient=True,
        )
        if settings.get("color_line_weight", 1) > 1:
            color_edges = cv2.dilate(color_edges, _ellipse_kernel(settings["color_line_weight"] + 1))

        paint_lut = np.arange(256, dtype=np.uint8)
        if settings.get("auto_normalize") and last_mean is not None and last_mean < 80:
            gamma = max(1.5, min(3.0, 1.0 + (80 - last_mean) / 40))
            paint_lut = np.array([round(((i / 255) ** (1 / gamma)) * 255) for i in range(256)], dtype=np.uint8)

        painted = paint_lut[rgb]
        op = settings.get("color_opacity", 1.0)
        if op < 1.0:
            painted = (bg[None, None, :].astype(np.float32) * (1 - op) + painted.astype(np.float32) * op).astype(np.uint8)

        color_pixels = (~ink_pixels) & (color_edges > 127)
        out[color_pixels] = painted[color_pixels]

    return out
