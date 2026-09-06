"""Per-frame image processing pipeline: white balance, smoothing, adaptive normalization,
Canny edge detection, morphology, and ink rendering (XDoG or classic).
"""

import numpy as np
import cv2

from ink import apply_gray_world, paint_ultimate_ink, temporal_denoise_gray, blend_body_overlay


def _rect_kernel(size):
    return np.ones((size, size), np.uint8)


def _ellipse_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _remove_small_components(edges, min_area):
    """Removes connected edge components with area strictly less than min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8, ltype=cv2.CV_32S)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    keep[0] = True  # Retain background label
    small_mask = (~keep)[labels] & (labels != 0)
    edges[small_mask] = 0
    return edges


def process_frame(rgba, settings, rgb=None):
    """Process a single video frame or image with automatic VRAM OOM fallback.

    Args:
        rgba: Input image as (H, W, 4) uint8 RGBA array.
        settings: Configuration dictionary containing preset parameters, thresholds, and flags.
        rgb: Optional pre-computed (H, W, 3) uint8 RGB array to avoid redundant color conversion.

    Returns:
        tuple: (out_rgb, new_prev_gray) where out_rgb is (H, W, 3) uint8, and
               new_prev_gray is (H, W) uint8 for sequential temporal denoising.
    """
    from hw_detect import is_gpu_active, set_force_cpu, clear_gpu_memory, is_oom_error

    try:
        return _process_frame_impl(rgba, settings, rgb=rgb)
    except Exception as exc:
        if is_oom_error(exc) and is_gpu_active():
            print("[WARNING] GPU VRAM Out-of-Memory spike detected on frame. Flushing memory pool and retrying on CPU...", flush=True)
            clear_gpu_memory()
            set_force_cpu(True)
            try:
                return _process_frame_impl(rgba, settings, rgb=rgb)
            finally:
                set_force_cpu(False)
        raise


def _process_frame_impl(rgba, settings, rgb=None):
    if rgb is None:
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
            # XDoG engine: convert directly to grayscale without bilateral smoothing.
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

    new_prev_gray = gray
    if settings.get("temporal_denoise"):
        gray = temporal_denoise_gray(
            gray, settings.get("prev_gray"),
            motion_threshold=settings.get("temporal_motion_threshold", 13.0),
            base_alpha=settings.get("temporal_base_alpha", 0.6),
        )
        new_prev_gray = gray

    # Three-stage adaptive normalization: gamma correction, min-max stretch, and CLAHE.
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
        out = paint_ultimate_ink(
            gray, edges, settings,
            class_mask=settings.get("class_mask"),
            extra_lines=settings.get("extra_lines"),
            alpha=settings.get("alpha"),
        )
    else:
        out = _paint_classic(rgb, gray_raw, edges, settings, last_mean)

    if settings.get("body_map_overlay") or settings.get("preset_name") == "body":
        out = blend_body_overlay(out, settings.get("class_mask"))

    return out, new_prev_gray


def _paint_classic(rgb, gray_raw, edges, settings, last_mean):
    """Renders classic binary edge ink with optional colored edge overlay."""
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
