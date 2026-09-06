"""Vectorized ink rendering and XDoG stylization pipeline.
Dispatches array operations to CuPy (GPU) or NumPy (CPU) via hw_detect.
"""

from hw_detect import get_xp, get_ndimage, to_device, to_host
import cv2

HUMAN_BG = 0
HUMAN_HAIR = 1
HUMAN_BODY = 2
HUMAN_FACE = 3
HUMAN_CLOTHES = 4
HUMAN_OTHER = 5

def _get_pal():
    xp = get_xp()
    return xp.array([
        [28, 28, 32],      # 0: HUMAN_BG (dark charcoal)
        [196, 146, 72],    # 1: HUMAN_HAIR (warm amber)
        [196, 112, 112],   # 2: HUMAN_BODY (dusty rose)
        [232, 186, 154],   # 3: HUMAN_FACE (light peach)
        [92, 124, 164],    # 4: HUMAN_CLOTHES (slate blue)
        [92, 164, 148],    # 5: HUMAN_OTHER (sage green)
    ], dtype=xp.uint8)

def colorize_class_mask(class_mask, alpha_blend=True):
    xp = get_xp()
    PAL = _get_pal()
    idx = xp.clip(class_mask, 0, len(PAL) - 1)
    rgb = PAL[idx]
    if not alpha_blend:
        return rgb
    alpha = xp.where(class_mask == HUMAN_BG, 70.0 / 255.0, 200.0 / 255.0)[..., None]
    blended = (rgb.astype(xp.float32) * alpha + 255.0 * (1.0 - alpha))
    return xp.clip(blended, 0, 255).astype(xp.uint8)

def blend_body_overlay(ink_rgb, class_mask):
    if class_mask is None:
        return ink_rgb
    xp = get_xp()
    ink_rgb = to_device(ink_rgb)
    class_mask = to_device(class_mask)
    
    ink_f = ink_rgb.astype(xp.float32)
    dimmed_ink = ink_f * 0.55 + 255.0 * 0.45
    
    overlay_rgb = colorize_class_mask(class_mask, alpha_blend=True).astype(xp.float32)
    mult = (dimmed_ink * overlay_rgb) / 255.0
    final = dimmed_ink * 0.10 + mult * 0.90
    return to_host(xp.clip(final, 0, 255).astype(xp.uint8))

def clamp01(arr):
    xp = get_xp()
    return xp.clip(arr, 0.0, 1.0)

def temporal_denoise_gray(gray, prev_gray, motion_threshold=13.0, base_alpha=0.6):
    if prev_gray is None:
        return gray
    xp = get_xp()
    gray = to_device(gray)
    prev_gray = to_device(prev_gray)
    
    diff = xp.abs(gray.astype(xp.int16) - prev_gray.astype(xp.int16)).astype(xp.float32)
    motion_weight = clamp01(diff / motion_threshold)
    effective_alpha = base_alpha + (1 - base_alpha) * motion_weight

    smoothed = effective_alpha * gray.astype(xp.float32) + (1 - effective_alpha) * prev_gray.astype(xp.float32)
    return to_host(xp.clip(smoothed, 0, 255).astype(xp.uint8))

def silhouette_mask(class_mask):
    xp = get_xp()
    ndimage = get_ndimage()
    binary = (class_mask != 0).astype(xp.uint8)
    dil = ndimage.maximum_filter(binary, size=5, mode='constant', cval=0.0)
    ero = ndimage.minimum_filter(binary, size=3, mode='constant', cval=0.0)
    return ((dil > 0) & (ero == 0)).astype(xp.float32)

def compute_class_confidence(class_mask, num_classes=6):
    xp = get_xp()
    ndimage = get_ndimage()
    h, w = class_mask.shape
    ones = xp.ones((h, w), dtype=xp.float32)
    kernel = xp.ones((3, 3), dtype=xp.float32)
    total = ndimage.correlate(ones, kernel, mode='constant', cval=0.0)

    same = xp.zeros((h, w), dtype=xp.float32)
    for c in range(num_classes):
        indicator = (class_mask == c).astype(xp.float32)
        same_c = ndimage.correlate(indicator, kernel, mode='constant', cval=0.0)
        same = xp.where(class_mask == c, same_c, same)

    return xp.where(total > 0, same / xp.maximum(total, 1e-6), 1.0).astype(xp.float32)

def apply_human_ink(ink, class_mask, extra_lines, settings):
    if class_mask is None or not settings.get("human_aware"):
        return ink
    xp = get_xp()
    
    isolation = settings.get("subject_isolation", 0.38)
    skin = settings.get("skin_smooth", 0.8)
    hair = settings.get("hair_boost", 1.32)
    sil_boost = settings.get("silhouette_boost", 0.72)

    sil = silhouette_mask(class_mask)
    conf = compute_class_confidence(class_mask)

    original = ink
    treated = original.copy()

    bg_mask = class_mask == HUMAN_BG
    treated = xp.where(bg_mask, original * (1 - isolation), treated)

    skin_mask = (class_mask == HUMAN_FACE) | (class_mask == HUMAN_BODY)
    keep = xp.where(original > 0.58, 1.0, 1 - skin)
    treated = xp.where(skin_mask, original * keep, treated)

    hair_mask = class_mask == HUMAN_HAIR
    treated = xp.where(hair_mask, xp.minimum(original * hair, 1.0), treated)

    other_mask = (class_mask == HUMAN_CLOTHES) | (class_mask == HUMAN_OTHER)
    treated = xp.where(other_mask, xp.minimum(original * 1.06, 1.0), treated)

    blended = original * (1 - conf) + treated * conf

    sil_val = sil * sil_boost
    blended = xp.maximum(blended, sil_val)

    if (settings.get("pose_lines") or settings.get("face_contours")) and extra_lines is not None:
        e = extra_lines.astype(xp.float32) / 255.0
        line_mask = extra_lines > 80
        blended = xp.where(line_mask & (e > blended), e, blended)

    return blended.astype(xp.float32)

def apply_human_ink_alpha(ink, alpha, extra_lines, settings):
    if alpha is None or not settings.get("human_aware"):
        return ink
    xp = get_xp()

    isolation = settings.get("subject_isolation", 0.38)
    skin = settings.get("skin_smooth", 0.8)
    sil_boost = settings.get("silhouette_boost", 0.72)

    binary_mask = (alpha > 0.5).astype(xp.uint8)
    sil = silhouette_mask(binary_mask)

    original = ink
    bg_treated = original * (1 - isolation)
    keep = xp.where(original > 0.58, 1.0, 1 - skin)
    fg_treated = original * keep

    a = alpha.astype(xp.float32)
    treated = bg_treated * (1 - a) + fg_treated * a
    blended = original * (1 - a) + treated * a

    sil_val = sil * sil_boost
    blended = xp.maximum(blended, sil_val)

    if (settings.get("pose_lines") or settings.get("face_contours")) and extra_lines is not None:
        e = extra_lines.astype(xp.float32) / 255.0
        line_mask = extra_lines > 80
        blended = xp.where(line_mask & (e > blended), e, blended)

    return blended.astype(xp.float32)

def apply_gray_world(rgb):
    xp = get_xp()
    rgb = to_device(rgb)
    channels = rgb.astype(xp.float64)
    r, g, b = channels[..., 0], channels[..., 1], channels[..., 2]
    m = xp.maximum(xp.maximum(r, g), b)
    valid = (m > 8) & (m < 250)
    if xp.count_nonzero(valid) < 64:
        return to_host(rgb)

    rm, gm, bm = r[valid].mean(), g[valid].mean(), b[valid].mean()
    gray = (rm + gm + bm) / 3
    scale = xp.array([gray / max(rm, 1), gray / max(gm, 1), gray / max(bm, 1)])

    out = xp.clip(channels * scale, 0, 255)
    return to_host(out.astype(xp.uint8))

def compute_xdog_map(gray, sigma, tau, phi):
    xp = get_xp()
    ndimage = get_ndimage()
    s = max(0.4, sigma or 0.82)
    gray_f = gray.astype(xp.float32)
    
    g1 = ndimage.gaussian_filter(gray_f, sigma=s, mode='reflect')
    g2 = ndimage.gaussian_filter(gray_f, sigma=s * 1.6, mode='reflect')

    t = 0.983 if tau is None else tau
    p = 210 if phi is None else phi
    eps = 0.01

    dog = g1 / 255.0 - t * (g2 / 255.0)
    return xp.where(dog >= eps, 1.0, 1.0 + xp.tanh(p * (dog - eps))).astype(xp.float32)

def paint_ultimate_ink(gray, edges, settings, class_mask=None, extra_lines=None, alpha=None):
    xp = get_xp()
    gray = to_device(gray)
    edges = to_device(edges)
    if class_mask is not None: class_mask = to_device(class_mask)
    if extra_lines is not None: extra_lines = to_device(extra_lines)
    if alpha is not None: alpha = to_device(alpha)

    preset = settings["preset"]
    xdog = compute_xdog_map(
        gray,
        preset.get("xdog_sigma", 0.82),
        preset.get("xdog_tau", 0.983),
        preset.get("xdog_phi", 210),
    )

    x_inv = 1 - xdog
    stroke = xp.where(x_inv > 0.07, clamp01((x_inv - 0.03) * 1.7), 0.0)
    structure = xp.where(edges >= 200, 0.48, 0.0)
    ink = xp.maximum(stroke, structure).astype(xp.float32)

    if alpha is not None:
        ink = apply_human_ink_alpha(ink, alpha, extra_lines, settings)
    else:
        ink = apply_human_ink(ink, class_mask, extra_lines, settings)

    bg = xp.array(preset.get("background", (250, 246, 238)), dtype=xp.float32)
    stroke_rgb = xp.array(preset.get("ink", (22, 28, 36)), dtype=xp.float32)

    t = clamp01(ink)
    a, b = 0.05, 0.55
    t = clamp01((t - a) / (b - a))
    t = t * t * (3 - 2 * t)

    t3 = t[..., None]
    blend = bg[None, None, :] * (1 - t3) + stroke_rgb[None, None, :] * t3
    below = (t < 0.018)[..., None]
    out = xp.where(below, bg[None, None, :], blend)
    
    return to_host(xp.clip(out, 0, 255).astype(xp.uint8))
