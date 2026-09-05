"""Port of studio-ink.js. Same algorithm, vectorized with numpy/cv2 instead
of per-pixel loops — a literal 1:1 translation of the JS nested loops would
be correct but catastrophically slow in pure Python; cv2's native C++ ops
and numpy's broadcasting are what make this port actually faster than the
browser version, not just "the same speed in a different language."

Arrays here are always float32 in [0, 1] for "ink" (0 = pure background,
1 = full stroke) and uint8 for RGB images, matching the JS convention.
Color tuples (preset["background"], preset["ink"]) are (R, G, B), same
order as the JS arrays — this module never touches BGR.

One documented, deliberate departure from pixel-exact JS parity: silhouette_
mask's erosion uses a plain 3x3 cv2.erode instead of JS's custom version,
which also forces the outer 2px of the frame to erode away regardless of
the actual 3x3 neighborhood (a defensive border margin baked into the
original). The practical effect is confined to a 2px border strip and isn't
visible in a real frame — not worth reimplementing a hand-rolled sliding
window in place of a well-tested native primitive for a border-only
difference. See the plan's verification note: visual parity, not
byte-identical output, was always the goal.
"""

import numpy as np
import cv2

HUMAN_BG = 0
HUMAN_HAIR = 1
HUMAN_BODY = 2
HUMAN_FACE = 3
HUMAN_CLOTHES = 4
HUMAN_OTHER = 5


def clamp01(arr):
    return np.clip(arr, 0.0, 1.0)


def silhouette_mask(class_mask):
    """A band around the subject's outline — pixels near a foreground/
    background transition — used to force a minimum ink value right at the
    silhouette edge. See the module docstring for the one departure from
    the JS version's exact erosion behavior."""
    binary = (class_mask != 0).astype(np.uint8)
    dilate_kernel = np.ones((5, 5), np.uint8)  # JS: full 5x5 square, radius 2
    erode_kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(binary, dilate_kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ero = cv2.erode(binary, erode_kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return ((dil > 0) & (ero == 0)).astype(np.float32)


def compute_class_confidence(class_mask, num_classes=6):
    """For each pixel, the fraction of its 3x3 neighbors (including itself)
    that share its own class — 1.0 deep inside a solid region, tapering
    toward 0 right at a class boundary. A cheap proxy for classification
    confidence built from the hard mask alone; see apply_human_ink for why
    this (not MediaPipe's actual per-class confidence) is what gets used.

    Vectorized via cv2.boxFilter with zero-padding: boxing a per-class binary
    indicator gives the same-class neighbor count directly, and boxing an
    all-ones array with the same zero-padding gives the true neighbor count
    (naturally smaller near borders) — exactly the JS loop's `same`/`total`,
    without a Python-level loop over pixels.
    """
    h, w = class_mask.shape
    ones = np.ones((h, w), dtype=np.float32)
    total = cv2.boxFilter(ones, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)

    same = np.zeros((h, w), dtype=np.float32)
    for c in range(num_classes):
        indicator = (class_mask == c).astype(np.float32)
        same_c = cv2.boxFilter(indicator, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        same = np.where(class_mask == c, same_c, same)

    return np.where(total > 0, same / np.maximum(total, 1e-6), 1.0).astype(np.float32)


def apply_human_ink(ink, class_mask, extra_lines, settings):
    """Blends each pixel's per-class ink treatment toward "untreated" in
    proportion to compute_class_confidence — see that function's docstring
    for why. Mutates nothing; returns the blended array."""
    if class_mask is None or not settings.get("human_aware"):
        return ink

    isolation = settings.get("subject_isolation", 0.38)
    skin = settings.get("skin_smooth", 0.8)
    hair = settings.get("hair_boost", 1.32)
    sil_boost = settings.get("silhouette_boost", 0.72)

    sil = silhouette_mask(class_mask)
    conf = compute_class_confidence(class_mask)

    original = ink
    treated = original.copy()

    bg_mask = class_mask == HUMAN_BG
    treated = np.where(bg_mask, original * (1 - isolation), treated)

    skin_mask = (class_mask == HUMAN_FACE) | (class_mask == HUMAN_BODY)
    keep = np.where(original > 0.58, 1.0, 1 - skin)
    treated = np.where(skin_mask, original * keep, treated)

    hair_mask = class_mask == HUMAN_HAIR
    treated = np.where(hair_mask, np.minimum(original * hair, 1.0), treated)

    other_mask = (class_mask == HUMAN_CLOTHES) | (class_mask == HUMAN_OTHER)
    treated = np.where(other_mask, np.minimum(original * 1.06, 1.0), treated)

    blended = original * (1 - conf) + treated * conf

    sil_val = sil * sil_boost
    blended = np.maximum(blended, sil_val)

    if (settings.get("pose_lines") or settings.get("face_contours")) and extra_lines is not None:
        e = extra_lines.astype(np.float32) / 255.0
        line_mask = extra_lines > 80
        blended = np.where(line_mask & (e > blended), e, blended)

    return blended.astype(np.float32)


def apply_gray_world(rgb):
    """In-place-equivalent white balance (returns a new array). rgb is
    (H, W, 3) uint8, channel order (R, G, B)."""
    channels = rgb.astype(np.float64)
    r, g, b = channels[..., 0], channels[..., 1], channels[..., 2]
    m = np.maximum(np.maximum(r, g), b)
    valid = (m > 8) & (m < 250)
    if np.count_nonzero(valid) < 64:
        return rgb

    rm, gm, bm = r[valid].mean(), g[valid].mean(), b[valid].mean()
    gray = (rm + gm + bm) / 3
    scale = np.array([gray / max(rm, 1), gray / max(gm, 1), gray / max(bm, 1)])

    out = np.clip(channels * scale, 0, 255)
    return out.astype(np.uint8)


def compute_xdog_map(gray, sigma, tau, phi):
    """gray: (H, W) uint8 or float grayscale. Returns a float32 (H, W) map."""
    s = max(0.4, sigma or 0.82)
    gray_f = gray.astype(np.float32)
    g1 = cv2.GaussianBlur(gray_f, (0, 0), s, s)
    g2 = cv2.GaussianBlur(gray_f, (0, 0), s * 1.6, s * 1.6)

    t = 0.983 if tau is None else tau
    p = 210 if phi is None else phi
    eps = 0.01

    dog = g1 / 255.0 - t * (g2 / 255.0)
    return np.where(dog >= eps, 1.0, 1.0 + np.tanh(p * (dog - eps))).astype(np.float32)


def paint_ultimate_ink(gray, edges, settings, class_mask=None, extra_lines=None):
    """gray: (H, W) grayscale (post-smoothing, pre-normalize is fine — same
    input studio-ink.js's paintUltimateInk receives). edges: (H, W) uint8
    Canny output. Returns an (H, W, 3) uint8 RGB image."""
    preset = settings["preset"]
    xdog = compute_xdog_map(
        gray,
        preset.get("xdog_sigma", 0.82),
        preset.get("xdog_tau", 0.983),
        preset.get("xdog_phi", 210),
    )

    x_inv = 1 - xdog
    stroke = np.where(x_inv > 0.07, clamp01((x_inv - 0.03) * 1.7), 0.0)
    structure = np.where(edges >= 200, 0.48, 0.0)
    ink = np.maximum(stroke, structure).astype(np.float32)

    ink = apply_human_ink(ink, class_mask, extra_lines, settings)

    bg = np.array(preset.get("background", (250, 246, 238)), dtype=np.float32)
    stroke_rgb = np.array(preset.get("ink", (22, 28, 36)), dtype=np.float32)

    t = clamp01(ink)
    a, b = 0.05, 0.55
    t = clamp01((t - a) / (b - a))
    t = t * t * (3 - 2 * t)

    t3 = t[..., None]
    blend = bg[None, None, :] * (1 - t3) + stroke_rgb[None, None, :] * t3
    below = (t < 0.018)[..., None]
    out = np.where(below, bg[None, None, :], blend)
    return np.clip(out, 0, 255).astype(np.uint8)
