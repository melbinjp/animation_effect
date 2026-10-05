"""Anime preset painter. Flat colour from the frame's own hue. No new motion."""

import numpy as np

from scene_seg import segment_rgb

HUMAN_BG = 0
HUMAN_FACE = 3
HUMAN_BODY = 2
HUMAN_HAIR = 1
HUMAN_CLOTHES = 4
FRINGE = 2


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _rgb_to_hsl(r, g, b):
    rr, gg, bb = r / 255.0, g / 255.0, b / 255.0
    maxc = max(rr, gg, bb)
    minc = min(rr, gg, bb)
    l = (maxc + minc) / 2
    d = maxc - minc
    h = 0.0
    s = 0.0
    if d != 0:
        s = d / (1 - abs(2 * l - 1))
        if maxc == rr:
            h = ((gg - bb) / d) % 6
        elif maxc == gg:
            h = (bb - rr) / d + 2
        else:
            h = (rr - gg) / d + 4
        h *= 60
        if h < 0:
            h += 360
    return h, s, l


def _hsl_to_rgb(h, s, l):
    c = (1 - abs(2 * l - 1)) * s
    hp = ((h % 360) + 360) % 360 / 60
    x = c * (1 - abs(hp % 2 - 1))
    r1 = g1 = b1 = 0.0
    if hp < 1:
        r1, g1 = c, x
    elif hp < 2:
        r1, g1 = x, c
    elif hp < 3:
        g1, b1 = c, x
    elif hp < 4:
        g1, b1 = x, c
    elif hp < 5:
        r1, b1 = x, c
    else:
        r1, b1 = c, x
    m = l - c / 2
    return (
        int(round(_clamp(r1 + m, 0, 1) * 255)),
        int(round(_clamp(g1 + m, 0, 1) * 255)),
        int(round(_clamp(b1 + m, 0, 1) * 255)),
    )


def _kind(key, scene_class, sample):
    if key >= 256:
        human = key - 256
        if human == HUMAN_FACE:
            return "face"
        if human == HUMAN_BODY:
            return "body"
        if human == HUMAN_HAIR:
            return "hair"
        if human == HUMAN_CLOTHES:
            return "clothes"
        return "other"
    cls = 4
    if scene_class is not None and sample is not None:
        cls = int(scene_class[sample[0], sample[1]])
    return {1: "sky", 2: "vegetation", 3: "ground"}.get(cls, "structure")


def tones_from(r, g, b, kind):
    h, s, l = _rgb_to_hsl(r, g, b)
    s_boost, s_add, s_max = 1.55, 0.22, 0.9
    if kind in ("face", "body"):
        s_boost, s_add, s_max = 1.15, 0.06, 0.55
    elif kind == "sky":
        s_boost, s_add, s_max = 1.35, 0.12, 0.75
    S = _clamp(s * s_boost + s_add, 0.28, s_max)
    lbase = _clamp(l * 0.45 + 0.48, 0.4, 0.8)
    lsh = _clamp(lbase * 0.62, 0.16, 0.55)
    lhi = _clamp(lbase + 0.16, 0.55, 0.92)
    return {
        "shadow": _hsl_to_rgb(h, S, lsh),
        "base": _hsl_to_rgb(h, S, lbase),
        "highlight": _hsl_to_rgb(h - 6, _clamp(S + 0.04, 0, s_max), lhi),
    }


def _mode_channel(values):
    bins = np.clip(values.astype(np.int32) >> 4, 0, 15)
    hist = np.bincount(bins, minlength=16)
    return int(hist.argmax()) * 16 + 8


def paint_anime(rgb, edges=None, class_mask=None, scene_mask=None, scene_class=None,
                extra_lines=None, ink=(28, 18, 42)):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    h, w = rgb.shape[:2]
    if scene_mask is None or scene_class is None:
        seg = segment_rgb(rgb)
        scene_mask = scene_mask if scene_mask is not None else seg["mask"]
        scene_class = scene_class if scene_class is not None else seg["klass"]
    keys = np.ones((h, w), np.int32)
    if scene_mask is not None and np.asarray(scene_mask).shape == (h, w):
        sm = np.asarray(scene_mask).astype(np.int32)
        keys = np.where(sm > 0, sm, 1)
    if class_mask is not None:
        cm = np.asarray(class_mask)
        if cm.ndim == 1 and cm.size == h * w:
            cm = cm.reshape(h, w)
        if cm.shape == (h, w):
            person = cm != HUMAN_BG
            keys = np.where(person, 256 + cm.astype(np.int32), keys)

    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    out = np.zeros((h, w, 3), np.uint8)
    shadow = np.zeros((h, w), bool)
    unique = np.unique(keys)
    for key in unique:
        sel = keys == key
        if not np.any(sel):
            continue
        pixels = rgb[sel]
        mode = (
            _mode_channel(pixels[:, 0]),
            _mode_channel(pixels[:, 1]),
            _mode_channel(pixels[:, 2]),
        )
        sample = tuple(int(v) for v in np.argwhere(sel)[0])
        kind = _kind(int(key), scene_class, sample)
        tone = tones_from(*mode, kind)
        y = luma[sel]
        lo = float(y.min())
        hi = float(y.max())
        choice = np.ones(y.shape[0], np.uint8)
        if hi - lo >= 18:
            t = (y - lo) / (hi - lo)
            choice[t < 0.34] = 0
            choice[t > 0.67] = 2
        painted = np.array([tone["shadow"], tone["base"], tone["highlight"]], np.uint8)[choice]
        out[sel] = painted
        shadow[sel] = choice == 0

    yy, xx = np.indices((h, w))
    dot = ((xx % 5 == 1) & (yy % 5 == 1)) | ((xx % 5 == 3) & (yy % 5 == 3))
    shade = shadow & dot
    darkened = out.astype(np.float32)
    darkened[shade] *= 0.82
    out = np.clip(darkened, 0, 255).astype(np.uint8)

    border = np.zeros((h, w), bool)
    if w > 1:
        differ = keys[:, 1:] != keys[:, :-1]
        border[:, 1:] |= differ
        border[:, :-1] |= differ
    if h > 1:
        differ = keys[1:, :] != keys[:-1, :]
        border[1:, :] |= differ
        border[:-1, :] |= differ
    if edges is not None and np.asarray(edges).shape == (h, w):
        border |= np.asarray(edges) > 127
    if extra_lines is not None:
        extra = np.asarray(extra_lines)
        if extra.ndim == 1 and extra.size == h * w:
            extra = extra.reshape(h, w)
        if extra.shape == (h, w):
            border |= extra > 0
    ink_rgb = np.array(ink, np.uint8)
    out[border] = ink_rgb

    fr = out.astype(np.float32)
    left = np.zeros((h, w), bool)
    right = np.zeros((h, w), bool)
    if w > FRINGE:
        left[:, FRINGE:] = border[:, :-FRINGE]
        right[:, :-FRINGE] = border[:, FRINGE:]
    magenta = np.array([255, 32, 180], np.float32)
    cyan = np.array([32, 220, 255], np.float32)
    m = left & ~border
    c = right & ~border & ~m
    fr[m] = fr[m] * 0.7 + magenta * 0.3
    fr[c] = fr[c] * 0.7 + cyan * 0.3
    return np.clip(fr, 0, 255).astype(np.uint8)
