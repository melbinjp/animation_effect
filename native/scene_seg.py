"""Scene classes for the Anime preset. Colour and position only, no extra model.

No mask is kept between calls. A sequential caller may hold one itself.
"""

import numpy as np

SKY = 1
VEGETATION = 2
GROUND = 3
STRUCTURE = 4
# Same island rule as native/human.py: clusters under 40 pixels at 256x256.
ISLAND_MIN_AT_256 = 40


def island_min(w, h):
    return max(1, int(round(ISLAND_MIN_AT_256 * (w * h) / (256 * 256))))


def classify_hsv(h, s, v, y_norm):
    if s < 0.18 and y_norm > 0.55:
        return GROUND
    if 80 <= h <= 170 and s >= 0.15 and v >= 0.12:
        return VEGETATION
    if 190 <= h <= 260 and s >= 0.12 and y_norm < 0.62 and v >= 0.35:
        return SKY
    return STRUCTURE


def _rgb_to_hsv(rgb):
    arr = rgb.astype(np.float32) / 255.0
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    d = maxc - minc
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    h = np.zeros_like(maxc)
    mask = d != 0
    rmax = mask & (maxc == r)
    gmax = mask & (maxc == g)
    bmax = mask & (maxc == b)
    h[rmax] = ((g[rmax] - b[rmax]) / d[rmax]) % 6
    h[gmax] = (b[gmax] - r[gmax]) / d[gmax] + 2
    h[bmax] = (r[bmax] - g[bmax]) / d[bmax] + 4
    h = (h * 60) % 360
    s = np.zeros_like(maxc)
    np.divide(d, maxc, out=s, where=maxc != 0)
    return h, s, maxc


def _downsample(rgb, sw, sh):
    h, w = rgb.shape[:2]
    out = np.zeros((sh, sw, 3), np.float32)
    for y in range(sh):
        y0 = int(y * h / sh)
        y1 = max(y0 + 1, int((y + 1) * h / sh))
        for x in range(sw):
            x0 = int(x * w / sw)
            x1 = max(x0 + 1, int((x + 1) * w / sw))
            block = rgb[y0:min(y1, h), x0:min(x1, w)]
            out[y, x] = block.reshape(-1, 3).mean(axis=0) if block.size else 0
    return out.astype(np.uint8)


def _majority(labels):
    h, w = labels.shape
    out = labels.copy()
    for y in range(h):
        for x in range(w):
            counts = np.zeros(5, np.int32)
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            patch = labels[y0:y1, x0:x1].ravel()
            for v in patch:
                counts[int(v)] += 1
            best = 1 + int(np.argmax(counts[1:]))
            out[y, x] = best
    return out


def _label_instances(klass):
    h, w = klass.shape
    n = h * w
    min_area = island_min(w, h)
    tmp = np.full(n, -1, np.int32)
    flat = klass.ravel()
    areas = []
    class_of = []
    stack = []
    next_id = 0
    for i in range(n):
        if tmp[i] != -1:
            continue
        cls = int(flat[i])
        stack.append(i)
        tmp[i] = next_id
        area = 0
        while stack:
            p = stack.pop()
            area += 1
            x, y = p % w, p // w
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                j = ny * w + nx
                if tmp[j] == -1 and int(flat[j]) == cls:
                    tmp[j] = next_id
                    stack.append(j)
        areas.append(area)
        class_of.append(cls)
        next_id += 1

    large = [a >= min_area for a in areas]
    if next_id and not any(large):
        large[int(np.argmax(areas))] = True
    compact = [-1] * next_id
    instances = 0
    for id_ in range(next_id):
        if not large[id_]:
            continue
        instances += 1
        compact[id_] = instances

    mask = np.zeros(n, np.uint8)
    class_of_final = np.zeros(instances + 1, np.uint8)
    for i in range(n):
        id_ = int(tmp[i])
        if id_ >= 0 and large[id_]:
            mask[i] = compact[id_]
            class_of_final[compact[id_]] = class_of[id_]

    for _ in range(8):
        prev = mask.copy()
        grew = False
        for y in range(h):
            for x in range(w):
                i = y * w + x
                if prev[i]:
                    continue
                pick = 0
                if x > 0 and prev[i - 1]:
                    pick = int(prev[i - 1])
                elif x + 1 < w and prev[i + 1]:
                    pick = int(prev[i + 1])
                elif y > 0 and prev[i - w]:
                    pick = int(prev[i - w])
                elif y + 1 < h and prev[i + w]:
                    pick = int(prev[i + w])
                if pick:
                    mask[i] = pick
                    grew = True
        if not grew:
            break

    klass_out = flat.copy()
    leftover = mask == 0
    if np.any(leftover):
        instances += 1
        dump = instances if instances <= 255 else 1
        mask[leftover] = dump
        klass_out[leftover] = STRUCTURE
        if dump == instances and dump < len(class_of_final):
            pass
        if dump == instances:
            class_of_final = np.pad(class_of_final, (0, 1))
            class_of_final[dump] = STRUCTURE
    for i in range(n):
        if mask[i] and int(mask[i]) < len(class_of_final) and class_of_final[mask[i]]:
            klass_out[i] = class_of_final[mask[i]]

    objects = []
    for id_ in range(1, instances + 1):
        sel = np.where(mask == id_)[0]
        if sel.size == 0:
            continue
        ys, xs = np.divmod(sel, w)
        objects.append({
            "id": int(id_ if id_ <= 255 else 1),
            "klass": int(class_of_final[id_] if id_ < len(class_of_final) else STRUCTURE),
            "x": int(xs.min()),
            "y": int(ys.min()),
            "w": int(xs.max() - xs.min() + 1),
            "h": int(ys.max() - ys.min() + 1),
            "area": int(sel.size),
        })
    return klass_out.reshape(h, w), mask.reshape(h, w), objects


def _bilinear_classes(klass, dw, dh):
    """Score planes are resized, then the winning class is chosen.

    Matches human.upsample_classes_bilinear: hard ids are not interpolated.
    """
    import cv2
    sh, sw = klass.shape
    if (sw, sh) == (dw, dh):
        return klass
    planes = [(klass == c).astype(np.float32) for c in range(1, 5)]
    resized = [cv2.resize(plane, (dw, dh), interpolation=cv2.INTER_LINEAR) for plane in planes]
    return (np.argmax(np.stack(resized, axis=-1), axis=-1) + 1).astype(np.uint8)


def segment_rgb(rgb):
    """Segment an HxWx3 uint8 image. Returns mask, klass, objects."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        empty = np.zeros((h, w), np.uint8)
        return {"mask": empty, "klass": empty, "objects": [], "instances": 0}
    sw, sh = min(256, w), min(256, h)
    small = _downsample(rgb, sw, sh)
    hsv_h, hsv_s, hsv_v = _rgb_to_hsv(small)
    klass = np.empty((sh, sw), np.uint8)
    for y in range(sh):
        y_norm = (y + 0.5) / sh
        for x in range(sw):
            klass[y, x] = classify_hsv(float(hsv_h[y, x]), float(hsv_s[y, x]), float(hsv_v[y, x]), y_norm)
    klass = _majority(_majority(klass))
    klass_s, mask_s, objects = _label_instances(klass)
    up_klass = _bilinear_classes(klass_s, w, h)
    if (sw, sh) == (w, h):
        full_klass, full_mask, full_objects = klass_s, mask_s, objects
    else:
        full_klass, full_mask, full_objects = _label_instances(up_klass)
    return {
        "mask": full_mask,
        "klass": full_klass,
        "objects": full_objects,
        "instances": len(full_objects),
    }


def next_scene(rgb, hold, sequential):
    h, w = np.asarray(rgb).shape[:2]
    if (
        sequential and hold and hold.get("w") == w and hold.get("h") == h
        and hold.get("left", 0) > 0 and hold.get("mask") is not None
    ):
        return {
            "mask": hold["mask"],
            "klass": hold["klass"],
            "objects": hold.get("objects") or [],
            "instances": hold.get("instances") or 0,
            "reused": True,
            "hold": {**hold, "left": hold["left"] - 1},
        }
    seg = segment_rgb(rgb)
    kept = None
    if sequential:
        kept = {
            "mask": seg["mask"],
            "klass": seg["klass"],
            "objects": seg["objects"],
            "instances": seg["instances"],
            "w": w,
            "h": h,
            "left": 2,
        }
    return {**seg, "reused": False, "hold": kept}
