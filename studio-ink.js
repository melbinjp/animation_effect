'use strict';

// Shared Ultimate studio ink: XDoG strokes + light Canny structure, soft paint.
// Used by worker.js and gpu-worker.js. Relies on OpenCV (`cv`) being loaded.
// Body-part maps (MediaPipe) hush background and quiet skin. Pose/face
// skeleton overlays stay off unless the user turns those toggles on.

const HUMAN_BG = 0;
const HUMAN_HAIR = 1;
const HUMAN_BODY = 2;
const HUMAN_FACE = 3;
const HUMAN_CLOTHES = 4;
const HUMAN_OTHER = 5;

function clamp01(v) {
    return v < 0 ? 0 : v > 1 ? 1 : v;
}

function silhouetteMask(classMask, w, h) {
    const n = w * h;
    const bin = new Uint8Array(n);
    for (let i = 0; i < n; i++) bin[i] = classMask[i] ? 1 : 0;
    const dil = new Uint8Array(n);
    const ero = new Uint8Array(n);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let d = 0;
            let e = 1;
            for (let dy = -2; dy <= 2; dy++) {
                const yy = y + dy;
                if (yy < 0 || yy >= h) { e = 0; continue; }
                for (let dx = -2; dx <= 2; dx++) {
                    const xx = x + dx;
                    if (xx < 0 || xx >= w) { e = 0; continue; }
                    const v = bin[yy * w + xx];
                    if (v) d = 1;
                    if (!v && Math.abs(dx) <= 1 && Math.abs(dy) <= 1) e = 0;
                }
            }
            dil[y * w + x] = d;
            ero[y * w + x] = bin[y * w + x] && e ? 1 : 0;
        }
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = dil[i] && !ero[i] ? 1 : 0;
    return out;
}

// For each pixel, how locally uniform its classification is — 1.0 deep
// inside a solid region of one class, tapering toward 0 right at a class
// boundary (all 8 neighbors agree vs. none of them do). This is a proxy for
// classification confidence built from the hard mask alone, not MediaPipe's
// actual per-class confidence: that data never leaves human.js today, and
// threading all 6 full-resolution float channels through the worker
// postMessage boundary for this would cost far more than the improvement is
// worth. Used below to stop committing fully to one class's treatment
// exactly where the classification itself is least trustworthy — a
// misclassified or boundary pixel gets a blended, partial treatment instead
// of a hard jump from its neighbor's completely different one.
function computeClassConfidence(classMask, w, h) {
    const n = w * h;
    const conf = new Float32Array(n);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = y * w + x;
            const cls = classMask[i];
            let same = 0;
            let total = 0;
            for (let dy = -1; dy <= 1; dy++) {
                const yy = y + dy;
                if (yy < 0 || yy >= h) continue;
                for (let dx = -1; dx <= 1; dx++) {
                    const xx = x + dx;
                    if (xx < 0 || xx >= w) continue;
                    total++;
                    if (classMask[yy * w + xx] === cls) same++;
                }
            }
            conf[i] = total > 0 ? same / total : 1;
        }
    }
    return conf;
}

function applyHumanInk(ink, classMask, extraLines, width, height, settings) {
    if (!classMask || !settings.humanAware) return;
    const n = width * height;
    if (classMask.length !== n) return;
    const isolation = settings.subjectIsolation == null ? 0.38 : settings.subjectIsolation;
    const skin = settings.skinSmooth == null ? 0.8 : settings.skinSmooth;
    const hair = settings.hairBoost == null ? 1.32 : settings.hairBoost;
    const silB = settings.silhouetteBoost == null ? 0.72 : settings.silhouetteBoost;
    const sil = silhouetteMask(classMask, width, height);
    const conf = computeClassConfidence(classMask, width, height);
    const poseOn = !!settings.poseLines;
    const faceOn = !!settings.faceContours;
    for (let i = 0; i < n; i++) {
        const cls = classMask[i];
        const original = ink[i];
        let treated = original;
        if (cls === HUMAN_BG) {
            treated = original * (1 - isolation);
        } else if (cls === HUMAN_FACE || cls === HUMAN_BODY) {
            const keep = original > 0.58 ? 1 : 1 - skin;
            treated = original * keep;
        } else if (cls === HUMAN_HAIR) {
            treated = original * hair > 1 ? 1 : original * hair;
        } else if (cls === HUMAN_CLOTHES || cls === HUMAN_OTHER) {
            treated = original * 1.06 > 1 ? 1 : original * 1.06;
        }
        // Blend toward the untreated value exactly where this pixel's
        // classification disagrees with its neighbors — a segmentation
        // boundary or an isolated misclassified pixel — instead of applying
        // the full treatment right up to a hard edge.
        const c = conf[i];
        ink[i] = original * (1 - c) + treated * c;

        const s = sil[i] * silB;
        if (s > ink[i]) ink[i] = s;
        if ((poseOn || faceOn) && extraLines && extraLines[i] > 80) {
            const e = extraLines[i] / 255;
            if (e > ink[i]) ink[i] = e;
        }
    }
}

function applyGrayWorld(rgbMat) {
    const data = rgbMat.data;
    let rs = 0, gs = 0, bs = 0, count = 0;
    for (let i = 0; i < data.length; i += 3) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const m = r > g ? (r > b ? r : b) : (g > b ? g : b);
        if (m > 8 && m < 250) {
            rs += r; gs += g; bs += b; count++;
        }
    }
    if (count < 64) return;
    const rm = rs / count, gm = gs / count, bm = bs / count;
    const gray = (rm + gm + bm) / 3;
    const sr = gray / Math.max(rm, 1);
    const sg = gray / Math.max(gm, 1);
    const sb = gray / Math.max(bm, 1);
    for (let i = 0; i < data.length; i += 3) {
        data[i]     = Math.max(0, Math.min(255, data[i] * sr));
        data[i + 1] = Math.max(0, Math.min(255, data[i + 1] * sg));
        data[i + 2] = Math.max(0, Math.min(255, data[i + 2] * sb));
    }
}

function computeXdogMap(grayMat, sigma, tau, phi) {
    const g1 = new cv.Mat();
    const g2 = new cv.Mat();
    const s = Math.max(0.4, sigma || 0.82);
    cv.GaussianBlur(grayMat, g1, new cv.Size(0, 0), s, s);
    cv.GaussianBlur(grayMat, g2, new cv.Size(0, 0), s * 1.6, s * 1.6);
    const n = grayMat.rows * grayMat.cols;
    const out = new Float32Array(n);
    const a = g1.data;
    const b = g2.data;
    const t = tau == null ? 0.983 : tau;
    const p = phi == null ? 210 : phi;
    const eps = 0.01;
    for (let i = 0; i < n; i++) {
        const dog = a[i] / 255 - t * (b[i] / 255);
        out[i] = dog >= eps ? 1 : 1 + Math.tanh(p * (dog - eps));
    }
    g1.delete();
    g2.delete();
    return out;
}

function paintUltimateInk(grayMat, edgeMat, width, height, settings, classMask, extraLines) {
    const preset = settings.preset || {};
    const xdog = computeXdogMap(
        grayMat,
        preset.xdogSigma || 0.82,
        preset.xdogTau || 0.983,
        preset.xdogPhi || 210,
    );
    const n = width * height;
    const ink = new Float32Array(n);
    const canny = edgeMat.data;
    for (let i = 0; i < n; i++) {
        const xInv = 1 - xdog[i];
        const stroke = xInv > 0.07 ? clamp01((xInv - 0.03) * 1.7) : 0;
        const structure = canny[i] >= 200 ? 0.48 : 0;
        ink[i] = stroke > structure ? stroke : structure;
    }
    applyHumanInk(ink, classMask, extraLines, width, height, settings);

    const bg = preset.background || [250, 246, 238];
    const strokeRgb = preset.ink || [22, 28, 36];
    const bgPx = ((255 << 24) | (bg[2] << 16) | (bg[1] << 8) | bg[0]) >>> 0;
    const out = new Uint8ClampedArray(n * 4);
    const out32 = new Uint32Array(out.buffer);
    const a = 0.05;
    const b = 0.55;
    const span = b - a;
    for (let i = 0; i < n; i++) {
        let t = clamp01(ink[i]);
        t = clamp01((t - a) / span);
        t = t * t * (3 - 2 * t);
        if (t < 0.018) {
            out32[i] = bgPx;
        } else {
            const R = (bg[0] * (1 - t) + strokeRgb[0] * t) | 0;
            const G = (bg[1] * (1 - t) + strokeRgb[1] * t) | 0;
            const B = (bg[2] * (1 - t) + strokeRgb[2] * t) | 0;
            out32[i] = ((255 << 24) | (B << 16) | (G << 8) | R) >>> 0;
        }
    }
    return out;
}
