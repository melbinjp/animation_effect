'use strict';

// Shared Ultimate studio ink: XDoG strokes + light Canny structure, soft paint.
// Used by worker.js and gpu-worker.js. Relies on OpenCV (`cv`) being loaded.
// Tracker overlays (face mesh, pose bones, hand skeleton) are intentionally
// not drawn — output is real edges from the frame only.

function clamp01(v) {
    return v < 0 ? 0 : v > 1 ? 1 : v;
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

function paintUltimateInk(grayMat, edgeMat, width, height, settings) {
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
