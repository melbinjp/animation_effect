// Anime preset painter. Turns one captured frame into flat illustrated colour.
// Hue comes from that frame. It does not invent motion.
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    root.AnimeCel = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
    const HUMAN_BG = 0;
    const HUMAN_HAIR = 1;
    const HUMAN_BODY = 2;
    const HUMAN_FACE = 3;
    const HUMAN_CLOTHES = 4;
    const FRINGE = 2;

    function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    function luma(r, g, b) {
        return 0.299 * r + 0.587 * g + 0.114 * b;
    }

    function rgbToHsl(r, g, b) {
        const rr = r / 255;
        const gg = g / 255;
        const bb = b / 255;
        const max = Math.max(rr, gg, bb);
        const min = Math.min(rr, gg, bb);
        const l = (max + min) / 2;
        const d = max - min;
        let h = 0;
        let s = 0;
        if (d !== 0) {
            s = d / (1 - Math.abs(2 * l - 1));
            if (max === rr) h = ((gg - bb) / d) % 6;
            else if (max === gg) h = (bb - rr) / d + 2;
            else h = (rr - gg) / d + 4;
            h *= 60;
            if (h < 0) h += 360;
        }
        return [h, s, l];
    }

    function hslToRgb(h, s, l) {
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const hp = ((h % 360) + 360) % 360 / 60;
        const x = c * (1 - Math.abs(hp % 2 - 1));
        let r1 = 0;
        let g1 = 0;
        let b1 = 0;
        if (hp < 1) { r1 = c; g1 = x; }
        else if (hp < 2) { r1 = x; g1 = c; }
        else if (hp < 3) { g1 = c; b1 = x; }
        else if (hp < 4) { g1 = x; b1 = c; }
        else if (hp < 5) { r1 = x; b1 = c; }
        else { r1 = c; b1 = x; }
        const m = l - c / 2;
        return [
            Math.round(clamp(r1 + m, 0, 1) * 255),
            Math.round(clamp(g1 + m, 0, 1) * 255),
            Math.round(clamp(b1 + m, 0, 1) * 255)
        ];
    }

    function kindFor(key, sceneClass, sampleIndex) {
        if (key >= 256) {
            const human = key - 256;
            if (human === HUMAN_FACE) return 'face';
            if (human === HUMAN_BODY) return 'body';
            if (human === HUMAN_HAIR) return 'hair';
            if (human === HUMAN_CLOTHES) return 'clothes';
            return 'other';
        }
        const cls = sceneClass && sampleIndex != null ? sceneClass[sampleIndex] : 4;
        if (cls === 1) return 'sky';
        if (cls === 2) return 'vegetation';
        if (cls === 3) return 'ground';
        return 'structure';
    }

    function tonesFrom(r, g, b, kind) {
        const hsl = rgbToHsl(r, g, b);
        let sBoost = 1.55;
        let sAdd = 0.22;
        let sMax = 0.9;
        if (kind === 'face' || kind === 'body') {
            sBoost = 1.15;
            sAdd = 0.06;
            sMax = 0.55;
        } else if (kind === 'sky') {
            sBoost = 1.35;
            sAdd = 0.12;
            sMax = 0.75;
        }
        const S = clamp(hsl[1] * sBoost + sAdd, 0.28, sMax);
        const Lbase = clamp(hsl[2] * 0.45 + 0.48, 0.4, 0.8);
        const Lsh = clamp(Lbase * 0.62, 0.16, 0.55);
        const Lhi = clamp(Lbase + 0.16, 0.55, 0.92);
        return {
            shadow: hslToRgb(hsl[0], S, Lsh),
            base: hslToRgb(hsl[0], S, Lbase),
            highlight: hslToRgb(hsl[0] - 6, clamp(S + 0.04, 0, sMax), Lhi)
        };
    }

    function isHalftoneDot(x, y) {
        return (x % 5 === 1 && y % 5 === 1) || (x % 5 === 3 && y % 5 === 3);
    }

    function modeBin(hist, ri) {
        let best = 0;
        let bn = -1;
        for (let b = 0; b < 16; b++) {
            const v = hist[ri * 16 + b];
            if (v > bn) {
                bn = v;
                best = b;
            }
        }
        return best * 16 + 8;
    }

    function paint(opts) {
        const width = opts.width | 0;
        const height = opts.height | 0;
        const n = width * height;
        const out = new Uint8ClampedArray(n * 4);
        if (!n) return out;
        const rgb = opts.rgb;
        const channels = opts.channels || (rgb.length === n * 4 ? 4 : 3);
        let sceneMask = opts.sceneMask || null;
        let sceneClass = opts.sceneClass || null;
        if (sceneMask && sceneMask.length !== n) sceneMask = null;
        if (sceneClass && sceneClass.length !== n) sceneClass = null;
        if ((!sceneMask || !sceneClass) && typeof SceneSeg !== 'undefined') {
            const seg = SceneSeg.segment(rgb, width, height, channels);
            sceneMask = sceneMask || seg.mask;
            sceneClass = sceneClass || seg.klass;
        }
        const classMask = opts.classMask && opts.classMask.length === n ? opts.classMask : null;
        const edges = opts.edges && opts.edges.length === n ? opts.edges : null;
        const extra = opts.extraLines && opts.extraLines.length === n ? opts.extraLines : null;
        const ink = opts.ink || [28, 18, 42];

        const keyOf = new Uint16Array(n);
        const indexOf = new Int16Array(272);
        indexOf.fill(-1);
        let regions = 0;
        for (let i = 0; i < n; i++) {
            let key = 1;
            if (classMask && classMask[i] !== HUMAN_BG) key = 256 + classMask[i];
            else if (sceneMask && sceneMask[i]) key = sceneMask[i];
            keyOf[i] = key;
            if (indexOf[key] < 0) indexOf[key] = regions++;
        }

        const histR = new Uint32Array(regions * 16);
        const histG = new Uint32Array(regions * 16);
        const histB = new Uint32Array(regions * 16);
        const sampleAt = new Int32Array(regions);
        const minY = new Float32Array(regions);
        const maxY = new Float32Array(regions);
        minY.fill(255);
        sampleAt.fill(-1);
        for (let i = 0; i < n; i++) {
            const ri = indexOf[keyOf[i]];
            const p = i * channels;
            const r = rgb[p];
            const g = rgb[p + 1];
            const b = rgb[p + 2];
            histR[ri * 16 + (r >> 4)]++;
            histG[ri * 16 + (g >> 4)]++;
            histB[ri * 16 + (b >> 4)]++;
            const yv = luma(r, g, b);
            if (yv < minY[ri]) minY[ri] = yv;
            if (yv > maxY[ri]) maxY[ri] = yv;
            if (sampleAt[ri] < 0) sampleAt[ri] = i;
        }

        const cols = new Uint8Array(regions * 9);
        for (let ri = 0; ri < regions; ri++) {
            let key = 1;
            for (let k = 0; k < indexOf.length; k++) if (indexOf[k] === ri) { key = k; break; }
            const kind = kindFor(key, sceneClass, sampleAt[ri]);
            const tone = tonesFrom(modeBin(histR, ri), modeBin(histG, ri), modeBin(histB, ri), kind);
            cols.set(tone.shadow, ri * 9);
            cols.set(tone.base, ri * 9 + 3);
            cols.set(tone.highlight, ri * 9 + 6);
        }

        const inkAt = new Uint8Array(n);
        const shadowAt = new Uint8Array(n);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = y * width + x;
                const ri = indexOf[keyOf[i]];
                const p = i * channels;
                const yv = luma(rgb[p], rgb[p + 1], rgb[p + 2]);
                const spread = maxY[ri] - minY[ri];
                let tone = 1;
                if (spread >= 18) {
                    const t = (yv - minY[ri]) / spread;
                    if (t < 0.34) tone = 0;
                    else if (t > 0.67) tone = 2;
                }
                let cr = cols[ri * 9 + tone * 3];
                let cg = cols[ri * 9 + tone * 3 + 1];
                let cb = cols[ri * 9 + tone * 3 + 2];
                if (tone === 0) shadowAt[i] = 1;
                let border = false;
                const key = keyOf[i];
                if (x + 1 < width && keyOf[i + 1] !== key) border = true;
                else if (x > 0 && keyOf[i - 1] !== key) border = true;
                else if (y + 1 < height && keyOf[i + width] !== key) border = true;
                else if (y > 0 && keyOf[i - width] !== key) border = true;
                if (edges && edges[i] > 127) border = true;
                if (extra && extra[i] > 0) border = true;
                const o = i * 4;
                if (border) {
                    inkAt[i] = 1;
                    out[o] = ink[0];
                    out[o + 1] = ink[1];
                    out[o + 2] = ink[2];
                } else if (shadowAt[i] && isHalftoneDot(x, y)) {
                    out[o] = cr * 0.82;
                    out[o + 1] = cg * 0.82;
                    out[o + 2] = cb * 0.82;
                } else {
                    out[o] = cr;
                    out[o + 1] = cg;
                    out[o + 2] = cb;
                }
                out[o + 3] = 255;
            }
        }

        const magenta = [255, 32, 180];
        const cyan = [32, 220, 255];
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = y * width + x;
                if (inkAt[i]) continue;
                const o = i * 4;
                if (x >= FRINGE && inkAt[i - FRINGE]) {
                    out[o] = out[o] * 0.7 + magenta[0] * 0.3;
                    out[o + 1] = out[o + 1] * 0.7 + magenta[1] * 0.3;
                    out[o + 2] = out[o + 2] * 0.7 + magenta[2] * 0.3;
                } else if (x + FRINGE < width && inkAt[i + FRINGE]) {
                    out[o] = out[o] * 0.7 + cyan[0] * 0.3;
                    out[o + 1] = out[o + 1] * 0.7 + cyan[1] * 0.3;
                    out[o + 2] = out[o + 2] * 0.7 + cyan[2] * 0.3;
                }
            }
        }
        return out;
    }

    return {
        HUMAN_BG,
        HUMAN_HAIR,
        HUMAN_BODY,
        HUMAN_FACE,
        HUMAN_CLOTHES,
        FRINGE,
        rgbToHsl,
        hslToRgb,
        tonesFrom,
        isHalftoneDot,
        paint
    };
});
