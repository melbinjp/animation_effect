// Browser scene classes for the Anime preset. Colour and position only:
// no extra model, no frame-to-frame state inside this file. Callers that
// render frames out of order must pass sequential=false so a mask is never
// reused from a different frame.
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    root.SceneSeg = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
    const SKY = 1;
    const VEGETATION = 2;
    const GROUND = 3;
    const STRUCTURE = 4;
    // Same island rule as human.js filterSmallComponents(..., 40) at 256x256.
    const ISLAND_MIN_AT_256 = 40;

    function islandMin(w, h) {
        return Math.max(1, Math.round(ISLAND_MIN_AT_256 * (w * h) / (256 * 256)));
    }

    function classifyHsv(h, s, v, yNorm) {
        if (s < 0.18 && yNorm > 0.55) return GROUND;
        if (h >= 80 && h <= 170 && s >= 0.15 && v >= 0.12) return VEGETATION;
        if (h >= 190 && h <= 260 && s >= 0.12 && yNorm < 0.62 && v >= 0.35) return SKY;
        return STRUCTURE;
    }

    function rgbToHsv(r, g, b) {
        const rr = r / 255;
        const gg = g / 255;
        const bb = b / 255;
        const max = Math.max(rr, gg, bb);
        const min = Math.min(rr, gg, bb);
        const d = max - min;
        let h = 0;
        if (d !== 0) {
            if (max === rr) h = ((gg - bb) / d) % 6;
            else if (max === gg) h = (bb - rr) / d + 2;
            else h = (rr - gg) / d + 4;
            h *= 60;
            if (h < 0) h += 360;
        }
        return [h, max === 0 ? 0 : d / max, max];
    }

    function downsample(pixels, width, height, sw, sh, channels) {
        const out = new Uint8ClampedArray(sw * sh * 3);
        for (let y = 0; y < sh; y++) {
            const y0 = (y * height / sh) | 0;
            let y1 = ((y + 1) * height / sh) | 0;
            if (y1 <= y0) y1 = y0 + 1;
            for (let x = 0; x < sw; x++) {
                const x0 = (x * width / sw) | 0;
                let x1 = ((x + 1) * width / sw) | 0;
                if (x1 <= x0) x1 = x0 + 1;
                let r = 0;
                let g = 0;
                let b = 0;
                let n = 0;
                for (let yy = y0; yy < y1 && yy < height; yy++) {
                    for (let xx = x0; xx < x1 && xx < width; xx++) {
                        const p = (yy * width + xx) * channels;
                        r += pixels[p];
                        g += pixels[p + 1];
                        b += pixels[p + 2];
                        n++;
                    }
                }
                const o = (y * sw + x) * 3;
                const d = n || 1;
                out[o] = r / d;
                out[o + 1] = g / d;
                out[o + 2] = b / d;
            }
        }
        return out;
    }

    function classifyGrid(rgb, w, h) {
        const klass = new Uint8Array(w * h);
        for (let y = 0; y < h; y++) {
            const yNorm = (y + 0.5) / h;
            for (let x = 0; x < w; x++) {
                const p = (y * w + x) * 3;
                const hsv = rgbToHsv(rgb[p], rgb[p + 1], rgb[p + 2]);
                klass[y * w + x] = classifyHsv(hsv[0], hsv[1], hsv[2], yNorm);
            }
        }
        return klass;
    }

    function majority(labels, w, h) {
        const out = new Uint8Array(labels.length);
        const counts = new Uint16Array(5);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                counts.fill(0);
                for (let dy = -1; dy <= 1; dy++) {
                    const yy = y + dy;
                    if (yy < 0 || yy >= h) continue;
                    for (let dx = -1; dx <= 1; dx++) {
                        const xx = x + dx;
                        if (xx < 0 || xx >= w) continue;
                        counts[labels[yy * w + xx]]++;
                    }
                }
                let best = labels[y * w + x];
                let bestN = -1;
                for (let v = 1; v <= 4; v++) {
                    if (counts[v] > bestN) {
                        bestN = counts[v];
                        best = v;
                    }
                }
                out[y * w + x] = best;
            }
        }
        return out;
    }

    function labelInstances(klass, w, h) {
        const n = w * h;
        const minArea = islandMin(w, h);
        const tmp = new Int32Array(n);
        tmp.fill(-1);
        const areas = [];
        const classOf = [];
        const bounds = [];
        const stack = [];
        let next = 0;
        for (let i = 0; i < n; i++) {
            if (tmp[i] !== -1) continue;
            const cls = klass[i];
            stack.push(i);
            tmp[i] = next;
            let area = 0;
            let minx = i % w;
            let maxx = minx;
            let miny = (i / w) | 0;
            let maxy = miny;
            while (stack.length) {
                const p = stack.pop();
                area++;
                const x = p % w;
                const y = (p / w) | 0;
                if (x < minx) minx = x;
                if (x > maxx) maxx = x;
                if (y < miny) miny = y;
                if (y > maxy) maxy = y;
                if (x > 0 && tmp[p - 1] === -1 && klass[p - 1] === cls) {
                    tmp[p - 1] = next;
                    stack.push(p - 1);
                }
                if (x + 1 < w && tmp[p + 1] === -1 && klass[p + 1] === cls) {
                    tmp[p + 1] = next;
                    stack.push(p + 1);
                }
                if (y > 0 && tmp[p - w] === -1 && klass[p - w] === cls) {
                    tmp[p - w] = next;
                    stack.push(p - w);
                }
                if (y + 1 < h && tmp[p + w] === -1 && klass[p + w] === cls) {
                    tmp[p + w] = next;
                    stack.push(p + w);
                }
            }
            areas.push(area);
            classOf.push(cls);
            bounds.push([minx, miny, maxx, maxy]);
            next++;
        }

        const large = areas.map((area) => area >= minArea);
        if (next && !large.some(Boolean)) {
            let big = 0;
            for (let id = 1; id < next; id++) if (areas[id] > areas[big]) big = id;
            large[big] = true;
        }

        const compact = new Int32Array(next);
        compact.fill(-1);
        let instances = 0;
        for (let id = 0; id < next; id++) {
            if (!large[id]) continue;
            instances++;
            compact[id] = instances;
        }

        const mask = new Uint8Array(n);
        const klassOut = new Uint8Array(klass);
        const classOfFinal = new Uint8Array(instances + 1);
        for (let i = 0; i < n; i++) {
            const id = tmp[i];
            if (id >= 0 && large[id]) {
                mask[i] = compact[id];
                classOfFinal[compact[id]] = classOf[id];
            }
        }

        for (let iter = 0; iter < 8; iter++) {
            const prev = mask.slice();
            let grew = false;
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const i = y * w + x;
                    if (prev[i]) continue;
                    let pick = 0;
                    if (x > 0 && prev[i - 1]) pick = prev[i - 1];
                    else if (x + 1 < w && prev[i + 1]) pick = prev[i + 1];
                    else if (y > 0 && prev[i - w]) pick = prev[i - w];
                    else if (y + 1 < h && prev[i + w]) pick = prev[i + w];
                    if (pick) {
                        mask[i] = pick;
                        grew = true;
                    }
                }
            }
            if (!grew) break;
        }

        let leftover = 0;
        for (let i = 0; i < n; i++) {
            if (mask[i]) klassOut[i] = classOfFinal[mask[i]];
            else leftover++;
        }
        if (leftover) {
            instances++;
            const id = instances > 255 ? 1 : instances;
            for (let i = 0; i < n; i++) {
                if (!mask[i]) {
                    mask[i] = id;
                    klassOut[i] = STRUCTURE;
                }
            }
            if (id === instances) classOfFinal[id] = STRUCTURE;
        }

        const objects = [];
        for (let id = 1; id <= instances && id <= 255; id++) {
            let area = 0;
            let minx = w;
            let miny = h;
            let maxx = 0;
            let maxy = 0;
            const cls = classOfFinal[id] || STRUCTURE;
            for (let i = 0; i < n; i++) {
                if (mask[i] !== id) continue;
                area++;
                const x = i % w;
                const y = (i / w) | 0;
                if (x < minx) minx = x;
                if (y < miny) miny = y;
                if (x > maxx) maxx = x;
                if (y > maxy) maxy = y;
            }
            if (!area) continue;
            objects.push({
                id,
                klass: cls,
                x: minx,
                y: miny,
                w: maxx - minx + 1,
                h: maxy - miny + 1,
                area
            });
        }
        return { klass: klassOut, mask, objects, instances: objects.length };
    }

    // Hard class ids are not averaged. Scores are bilinear, then the winner
    // is chosen, same as human.js upsampleClassesBilinear.
    function bilinearClasses(klass, sw, sh, dw, dh) {
        if (sw === dw && sh === dh) return new Uint8Array(klass);
        const planes = [];
        for (let c = 1; c <= 4; c++) {
            const plane = new Float32Array(sw * sh);
            for (let i = 0; i < plane.length; i++) if (klass[i] === c) plane[i] = 1;
            planes.push(plane);
        }
        const out = new Uint8Array(dw * dh);
        const scaleX = (sw - 1) / Math.max(dw - 1, 1);
        const scaleY = (sh - 1) / Math.max(dh - 1, 1);
        for (let y = 0; y < dh; y++) {
            const sy = y * scaleY;
            const y0 = Math.floor(sy);
            const y1 = Math.min(sh - 1, y0 + 1);
            const fy = sy - y0;
            for (let x = 0; x < dw; x++) {
                const sx = x * scaleX;
                const x0 = Math.floor(sx);
                const x1 = Math.min(sw - 1, x0 + 1);
                const fx = sx - x0;
                let bestC = 1;
                let bestV = -1;
                for (let c = 0; c < 4; c++) {
                    const plane = planes[c];
                    const top = plane[y0 * sw + x0] + (plane[y0 * sw + x1] - plane[y0 * sw + x0]) * fx;
                    const bot = plane[y1 * sw + x0] + (plane[y1 * sw + x1] - plane[y1 * sw + x0]) * fx;
                    const v = top + (bot - top) * fy;
                    if (v > bestV) {
                        bestV = v;
                        bestC = c + 1;
                    }
                }
                out[y * dw + x] = bestC;
            }
        }
        return out;
    }

    function segment(pixels, width, height, channels) {
        const ch = channels || 4;
        if (!width || !height) {
            return { mask: new Uint8Array(0), klass: new Uint8Array(0), objects: [], instances: 0 };
        }
        const sw = Math.min(256, width);
        const sh = Math.min(256, height);
        const small = downsample(pixels, width, height, sw, sh, ch);
        let grid = classifyGrid(small, sw, sh);
        grid = majority(grid, sw, sh);
        grid = majority(grid, sw, sh);
        const labeled = labelInstances(grid, sw, sh);
        const klass = bilinearClasses(labeled.klass, sw, sh, width, height);
        const full = (width === sw && height === sh) ? labeled : labelInstances(klass, width, height);
        return {
            mask: full.mask,
            klass: full.klass,
            objects: full.objects,
            instances: full.instances
        };
    }

    // Live playback may reuse a mask for two following frames. Batch export
    // passes sequential=false and always gets a fresh mask.
    function nextScene(pixels, width, height, channels, hold, sequential) {
        if (
            sequential && hold && hold.w === width && hold.h === height &&
            hold.left > 0 && hold.mask && hold.klass
        ) {
            return {
                mask: hold.mask,
                klass: hold.klass,
                objects: hold.objects || [],
                instances: hold.instances || 0,
                reused: true,
                hold: {
                    mask: hold.mask,
                    klass: hold.klass,
                    objects: hold.objects || [],
                    instances: hold.instances || 0,
                    w: width,
                    h: height,
                    left: hold.left - 1
                }
            };
        }
        const seg = segment(pixels, width, height, channels);
        return {
            mask: seg.mask,
            klass: seg.klass,
            objects: seg.objects,
            instances: seg.instances,
            reused: false,
            hold: sequential ? {
                mask: seg.mask,
                klass: seg.klass,
                objects: seg.objects,
                instances: seg.instances,
                w: width,
                h: height,
                left: 2
            } : null
        };
    }

    return {
        SKY,
        VEGETATION,
        GROUND,
        STRUCTURE,
        ISLAND_MIN_AT_256,
        classifyHsv,
        rgbToHsv,
        segment,
        nextScene
    };
});
