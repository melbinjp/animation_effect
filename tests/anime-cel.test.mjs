import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const SceneSeg = require(join(root, 'scene-segmentation.js'));
const AnimeCel = require(join(root, 'anime-cel.js'));

function fill(w, h, fn) {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const [r, g, b] = fn(x, y);
            const p = (y * w + x) * 4;
            data[p] = r;
            data[p + 1] = g;
            data[p + 2] = b;
            data[p + 3] = 255;
        }
    }
    return data;
}

function luma(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

function sat(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    return max === 0 ? 0 : (max - min) / max;
}

function hue(r, g, b) {
    return AnimeCel.rgbToHsl(r, g, b)[0];
}

function hueDist(a, b) {
    const d = Math.abs(a - b) % 360;
    return Math.min(d, 360 - d);
}

test('classifyHsv names sky, vegetation, ground and structure', () => {
    assert.equal(SceneSeg.classifyHsv(220, 0.5, 0.8, 0.2), SceneSeg.SKY);
    assert.equal(SceneSeg.classifyHsv(120, 0.45, 0.5, 0.4), SceneSeg.VEGETATION);
    assert.equal(SceneSeg.classifyHsv(0, 0.05, 0.5, 0.8), SceneSeg.GROUND);
    assert.equal(SceneSeg.classifyHsv(20, 0.6, 0.7, 0.4), SceneSeg.STRUCTURE);
    assert.equal(SceneSeg.classifyHsv(220, 0.5, 0.8, 0.9), SceneSeg.STRUCTURE);
});

test('segment finds sky above ground and drops a speck', () => {
    const w = 80;
    const h = 80;
    const rgba = fill(w, h, (x, y) => {
        if (y < 24) return [40, 110, 230];
        if (y > 56) return [128, 128, 128];
        if (x >= 40 && x < 42 && y >= 30 && y < 32) return [20, 180, 40];
        return [150, 140, 130];
    });
    const seg = SceneSeg.segment(rgba, w, h, 4);
    let sky = 0;
    let ground = 0;
    let veg = 0;
    const top = 16 * w;
    const bottomStart = 64 * w;
    for (let i = 0; i < top; i++) if (seg.klass[i] === SceneSeg.SKY) sky++;
    for (let i = bottomStart; i < w * h; i++) if (seg.klass[i] === SceneSeg.GROUND) ground++;
    for (let i = 0; i < seg.klass.length; i++) if (seg.klass[i] === SceneSeg.VEGETATION) veg++;
    assert.ok(sky > top * 0.7, `sky pixels ${sky}`);
    assert.ok(ground > (w * h - bottomStart) * 0.7, `ground pixels ${ground}`);
    assert.ok(veg < 8, `speck survived as vegetation ${veg}`);
});

test('segment keeps two separate vegetation objects', () => {
    const w = 96;
    const h = 96;
    const rgba = fill(w, h, (x, y) => {
        if (x >= 8 && x < 36 && y >= 20 && y < 52) return [30, 150, 40];
        if (x >= 58 && x < 88 && y >= 20 && y < 52) return [40, 170, 50];
        return [170, 150, 130];
    });
    const seg = SceneSeg.segment(rgba, w, h, 4);
    const veg = seg.objects.filter((obj) => obj.klass === SceneSeg.VEGETATION);
    assert.ok(veg.length >= 2, `vegetation objects ${veg.length}`);
});

test('nextScene reuses a mask only for sequential frames', () => {
    const w = 32;
    const h = 32;
    const a = fill(w, h, () => [40, 110, 230]);
    const b = fill(w, h, () => [30, 160, 40]);
    const first = SceneSeg.nextScene(a, w, h, 4, null, true);
    assert.equal(first.reused, false);
    assert.equal(first.hold.left, 2);
    const second = SceneSeg.nextScene(b, w, h, 4, first.hold, true);
    assert.equal(second.reused, true);
    assert.equal(second.mask, first.mask);
    const third = SceneSeg.nextScene(b, w, h, 4, second.hold, true);
    assert.equal(third.reused, true);
    const fourth = SceneSeg.nextScene(b, w, h, 4, third.hold, true);
    assert.equal(fourth.reused, false);
    const batch = SceneSeg.nextScene(a, w, h, 4, first.hold, false);
    assert.equal(batch.reused, false);
    assert.equal(batch.hold, null);
    const wide = SceneSeg.nextScene(fill(40, 32, () => [40, 110, 230]), 40, 32, 4, first.hold, true);
    assert.equal(wide.reused, false);
});

test('paint keeps source hue, raises saturation, and lifts a dark flat', () => {
    const w = 32;
    const h = 32;
    const rgb = new Uint8Array(w * h * 3);
    for (let i = 0; i < w * h; i++) {
        rgb[i * 3] = 50;
        rgb[i * 3 + 1] = 70;
        rgb[i * 3 + 2] = 50;
    }
    const mask = new Uint8Array(w * h).fill(1);
    const klass = new Uint8Array(w * h).fill(SceneSeg.VEGETATION);
    const out = AnimeCel.paint({
        rgb, channels: 3, width: w, height: h,
        sceneMask: mask, sceneClass: klass, edges: new Uint8Array(w * h)
    });
    const p = (16 * w + 8) * 4;
    const r = out[p];
    const g = out[p + 1];
    const b = out[p + 2];
    assert.ok(g > r && g > b, `not green ${r},${g},${b}`);
    assert.ok(sat(r, g, b) > sat(50, 70, 50));
    assert.ok(luma(r, g, b) > luma(50, 70, 50) + 30);
    assert.ok(hueDist(hue(r, g, b), hue(50, 70, 50)) < 25);
});

test('paint uses three tones and keeps shadows brighter than the source darks', () => {
    const w = 48;
    const h = 16;
    const rgb = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const dark = x < w / 2;
            const p = (y * w + x) * 3;
            rgb[p] = dark ? 20 : 120;
            rgb[p + 1] = dark ? 40 : 200;
            rgb[p + 2] = dark ? 20 : 120;
        }
    }
    const out = AnimeCel.paint({
        rgb, channels: 3, width: w, height: h,
        sceneMask: new Uint8Array(w * h).fill(1),
        sceneClass: new Uint8Array(w * h).fill(SceneSeg.VEGETATION),
        edges: new Uint8Array(w * h)
    });
    let left = 0;
    let right = 0;
    const mid = (h / 2) | 0;
    for (let x = 2; x < w / 2 - 2; x++) {
        const p = (mid * w + x) * 4;
        left += luma(out[p], out[p + 1], out[p + 2]);
    }
    for (let x = w / 2 + 2; x < w - 2; x++) {
        const p = (mid * w + x) * 4;
        right += luma(out[p], out[p + 1], out[p + 2]);
    }
    left /= (w / 2 - 4);
    right /= (w / 2 - 4);
    assert.ok(left < right, `tones did not separate ${left} vs ${right}`);
    assert.ok(left > luma(20, 40, 20) + 15, `shadow was not lifted ${left}`);
});

test('paint inks a region boundary and fringes the next pixels', () => {
    const w = 48;
    const h = 24;
    const rgb = new Uint8Array(w * h * 3);
    const mask = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const p = (y * w + x) * 3;
            rgb[p] = 40;
            rgb[p + 1] = 140;
            rgb[p + 2] = 50;
            mask[y * w + x] = x < 24 ? 1 : 2;
        }
    }
    const out = AnimeCel.paint({
        rgb, channels: 3, width: w, height: h,
        sceneMask: mask,
        sceneClass: new Uint8Array(w * h).fill(SceneSeg.VEGETATION),
        edges: new Uint8Array(w * h),
        ink: [28, 18, 42]
    });
    const mid = 12;
    const flat = (mid * w + 6) * 4;
    const edge = (mid * w + 24) * 4;
    const fringe = (mid * w + 26) * 4;
    assert.ok(luma(out[edge], out[edge + 1], out[edge + 2]) < luma(out[flat], out[flat + 1], out[flat + 2]) - 40);
    assert.ok(out[fringe] > out[flat], `fringe red ${out[fringe]} vs flat ${out[flat]}`);
});

test('face colour stays below the neon cap', () => {
    const w = 24;
    const h = 24;
    const rgb = new Uint8Array(w * h * 3);
    const classes = new Uint8Array(w * h).fill(AnimeCel.HUMAN_FACE);
    for (let i = 0; i < w * h; i++) {
        rgb[i * 3] = 180;
        rgb[i * 3 + 1] = 140;
        rgb[i * 3 + 2] = 120;
    }
    const out = AnimeCel.paint({
        rgb, channels: 3, width: w, height: h,
        classMask: classes,
        sceneMask: new Uint8Array(w * h).fill(1),
        sceneClass: new Uint8Array(w * h).fill(SceneSeg.STRUCTURE),
        edges: new Uint8Array(w * h)
    });
    const p = (12 * w + 8) * 4;
    assert.ok(sat(out[p], out[p + 1], out[p + 2]) <= 0.62);
    assert.ok(hueDist(hue(out[p], out[p + 1], out[p + 2]), hue(180, 140, 120)) < 30);
});

test('scene islands use the same 40-pixel rule as the body map', () => {
    assert.equal(SceneSeg.ISLAND_MIN_AT_256, 40);
    const human = readFileSync(join(root, 'human.js'), 'utf8');
    const py = readFileSync(join(root, 'native', 'human.py'), 'utf8');
    assert.match(human, /filterSmallComponents\(nativeMask, sw, sh, 40\)/);
    assert.match(py, /stats\[:, cv2\.CC_STAT_AREA\] < 40/);
});

test('preset list gains anime and keeps the engines that were already there', () => {
    const script = readFileSync(join(root, 'script.js'), 'utf8');
    const html = readFileSync(join(root, 'index.html'), 'utf8');
    const worker = readFileSync(join(root, 'gpu-worker.js'), 'utf8');
    const block = script.match(/const STYLE_PRESETS = \{([\s\S]*?)\n\};/);
    assert.ok(block);
    const names = [...block[1].matchAll(/^\s+([a-zA-Z0-9_]+):\s*\{/gm)].map((m) => m[1]);
    for (const name of ['ultimate', 'manga', 'studio', 'neon', 'warm', 'vivid', 'blueprint', 'classic', 'human', 'subject', 'pencil', 'body', 'custom']) {
        assert.ok(names.includes(name), name);
    }
    assert.ok(names.includes('anime'));
    assert.match(script, /ultimate:\s*\{[\s\S]*?engine:\s*'ultimate'/);
    assert.match(script, /classic:\s*\{[\s\S]*?engine:\s*'classic'/);
    assert.match(script, /anime:\s*\{[\s\S]*?engine:\s*'anime'/);
    assert.match(html, /data-preset="anime"/);
    assert.match(html, /value="ultimate"/);
    assert.match(html, /value="classic"/);
    assert.match(worker, /settings\.engine === 'ultimate'/);
    assert.match(worker, /settings\.engine === 'anime'/);
    assert.match(worker, /paintUltimateInk\(/);
});
