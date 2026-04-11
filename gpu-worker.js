'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// gpu-worker.js — WebGPU-accelerated line-art pipeline with OpenCV CPU fallback
//
// On start-up the worker tries to initialise a WebGPU device and compile all
// WGSL compute pipelines.  When that succeeds every frame is processed on the
// GPU (typically 5–50 ms/frame vs. 200–2000 ms on the CPU).  When WebGPU is
// unavailable or the device request fails the worker transparently falls back
// to loading OpenCV.js and running the identical CPU pipeline from worker.js,
// so the caller never needs to know which path is active.
//
// Both paths post { type: 'cv-ready' } when initialised and respond to the
// standard { type: 'process' } / { type: 'reset' } message protocol used by
// LineArtProcessor in script.js.
//
// GPU pipeline stages
// ────────────────────
//  1. RGBA u8 → greyscale f32
//  2. Bilateral filter (greyscale; N passes, refineSigma on passes 2+)
//  3. Optional custom Gaussian blur passes (5×5, sigma ≈ 1.1)
//  4. Optional custom median blur passes (3×3)
//  5. Light Gaussian blur before Canny (3×3, sigma ≈ 1 px)
//  6. Sobel gradient — L1 magnitude + quantised direction
//  7. Non-maximum suppression
//  8. Double threshold (strong / weak edges)
//  9. Canny hysteresis (8 fixed iterations)
// 10. Finalise: {0,1,2} → binary mask {0,255}
// 11. Morphological close (3×3 dilate → erode)
// 12. Optional clean-speckles (morphological open, variable kernel)
// 13. Optional merge-double-edge (large close + thin erode)
// 14. Optional line-weight dilation
// 15. Colourize: edge mask → ink/background RGBA
//
// Note: CLAHE (darkBoost) is not implemented on the GPU path; that step is
// silently skipped.  All other settings are fully respected.
// ─────────────────────────────────────────────────────────────────────────────

let useGpu = false;
let gpuProcessor = null;
let cpuProcessor = null;

// ── WGSL shader sources ──────────────────────────────────────────────────────

const WGSL_RGBA_TO_GRAY = `
struct Dims { width: u32, height: u32 }
@group(0) @binding(0) var<storage, read>       rgba_in  : array<u32>;
@group(0) @binding(1) var<storage, read_write> gray_out : array<f32>;
@group(0) @binding(2) var<uniform>             dims     : Dims;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.width * dims.height) { return; }
    let px = rgba_in[idx];
    let r  = f32(px         & 0xffu) / 255.0;
    let g  = f32((px >>  8u) & 0xffu) / 255.0;
    let b  = f32((px >> 16u) & 0xffu) / 255.0;
    gray_out[idx] = 0.299 * r + 0.587 * g + 0.114 * b;
}
`;

// Bilateral filter operating on a single-channel f32 greyscale image.
// inv2ss = -0.5 / sigma_space²  (sigma in pixels)
// inv2sr = -0.5 / sigma_range²  (sigma_range = sigma_colour / 255, normalised to [0,1])
const WGSL_BILATERAL = `
struct BilParams { width: u32, height: u32, radius: u32, inv2ss: f32, inv2sr: f32 }
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             p   : BilParams;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= p.width || y >= p.height) { return; }
    let w = p.width; let cx = i32(x); let cy = i32(y); let r = i32(p.radius);
    let cv = src[y * w + x];
    var sum = 0.0; var wsum = 0.0;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let nx  = u32(clamp(cx + dx, 0, i32(p.width)  - 1));
            let ny  = u32(clamp(cy + dy, 0, i32(p.height) - 1));
            let val = src[ny * w + nx];
            let dv  = val - cv;
            let wt  = exp(f32(dx * dx + dy * dy) * p.inv2ss) * exp(dv * dv * p.inv2sr);
            sum  += val * wt;
            wsum += wt;
        }
    }
    dst[y * w + x] = sum / wsum;
}
`;

// Separable 1-D Gaussian blur.  horiz=1 → horizontal pass, horiz=0 → vertical.
// The kernel uses exp(-d²/2), giving sigma ≈ 1 px for any radius.
const WGSL_GAUSSIAN = `
struct GParams { width: u32, height: u32, radius: u32, horiz: u32 }
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             p   : GParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.width * p.height) { return; }
    let x = i32(idx % p.width); let y = i32(idx / p.width);
    let r = i32(p.radius);
    var sum = 0.0; var wsum = 0.0;
    for (var d = -r; d <= r; d++) {
        var nx: i32; var ny: i32;
        if (p.horiz == 1u) { nx = clamp(x + d, 0, i32(p.width)  - 1); ny = y; }
        else               { nx = x; ny = clamp(y + d, 0, i32(p.height) - 1); }
        let gw = exp(-f32(d * d) * 0.5);
        sum  += src[u32(ny) * p.width + u32(nx)] * gw;
        wsum += gw;
    }
    dst[idx] = sum / wsum;
}
`;

// 3×3 median filter using a fully-inlined 24-comparison sorting network.
// The previous version used a helper fn swp(ptr<function,f32>, ptr<function,f32>)
// and called it as swp(&v[0], &v[1]) etc.  That violates the WGSL specification
// rule that two pointer-typed arguments to the same call must not share the same
// root identifier — both &v[i] and &v[j] have root identifier `v`, causing a
// shader-creation error.  On strict runtimes (naga/Firefox) the pipeline becomes
// invalid and the pass is silently skipped; on lenient ones (tint/Chrome) the
// swap may degenerate to the identity operation, leaving the image unsmoothed.
// Either way the subsequent Canny step sees a noisy image and over-detects edges,
// producing a dark preview.  The fix reads each neighbour into its own scalar
// variable and implements every compare-and-swap inline with min/max — no
// function calls and no pointer aliasing.
const WGSL_MEDIAN3 = `
struct Dims { width: u32, height: u32 }
@group(0) @binding(0) var<storage, read>       src  : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst  : array<f32>;
@group(0) @binding(2) var<uniform>             dims : Dims;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let cx = i32(x); let cy = i32(y); let w = dims.width;
    let xL = u32(clamp(cx - 1, 0, i32(dims.width)  - 1));
    let xR = u32(clamp(cx + 1, 0, i32(dims.width)  - 1));
    let yT = u32(clamp(cy - 1, 0, i32(dims.height) - 1));
    let yB = u32(clamp(cy + 1, 0, i32(dims.height) - 1));
    var a0 = src[yT*w+xL]; var a1 = src[yT*w+x]; var a2 = src[yT*w+xR];
    var a3 = src[ y*w+xL]; var a4 = src[ y*w+x]; var a5 = src[ y*w+xR];
    var a6 = src[yB*w+xL]; var a7 = src[yB*w+x]; var a8 = src[yB*w+xR];
    // Inline compare-and-swap: t=ai; ai=min(t,aj); aj=max(t,aj)
    var t: f32;
    t=a0; a0=min(t,a1); a1=max(t,a1);  t=a3; a3=min(t,a4); a4=max(t,a4);  t=a6; a6=min(t,a7); a7=max(t,a7);
    t=a1; a1=min(t,a2); a2=max(t,a2);  t=a4; a4=min(t,a5); a5=max(t,a5);  t=a7; a7=min(t,a8); a8=max(t,a8);
    t=a0; a0=min(t,a1); a1=max(t,a1);  t=a3; a3=min(t,a4); a4=max(t,a4);  t=a6; a6=min(t,a7); a7=max(t,a7);
    t=a0; a0=min(t,a3); a3=max(t,a3);  t=a3; a3=min(t,a6); a6=max(t,a6);  t=a0; a0=min(t,a3); a3=max(t,a3);
    t=a1; a1=min(t,a4); a4=max(t,a4);  t=a4; a4=min(t,a7); a7=max(t,a7);  t=a1; a1=min(t,a4); a4=max(t,a4);
    t=a2; a2=min(t,a5); a5=max(t,a5);  t=a5; a5=min(t,a8); a8=max(t,a8);  t=a2; a2=min(t,a5); a5=max(t,a5);
    t=a1; a1=min(t,a3); a3=max(t,a3);  t=a2; a2=min(t,a6); a6=max(t,a6);  t=a2; a2=min(t,a3); a3=max(t,a3);
    t=a4; a4=min(t,a6); a6=max(t,a6);  t=a4; a4=min(t,a5); a5=max(t,a5);  t=a3; a3=min(t,a4); a4=max(t,a4);
    dst[y * w + x] = a4;
}
`;

// Sobel operator: L1 gradient magnitude and quantised direction (0=H, 1=NE, 2=V, 3=NW).
// Using L1 (|Gx|+|Gy|) matches OpenCV Canny's default L2gradient=false so the same
// threshold values can be scaled directly by 1/255.
const WGSL_SOBEL = `
struct Dims { width: u32, height: u32 }
@group(0) @binding(0) var<storage, read>       gray_in  : array<f32>;
@group(0) @binding(1) var<storage, read_write> magnitude: array<f32>;
@group(0) @binding(2) var<storage, read_write> direction: array<u32>;
@group(0) @binding(3) var<uniform>             dims     : Dims;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let cx = i32(x); let cy = i32(y);
    let w = dims.width; let h = dims.height;
    let x0 = u32(clamp(cx - 1, 0, i32(w) - 1));
    let x2 = u32(clamp(cx + 1, 0, i32(w) - 1));
    let y0 = u32(clamp(cy - 1, 0, i32(h) - 1));
    let y2 = u32(clamp(cy + 1, 0, i32(h) - 1));
    let tl = gray_in[y0*w+x0]; let tc = gray_in[y0*w+x]; let tr = gray_in[y0*w+x2];
    let ml = gray_in[ y*w+x0];                            let mr = gray_in[ y*w+x2];
    let bl = gray_in[y2*w+x0]; let bc = gray_in[y2*w+x]; let br = gray_in[y2*w+x2];
    let gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
    let gy = (bl + 2.0*bc + br) - (tl + 2.0*tc + tr);
    magnitude[y*w+x] = abs(gx) + abs(gy);
    var deg = atan2(gy, gx) * (180.0 / 3.14159265358979);
    if (deg < 0.0) { deg += 180.0; }
    var dir: u32;
    if      (deg < 22.5  || deg >= 157.5) { dir = 0u; }
    else if (deg < 67.5)                  { dir = 1u; }
    else if (deg < 112.5)                 { dir = 2u; }
    else                                  { dir = 3u; }
    direction[y*w+x] = dir;
}
`;

// Non-maximum suppression: suppress pixels that are not local maxima in their
// gradient direction.
const WGSL_NMS = `
struct Dims { width: u32, height: u32 }
@group(0) @binding(0) var<storage, read>       magnitude : array<f32>;
@group(0) @binding(1) var<storage, read>       direction : array<u32>;
@group(0) @binding(2) var<storage, read_write> suppressed: array<f32>;
@group(0) @binding(3) var<uniform>             dims      : Dims;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let cx = i32(x); let cy = i32(y);
    let w = dims.width; let h = dims.height;
    let idx = y*w+x; let mag = magnitude[idx]; let dir = direction[idx];
    var n1: f32; var n2: f32;
    if (dir == 0u) {
        n1 = magnitude[y*w + u32(clamp(cx - 1, 0, i32(w) - 1))];
        n2 = magnitude[y*w + u32(clamp(cx + 1, 0, i32(w) - 1))];
    } else if (dir == 1u) {
        n1 = magnitude[u32(clamp(cy-1,0,i32(h)-1))*w + u32(clamp(cx+1,0,i32(w)-1))];
        n2 = magnitude[u32(clamp(cy+1,0,i32(h)-1))*w + u32(clamp(cx-1,0,i32(w)-1))];
    } else if (dir == 2u) {
        n1 = magnitude[u32(clamp(cy - 1, 0, i32(h) - 1))*w + x];
        n2 = magnitude[u32(clamp(cy + 1, 0, i32(h) - 1))*w + x];
    } else {
        n1 = magnitude[u32(clamp(cy-1,0,i32(h)-1))*w + u32(clamp(cx-1,0,i32(w)-1))];
        n2 = magnitude[u32(clamp(cy+1,0,i32(h)-1))*w + u32(clamp(cx+1,0,i32(w)-1))];
    }
    suppressed[idx] = select(0.0, mag, mag >= n1 && mag >= n2);
}
`;

// Double threshold: classify each suppressed gradient as strong (2), weak (1), or none (0).
// Thresholds must be pre-scaled to the [0,1] float gradient range (divide OpenCV values by 255).
const WGSL_THRESHOLD = `
struct TParams { width: u32, height: u32, low: f32, high: f32 }
@group(0) @binding(0) var<storage, read>       suppressed: array<f32>;
@group(0) @binding(1) var<storage, read_write> edges     : array<u32>;
@group(0) @binding(2) var<uniform>             p         : TParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.width * p.height) { return; }
    let mag = suppressed[idx];
    if      (mag >= p.high) { edges[idx] = 2u; }
    else if (mag >= p.low)  { edges[idx] = 1u; }
    else                    { edges[idx] = 0u; }
}
`;

// One hysteresis pass: promote weak edges (1) that are 8-connected to a strong
// edge (2) to strong; suppress remaining weak edges to none (0).
// Run this shader 8 times to converge for typical edge connectivity lengths.
const WGSL_HYSTERESIS = `
struct Dims { width: u32, height: u32 }
@group(0) @binding(0) var<storage, read>       edges_in : array<u32>;
@group(0) @binding(1) var<storage, read_write> edges_out: array<u32>;
@group(0) @binding(2) var<uniform>             dims     : Dims;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= dims.width || y >= dims.height) { return; }
    let cx = i32(x); let cy = i32(y);
    let w = dims.width; let h = dims.height;
    let idx = y*w+x; let cur = edges_in[idx];
    if (cur != 1u) { edges_out[idx] = cur; return; }
    var found = false;
    for (var dy = -1; dy <= 1 && !found; dy++) {
        for (var dx = -1; dx <= 1 && !found; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            if (edges_in[u32(clamp(cy+dy, 0, i32(h)-1))*w
                        + u32(clamp(cx+dx, 0, i32(w)-1))] == 2u) {
                found = true;
            }
        }
    }
    edges_out[idx] = select(0u, 2u, found);
}
`;

// Convert the {0,1,2} edge state to a binary {0,255} mask ready for morphology.
const WGSL_FINALIZE = `
struct Dims { width: u32, height: u32 }
@group(0) @binding(0) var<storage, read>       edges_in : array<u32>;
@group(0) @binding(1) var<storage, read_write> mask_out : array<u32>;
@group(0) @binding(2) var<uniform>             dims     : Dims;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.width * dims.height) { return; }
    mask_out[idx] = select(0u, 255u, edges_in[idx] == 2u);
}
`;

// Variable-radius morphological dilation (max-pooling over a square neighbourhood).
const WGSL_MORPH_DILATE = `
struct MParams { width: u32, height: u32, radius: u32 }
@group(0) @binding(0) var<storage, read>       mask_in : array<u32>;
@group(0) @binding(1) var<storage, read_write> mask_out: array<u32>;
@group(0) @binding(2) var<uniform>             p       : MParams;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= p.width || y >= p.height) { return; }
    let cx = i32(x); let cy = i32(y); let r = i32(p.radius);
    var mv: u32 = 0u;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            mv = max(mv, mask_in[u32(clamp(cy+dy, 0, i32(p.height)-1))*p.width
                                + u32(clamp(cx+dx, 0, i32(p.width) -1))]);
        }
    }
    mask_out[y*p.width+x] = mv;
}
`;

// Variable-radius morphological erosion (min-pooling over a square neighbourhood).
const WGSL_MORPH_ERODE = `
struct MParams { width: u32, height: u32, radius: u32 }
@group(0) @binding(0) var<storage, read>       mask_in : array<u32>;
@group(0) @binding(1) var<storage, read_write> mask_out: array<u32>;
@group(0) @binding(2) var<uniform>             p       : MParams;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= p.width || y >= p.height) { return; }
    let cx = i32(x); let cy = i32(y); let r = i32(p.radius);
    var mv: u32 = 255u;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            mv = min(mv, mask_in[u32(clamp(cy+dy, 0, i32(p.height)-1))*p.width
                                + u32(clamp(cx+dx, 0, i32(p.width) -1))]);
        }
    }
    mask_out[y*p.width+x] = mv;
}
`;

// Map binary edge mask (255 = edge, 0 = background) to packed RGBA output.
// Edge pixels → ink colour; background pixels → bg colour.
const WGSL_COLORIZE = `
struct CParams { width: u32, height: u32, ink: u32, bg: u32 }
@group(0) @binding(0) var<storage, read>       mask    : array<u32>;
@group(0) @binding(1) var<storage, read_write> rgba_out: array<u32>;
@group(0) @binding(2) var<uniform>             p       : CParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.width * p.height) { return; }
    rgba_out[idx] = select(p.bg, p.ink, mask[idx] > 127u);
}
`;

// ── GpuProcessor ─────────────────────────────────────────────────────────────

class GpuProcessor {
    constructor(device) {
        this._device = device;
        this._width  = 0;
        this._height = 0;
        this._bufs   = null;
        this._pl     = {};
        // Cached uniform buffers — rebuilt only when settings or dimensions change.
        // Eliminates ~12 GPU buffer alloc/destroy cycles per frame during steady-
        // state rendering, removing the per-frame GPU memory allocation spikes.
        this._uniformCache    = null;
        this._uniformCacheKey = '';
        // Persistent MAP_READ staging buffer — recreated only on dimension change.
        // Avoids one GPU allocation + deallocation per frame.
        this._readBuf = null;
    }

    // Compile all WGSL compute pipelines synchronously.
    buildPipelines() {
        const d    = this._device;
        const mkPl = (code) => d.createComputePipeline({
            layout : 'auto',
            compute: { module: d.createShaderModule({ code }), entryPoint: 'main' },
        });
        this._pl = {
            rgbaToGray : mkPl(WGSL_RGBA_TO_GRAY),
            bilateral  : mkPl(WGSL_BILATERAL),
            gaussian   : mkPl(WGSL_GAUSSIAN),
            median3    : mkPl(WGSL_MEDIAN3),
            sobel      : mkPl(WGSL_SOBEL),
            nms        : mkPl(WGSL_NMS),
            threshold  : mkPl(WGSL_THRESHOLD),
            hysteresis : mkPl(WGSL_HYSTERESIS),
            finalize   : mkPl(WGSL_FINALIZE),
            morphDilate: mkPl(WGSL_MORPH_DILATE),
            morphErode : mkPl(WGSL_MORPH_ERODE),
            colorize   : mkPl(WGSL_COLORIZE),
        };
    }

    // (Re-)allocate GPU storage buffers when the frame dimensions change.
    _ensureBuffers(width, height) {
        if (this._width === width && this._height === height) return;
        if (this._bufs) Object.values(this._bufs).forEach(b => b.destroy());
        // Dimension change invalidates all caches derived from W/H.
        this._destroyUniformCache();
        if (this._readBuf) { this._readBuf.destroy(); this._readBuf = null; }
        this._width  = width;
        this._height = height;
        const n4      = width * height * 4; // bytes: one f32 or u32 per pixel
        const STORAGE = GPUBufferUsage.STORAGE;
        const COPY_DST= GPUBufferUsage.COPY_DST;
        const COPY_SRC= GPUBufferUsage.COPY_SRC;
        const mk = (usage) => this._device.createBuffer({ size: n4, usage });
        this._bufs = {
            rgbaIn    : mk(STORAGE | COPY_DST),
            gray      : mk(STORAGE),
            smooth1   : mk(STORAGE),
            smooth2   : mk(STORAGE),
            magnitude : mk(STORAGE),
            direction : mk(STORAGE),
            suppressed: mk(STORAGE),
            edgesA    : mk(STORAGE),
            edgesB    : mk(STORAGE),
            maskA     : mk(STORAGE),
            maskB     : mk(STORAGE),
            rgbaOut   : mk(STORAGE | COPY_SRC),
        };
    }

    // Create a small uniform buffer from a typed field descriptor array.
    _mkUniform(fields) {
        const UNIFORM_MIN_BINDING_SIZE = 16;
        const size = Math.max(
            UNIFORM_MIN_BINDING_SIZE,
            Math.ceil(fields.length * 4 / UNIFORM_MIN_BINDING_SIZE) * UNIFORM_MIN_BINDING_SIZE,
        );
        const buf = this._device.createBuffer({
            size,
            usage           : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        const view = new DataView(buf.getMappedRange());
        fields.forEach(({ type, value }, i) => {
            if (type === 'f32') view.setFloat32(i * 4, value, true);
            else                view.setUint32 (i * 4, value >>> 0, true);
        });
        buf.unmap();
        return buf;
    }

    // Compact key that captures every setting affecting uniform buffer values.
    // Same key across frames → cached uniforms are still valid.
    _computeUniformKey(W, H, settings) {
        const p = settings.preset;
        return [
            W, H, settings.detail,
            p.lowThreshold, p.highThreshold, p.bilateralDiameter, p.sigma,
            p.background[0], p.background[1], p.background[2],
            p.ink[0], p.ink[1], p.ink[2],
            settings.lineWeight,
            settings.cleanSpeckles       ? 1 : 0,
            settings.cleanSpecklesIntensity    || 0,
            settings.mergeDoubleEdge     ? 1 : 0,
            settings.mergeDoubleEdgeIntensity  || 0,
        ].join(':');
    }

    // Build every GPU uniform buffer needed for the given (W, H, settings).
    // Returned object is stored in this._uniformCache for reuse.
    _buildUniformCache(W, H, settings) {
        const mk = (fields) => this._mkUniform(fields);
        const detailFactor  = settings.detail / 62;
        const lowThreshold  = Math.max(12,
            Math.round(settings.preset.lowThreshold  / detailFactor));
        const highThreshold = Math.max(lowThreshold + 24,
            Math.round(settings.preset.highThreshold / detailFactor));
        const sigma         = Math.max(20,
            Math.round(settings.preset.sigma * (0.75 + (settings.detail - 35) / 100)));
        const bilRadius     = Math.floor(settings.preset.bilateralDiameter / 2);
        const refineSigma   = Math.max(15, Math.round(sigma * 0.5));
        const gpuLow        = lowThreshold  / 255.0;
        const gpuHigh       = highThreshold / 255.0;

        const mkBilU = (s) => mk([
            { type: 'u32', value: W },
            { type: 'u32', value: H },
            { type: 'u32', value: bilRadius },
            { type: 'f32', value: -0.5 / (s * s) },
            { type: 'f32', value: -0.5 / ((s / 255.0) * (s / 255.0)) },
        ]);

        const bgArr  = settings.preset.background;
        const inkArr = settings.preset.ink;
        const bgPx   = ((255 << 24) | (bgArr[2]  << 16) | (bgArr[1]  << 8) | bgArr[0])  >>> 0;
        const inkPx  = ((255 << 24) | (inkArr[2] << 16) | (inkArr[1] << 8) | inkArr[0]) >>> 0;

        const u = {
            dims    : mk([{type:'u32',value:W},{type:'u32',value:H}]),
            bilFull : mkBilU(sigma),
            bilRef  : mkBilU(refineSigma),
            g3h : mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:1},{type:'u32',value:1}]),
            g3v : mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:1},{type:'u32',value:0}]),
            g5h : mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:2},{type:'u32',value:1}]),
            g5v : mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:2},{type:'u32',value:0}]),
            thresh  : mk([{type:'u32',value:W},{type:'u32',value:H},{type:'f32',value:gpuLow},{type:'f32',value:gpuHigh}]),
            morph1  : mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:1}]),
            colorize: mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:inkPx},{type:'u32',value:bgPx}]),
        };

        // Conditional uniforms — only allocated when the corresponding option is on.
        if (settings.cleanSpeckles) {
            const openR = Math.max(1, Math.min(3, settings.cleanSpecklesIntensity || 1));
            u.open = mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:openR}]);
        }
        if (settings.mergeDoubleEdge) {
            const intensity = Math.max(1, Math.min(5, settings.mergeDoubleEdgeIntensity || 2));
            const mergeR    = 1 + intensity;
            u.merge = mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:mergeR}]);
            u.thin  = mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:1}]);
        }
        if (settings.lineWeight > 1) {
            const lwR = settings.lineWeight - 1;
            u.lw = mk([{type:'u32',value:W},{type:'u32',value:H},{type:'u32',value:lwR}]);
        }
        return u;
    }

    _destroyUniformCache() {
        if (this._uniformCache) {
            Object.values(this._uniformCache).forEach(b => b.destroy());
            this._uniformCache    = null;
            this._uniformCacheKey = '';
        }
    }

    async process(rgbaData, width, height, settings) {
        const dev = this._device;
        this._ensureBuffers(width, height);
        const b  = this._bufs;
        const pl = this._pl;
        const W  = width;
        const H  = height;

        // Upload RGBA input.
        dev.queue.writeBuffer(b.rgbaIn, 0, rgbaData.buffer,
            rgbaData.byteOffset, rgbaData.byteLength);

        // ── Uniform buffers: reuse cache if settings haven't changed ───────────
        // For video rendering every frame shares the same settings, so this
        // eliminates ~12 GPU buffer alloc/destroy operations per frame.
        const uKey = this._computeUniformKey(W, H, settings);
        if (uKey !== this._uniformCacheKey) {
            this._destroyUniformCache();
            this._uniformCache    = this._buildUniformCache(W, H, settings);
            this._uniformCacheKey = uKey;
        }
        const u = this._uniformCache;

        // ── Helpers ────────────────────────────────────────────────────────────
        const mkBg = (pipeline, ...buffers) => dev.createBindGroup({
            layout : pipeline.getBindGroupLayout(0),
            entries: buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
        });

        // Single command encoder covers both compute passes AND the copy-to-staging
        // pass.  Submitting everything in one call reduces driver round-trips and
        // allows the GPU to pipeline the copy immediately after compute.
        const enc = dev.createCommandEncoder();

        const disp1d = (pipeline, bg) => {
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline); pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(W * H / 256));
            pass.end();
        };
        const disp2d = (pipeline, bg) => {
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline); pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16));
            pass.end();
        };

        const getOther = (cur) => (cur === b.smooth1) ? b.smooth2 : b.smooth1;

        // ── Step 1: RGBA → greyscale ───────────────────────────────────────────
        disp1d(pl.rgbaToGray, mkBg(pl.rgbaToGray, b.rgbaIn, b.gray, u.dims));

        // ── Step 2: Pre-smoothing ──────────────────────────────────────────────
        let curSmooth = b.gray;
        const useBilateral = !settings.customMode || settings.useBilateral;
        if (useBilateral) {
            const bilPasses = settings.customMode
                ? Math.max(1, Math.min(5, settings.bilateralPasses || 2))
                : (settings.preset.smoothPasses || 2);
            for (let p = 0; p < bilPasses; p++) {
                const uBil = (p === 0) ? u.bilFull : u.bilRef;
                const next = getOther(curSmooth);
                disp2d(pl.bilateral, mkBg(pl.bilateral, curSmooth, next, uBil));
                curSmooth = next;
            }
        }

        if (settings.customMode && settings.useGaussian) {
            const gaussPasses = Math.max(1, Math.min(5, settings.gaussianPasses || 1));
            for (let p = 0; p < gaussPasses; p++) {
                const other = getOther(curSmooth);
                disp1d(pl.gaussian, mkBg(pl.gaussian, curSmooth, other,     u.g5h));
                disp1d(pl.gaussian, mkBg(pl.gaussian, other,     curSmooth, u.g5v));
            }
        }

        if (settings.customMode && settings.useMedian) {
            const medPasses = Math.max(1, Math.min(3, settings.medianPasses || 1));
            for (let p = 0; p < medPasses; p++) {
                const other = getOther(curSmooth);
                disp2d(pl.median3, mkBg(pl.median3, curSmooth, other, u.dims));
                curSmooth = other;
            }
        }

        // ── Step 3: Light 3×3 Gaussian before Canny ───────────────────────────
        {
            const other = getOther(curSmooth);
            disp1d(pl.gaussian, mkBg(pl.gaussian, curSmooth, other,     u.g3h));
            disp1d(pl.gaussian, mkBg(pl.gaussian, other,     curSmooth, u.g3v));
        }

        // ── Steps 4–8: Sobel → NMS → threshold → hysteresis → finalise ────────
        disp2d(pl.sobel,     mkBg(pl.sobel,     curSmooth,   b.magnitude, b.direction, u.dims));
        disp2d(pl.nms,       mkBg(pl.nms,       b.magnitude, b.direction, b.suppressed, u.dims));
        disp1d(pl.threshold, mkBg(pl.threshold, b.suppressed, b.edgesA,   u.thresh));

        let eIn = b.edgesA, eOut = b.edgesB;
        for (let i = 0; i < 8; i++) {
            disp2d(pl.hysteresis, mkBg(pl.hysteresis, eIn, eOut, u.dims));
            [eIn, eOut] = [eOut, eIn];
        }
        disp1d(pl.finalize, mkBg(pl.finalize, eIn, b.maskA, u.dims));

        // ── Step 9: Morphological close ────────────────────────────────────────
        disp2d(pl.morphDilate, mkBg(pl.morphDilate, b.maskA, b.maskB, u.morph1));
        disp2d(pl.morphErode,  mkBg(pl.morphErode,  b.maskB, b.maskA, u.morph1));
        let curMask = b.maskA;

        // ── Step 10: Optional clean speckles ──────────────────────────────────
        if (settings.cleanSpeckles) {
            disp2d(pl.morphErode,  mkBg(pl.morphErode,  curMask, b.maskB, u.open));
            disp2d(pl.morphDilate, mkBg(pl.morphDilate, b.maskB, b.maskA, u.open));
            curMask = b.maskA;
        }

        // ── Step 11: Optional merge double-edges ──────────────────────────────
        if (settings.mergeDoubleEdge) {
            const alt = (curMask === b.maskA) ? b.maskB : b.maskA;
            disp2d(pl.morphDilate, mkBg(pl.morphDilate, curMask, alt,     u.merge));
            disp2d(pl.morphErode,  mkBg(pl.morphErode,  alt,     curMask, u.merge));
            disp2d(pl.morphErode,  mkBg(pl.morphErode,  curMask, alt,     u.thin));
            curMask = alt;
        }

        // ── Step 12: Optional line-weight dilation ────────────────────────────
        if (settings.lineWeight > 1) {
            const alt = (curMask === b.maskA) ? b.maskB : b.maskA;
            disp2d(pl.morphDilate, mkBg(pl.morphDilate, curMask, alt, u.lw));
            curMask = alt;
        }

        // ── Step 13: Colourize ─────────────────────────────────────────────────
        disp1d(pl.colorize, mkBg(pl.colorize, curMask, b.rgbaOut, u.colorize));

        // ── Copy output to the persistent staging buffer ───────────────────────
        // Reusing this._readBuf across frames avoids one GPU alloc/dealloc per
        // frame — a significant GPU memory allocation hotspot at high frame rates.
        if (!this._readBuf) {
            this._readBuf = dev.createBuffer({
                size : W * H * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
        }
        // Fold the copy command into the same encoder that holds all compute work.
        // Submitting one command buffer instead of two reduces driver overhead and
        // allows the GPU driver to pipeline the copy right after the last compute.
        enc.copyBufferToBuffer(b.rgbaOut, 0, this._readBuf, 0, W * H * 4);
        dev.queue.submit([enc.finish()]);

        // mapAsync resolves once all submitted work (compute + copy) is done.
        await this._readBuf.mapAsync(GPUMapMode.READ);
        const result = new Uint8ClampedArray(this._readBuf.getMappedRange().slice(0));
        this._readBuf.unmap();
        // Do NOT destroy — buffer is reused next frame.

        return result;
    }

    destroy() {
        if (this._bufs) {
            Object.values(this._bufs).forEach(b => b.destroy());
            this._bufs = null;
        }
        this._destroyUniformCache();
        if (this._readBuf) {
            this._readBuf.destroy();
            this._readBuf = null;
        }
    }
}

// ── CpuProcessor (OpenCV fallback — same logic as WorkerProcessor in worker.js) ─
// Keep this class in sync with WorkerProcessor whenever worker.js is updated.

class CpuProcessor {
    constructor() {
        this.width    = 0;
        this.height   = 0;
        this.src      = null;
        this.rgb      = null;
        this.smoothed = null;
        this.gray     = null;
        this.edges    = null;
    }

    reset() {
        [this.src, this.rgb, this.smoothed, this.gray, this.edges].forEach((mat) => {
            if (mat) mat.delete();
        });
        this.src = this.rgb = this.smoothed = this.gray = this.edges = null;
        this.width = this.height = 0;
    }

    ensureSize(width, height) {
        if (this.width === width && this.height === height) return;
        this.reset();
        this.width    = width;
        this.height   = height;
        this.src      = new cv.Mat(height, width, cv.CV_8UC4);
        this.rgb      = new cv.Mat(height, width, cv.CV_8UC3);
        this.smoothed = new cv.Mat(height, width, cv.CV_8UC3);
        this.gray     = new cv.Mat(height, width, cv.CV_8UC1);
        this.edges    = new cv.Mat(height, width, cv.CV_8UC1);
    }

    process(rgbaData, width, height, settings) {
        this.ensureSize(width, height);
        this.src.data.set(rgbaData);
        cv.cvtColor(this.src, this.rgb, cv.COLOR_RGBA2RGB);

        const detailFactor  = settings.detail / 62;
        const lowThreshold  = Math.max(12, Math.round(settings.preset.lowThreshold  / detailFactor));
        const highThreshold = Math.max(lowThreshold + 24, Math.round(settings.preset.highThreshold / detailFactor));
        const sigma         = Math.max(20, Math.round(settings.preset.sigma * (0.75 + (settings.detail - 35) / 100)));
        const d             = settings.preset.bilateralDiameter;

        if (!settings.customMode) {
            cv.bilateralFilter(this.rgb, this.smoothed, d, sigma, sigma, cv.BORDER_DEFAULT);
            if (settings.preset.smoothPasses >= 2) {
                const refineSigma = Math.max(15, Math.round(sigma * 0.5));
                cv.bilateralFilter(this.smoothed, this.rgb,     d, refineSigma, refineSigma, cv.BORDER_DEFAULT);
                cv.bilateralFilter(this.rgb,     this.smoothed, d, refineSigma, refineSigma, cv.BORDER_DEFAULT);
            }
            cv.cvtColor(this.smoothed, this.gray, cv.COLOR_RGB2GRAY);
        } else {
            if (settings.useBilateral) {
                const bilateralPasses = Math.max(1, Math.min(5, settings.bilateralPasses || 2));
                cv.bilateralFilter(this.rgb, this.smoothed, d, sigma, sigma, cv.BORDER_DEFAULT);
                let bilateralResult = this.smoothed;
                let bilateralAlt    = this.rgb;
                if (bilateralPasses > 1) {
                    const refineSigma = Math.max(15, Math.round(sigma * 0.5));
                    for (let p = 1; p < bilateralPasses; p++) {
                        cv.bilateralFilter(bilateralResult, bilateralAlt, d, refineSigma, refineSigma, cv.BORDER_DEFAULT);
                        const tmp = bilateralResult; bilateralResult = bilateralAlt; bilateralAlt = tmp;
                    }
                }
                cv.cvtColor(bilateralResult, this.gray, cv.COLOR_RGB2GRAY);
            } else {
                cv.cvtColor(this.rgb, this.gray, cv.COLOR_RGB2GRAY);
            }
            if (settings.useGaussian) {
                const gaussianPasses = Math.max(1, Math.min(5, settings.gaussianPasses || 1));
                for (let p = 0; p < gaussianPasses; p++) {
                    cv.GaussianBlur(this.gray, this.gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
                }
            }
            if (settings.useMedian) {
                const medianPasses = Math.max(1, Math.min(3, settings.medianPasses || 1));
                for (let p = 0; p < medianPasses; p++) {
                    cv.medianBlur(this.gray, this.gray, 3);
                }
            }
        }

        cv.GaussianBlur(this.gray, this.gray, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT);

        if (settings.darkBoost) {
            const clahe = new cv.CLAHE(2.5, new cv.Size(8, 8));
            clahe.apply(this.gray, this.gray);
            clahe.delete();
        }

        cv.Canny(this.gray, this.edges, lowThreshold, highThreshold, 3, false);

        const closeKernel = cv.Mat.ones(3, 3, cv.CV_8U);
        cv.morphologyEx(this.edges, this.edges, cv.MORPH_CLOSE, closeKernel);
        closeKernel.delete();

        if (settings.cleanSpeckles) {
            const speckleIntensity = Math.max(1, Math.min(3, settings.cleanSpecklesIntensity || 1));
            const openSize         = 1 + speckleIntensity * 2;
            const openKernel       = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(openSize, openSize));
            cv.morphologyEx(this.edges, this.edges, cv.MORPH_OPEN, openKernel);
            openKernel.delete();
        }

        if (settings.mergeDoubleEdge) {
            const intensity   = Math.max(1, Math.min(5, settings.mergeDoubleEdgeIntensity || 2));
            const mergeSize   = 3 + intensity * 2;
            const mergeKernel = cv.Mat.ones(mergeSize, mergeSize, cv.CV_8U);
            cv.morphologyEx(this.edges, this.edges, cv.MORPH_CLOSE, mergeKernel);
            mergeKernel.delete();
            const thinKernel  = cv.Mat.ones(3, 3, cv.CV_8U);
            cv.erode(this.edges, this.edges, thinKernel);
            thinKernel.delete();
        }

        if (settings.lineWeight > 1) {
            const kernel = cv.Mat.ones(settings.lineWeight, settings.lineWeight, cv.CV_8U);
            cv.dilate(this.edges, this.edges, kernel);
            kernel.delete();
        }

        cv.bitwise_not(this.edges, this.edges);

        const bg    = settings.preset.background;
        const ink   = settings.preset.ink;
        const bgPx  = ((255 << 24) | (bg[2]  << 16) | (bg[1]  << 8) | bg[0])  >>> 0;
        const inkPx = ((255 << 24) | (ink[2] << 16) | (ink[1] << 8) | ink[0]) >>> 0;
        const mask  = this.edges.data;
        const out   = new Uint8ClampedArray(width * height * 4);
        const out32 = new Uint32Array(out.buffer);
        for (let i = 0, len = mask.length; i < len; i++) {
            out32[i] = mask[i] > 127 ? bgPx : inkPx;
        }
        return out;
    }
}

// ── Initialisation ────────────────────────────────────────────────────────────

async function initGpu() {
    if (!self.navigator?.gpu) throw new Error('WebGPU not available in this context');
    const adapter = await self.navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice();

    // Log but do not crash on device loss — the next process() call will surface
    // the error via the normal error-reporting path.
    device.lost.then((info) => {
        console.warn('[gpu-worker] GPU device lost:', info.reason, info.message);
        useGpu = false;
    });

    const proc = new GpuProcessor(device);
    proc.buildPipelines();
    return proc;
}

function initCpu() {
    console.log('[gpu-worker] Falling back to OpenCV CPU path...');
    try {
        importScripts('vendor/opencv.js');
        cv.onRuntimeInitialized = () => {
            console.log('[gpu-worker] OpenCV ready (CPU fallback)');
            cpuProcessor = new CpuProcessor();
            self.postMessage({ type: 'cv-ready' });
        };
    } catch (err) {
        self.postMessage({ type: 'cv-error', message: 'Failed to load OpenCV: ' + err.message });
    }
}

// Try GPU first; fall back to CPU on any failure.
initGpu()
    .then((proc) => {
        gpuProcessor = proc;
        useGpu       = true;
        console.log('[gpu-worker] WebGPU pipeline ready');
        self.postMessage({ type: 'cv-ready' });
    })
    .catch((err) => {
        console.log('[gpu-worker] WebGPU unavailable (' + err.message + '), loading OpenCV...');
        initCpu();
    });

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = function ({ data: msg }) {
    if (msg.type === 'process') {
        if (useGpu && gpuProcessor) {
            gpuProcessor.process(msg.rgbaData, msg.width, msg.height, msg.settings)
                .then((result) => {
                    self.postMessage({ type: 'result', id: msg.id, data: result }, [result.buffer]);
                })
                .catch((err) => {
                    const errMsg = err?.message || String(err);
                    console.error('[gpu-worker] GPU process error:', errMsg);
                    self.postMessage({ type: 'error', id: msg.id, message: errMsg });
                });
        } else if (cpuProcessor) {
            try {
                const result = cpuProcessor.process(msg.rgbaData, msg.width, msg.height, msg.settings);
                self.postMessage({ type: 'result', id: msg.id, data: result }, [result.buffer]);
            } catch (err) {
                let errMsg = err?.message || String(err);
                if (typeof err === 'number' && typeof cv !== 'undefined' && typeof cv.exceptionFromPtr === 'function') {
                    errMsg = cv.exceptionFromPtr(err).msg;
                }
                self.postMessage({ type: 'error', id: msg.id, message: errMsg });
            }
        } else {
            self.postMessage({ type: 'error', id: msg.id, message: 'Processor not ready.' });
        }
    } else if (msg.type === 'reset') {
        if (!useGpu && cpuProcessor) cpuProcessor.reset();
        // GPU path: per-frame buffers are recreated as needed; no explicit reset required.
    }
};
