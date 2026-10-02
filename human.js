import {
    FilesetResolver,
    ImageSegmenter,
    PoseLandmarker,
    FaceLandmarker,
} from './vendor/mediapipe/vision_bundle.mjs';

const HUMAN_BG = 0;
const HUMAN_HAIR = 1;
const HUMAN_BODY = 2;
const HUMAN_FACE = 3;
const HUMAN_CLOTHES = 4;
const HUMAN_OTHER = 5;

const WASM_PATH = new URL('mediapipe/wasm', import.meta.url).href.replace(/\/$/, '');
const SEG_MODEL = new URL('mediapipe/models/selfie_multiclass_256x256.tflite', import.meta.url).href;
const POSE_MODEL = new URL('mediapipe/models/pose_landmarker_lite.task', import.meta.url).href;
const FACE_MODEL = new URL('mediapipe/models/face_landmarker.task', import.meta.url).href;

// Two independent sets of engines, one per MediaPipe running mode. VIDEO mode
// is stateful (tracking assumes strictly-increasing timestamps on the same
// instance, in the same order frames actually occur), so it's only correct
// for a genuinely sequential stream — live camera/screen capture. Batch video
// export processes frames concurrently across a decode pool for throughput,
// which is fundamentally incompatible with that assumption, so it stays on
// IMAGE mode. A single instance can't switch modes after creation, hence two
// caches rather than one.
const engineCache = { IMAGE: null, VIDEO: null };
const loadPromise = { IMAGE: null, VIDEO: null };
let wantLandmarks = false;
// Use real wall-clock time for VIDEO mode timestamps so MediaPipe's temporal
// tracking model gets accurate inter-frame deltas. A +1 ms increment per call
// (the previous approach) tells the tracker every frame is 1 ms apart, which
// makes it treat every frame as a near-duplicate and suppresses the very
// motion continuity it is designed to exploit.
let videoSessionStart = 0;

// Temporal smoothing state.
// Smoothing is done on the raw per-class confidence channels at the model's
// native 256×256 resolution, BEFORE bilinear upsampling and argmax.
// This is the correct level: blending float confidence values means a pixel
// can only receive a non-background label if its EWA-smoothed confidence is
// genuinely elevated there. Blending hard argmax labels (previous approach)
// scattered old body labels into background regions when the mask changed,
// producing the "extending to non-body areas" artifact.
const CHAN_ALPHA = 0.55;    // weight on the incoming frame; 1-CHAN_ALPHA on history
const MASK_HOLD_FRAMES = 8; // max consecutive null-inference frames before giving up
let _prevChannels = null;   // Float32Array[numClasses] at model resolution
let _prevChanW = 0;
let _prevChanH = 0;
let _prevMask = null;       // last good Uint8Array classMask at output resolution (for hold)
let _prevMaskW = 0;
let _prevMaskH = 0;
let _holdCount = 0;


function quietLogs() {
    if (typeof window === 'undefined' || window.__lineartyQuiet) return;
    window.__lineartyQuiet = true;
    const origError = console.error.bind(console);
    const origWarn = console.warn.bind(console);
    const origInfo = console.info.bind(console);
    const skip = (args) => {
        const s = args.map((a) => (typeof a === 'string' ? a : '')).join(' ');
        return /XNNPACK|TensorFlow Lite|Created TensorFlow|inference_feedback_manager|W0000/i.test(s);
    };
    console.error = (...args) => { if (!skip(args)) origError(...args); };
    console.warn = (...args) => { if (!skip(args)) origWarn(...args); };
    console.info = (...args) => { if (!skip(args)) origInfo(...args); };
}

// Pose/face landmark models are an extra download plus extra per-frame
// inference on top of segmentation, for a purely optional overlay effect —
// worth skipping on a genuinely low-power device. A viewport-width media
// query was the previous proxy for that, but window width tracks browser
// zoom/split-screen, not device capability: a narrow desktop window
// incorrectly skipped landmarks, and a maximized tablet on weak hardware
// incorrectly loaded them. navigator.hardwareConcurrency/deviceMemory are
// the same real capability signals script.js already uses for worker sizing.
function skipLandmarks() {
    if (typeof navigator === 'undefined') return false;
    const cores = navigator.hardwareConcurrency || 4;
    const memoryGB = (typeof navigator.deviceMemory === 'number') ? navigator.deviceMemory : null;
    return cores <= 2 || (memoryGB !== null && memoryGB <= 2);
}

const emptyDetect = { detect: () => ({ landmarks: [], faceLandmarks: [] }) };

async function createWithDelegate(delegate, mode) {
    const wasm = await FilesetResolver.forVisionTasks(WASM_PATH);
    const common = { delegate };
    const loadPose = wantLandmarks && !skipLandmarks();
    const [seg, pose, face] = await Promise.all([
        ImageSegmenter.createFromOptions(wasm, {
            baseOptions: { modelAssetPath: SEG_MODEL, ...common },
            runningMode: mode,
            // Confidence masks (one float32 mask per class), not the hard
            // categoryMask: upsampling a per-class confidence value with
            // bilinear interpolation is correct; interpolating between two
            // label IDs (e.g. hair=1, face=3) is not — it produces a
            // spurious third class at the boundary. See upsampleClassesBilinear.
            outputCategoryMask: false,
            outputConfidenceMasks: true,
        }),
        loadPose
            ? PoseLandmarker.createFromOptions(wasm, {
                baseOptions: { modelAssetPath: POSE_MODEL, ...common },
                runningMode: mode,
                numPoses: 2,
                minPoseDetectionConfidence: 0.35,
                minPosePresenceConfidence: 0.35,
                minTrackingConfidence: 0.4,
            })
            : Promise.resolve(emptyDetect),
        loadPose
            ? FaceLandmarker.createFromOptions(wasm, {
                baseOptions: { modelAssetPath: FACE_MODEL, ...common },
                runningMode: mode,
                numFaces: 2,
                minFaceDetectionConfidence: 0.4,
                minFacePresenceConfidence: 0.4,
                minTrackingConfidence: 0.4,
            })
            : Promise.resolve(emptyDetect),
    ]);
    return {
        seg,
        pose,
        face,
        poseConn: loadPose ? PoseLandmarker.POSE_CONNECTIONS : [],
        faceContours: loadPose ? FaceLandmarker.FACE_LANDMARKS_CONTOURS : [],
        faceLips: loadPose ? FaceLandmarker.FACE_LANDMARKS_LIPS : [],
        faceLeft: loadPose ? FaceLandmarker.FACE_LANDMARKS_LEFT_EYE : [],
        faceRight: loadPose ? FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE : [],
    };
}

// options.mode: 'IMAGE' (default) or 'VIDEO'. Pass 'VIDEO' only for a
// genuinely sequential stream — see the engineCache comment above.
export async function ensureHuman(options = {}) {
    const mode = options.mode === 'VIDEO' ? 'VIDEO' : 'IMAGE';
    if (options.landmarks) wantLandmarks = true;
    const cached = engineCache[mode];
    if (cached && (!wantLandmarks || cached.poseConn.length)) return true;
    quietLogs();
    if (!loadPromise[mode] || (wantLandmarks && cached && !cached.poseConn.length)) {
        engineCache[mode] = null;
        loadPromise[mode] = (async () => {
            try {
                engineCache[mode] = await createWithDelegate('GPU', mode);
                return engineCache[mode];
            } catch {
                try {
                    engineCache[mode] = await createWithDelegate('CPU', mode);
                    return engineCache[mode];
                } catch (err) {
                    console.warn('Linearty: body maps unavailable', err);
                    engineCache[mode] = null;
                    return null;
                }
            }
        })();
    }
    return !!(await loadPromise[mode]);
}

// Upsamples the model's 6 per-class confidence channels with bilinear
// interpolation, then takes the highest-confidence class per output pixel.
// This is the correct way to upsample a categorical segmentation mask: the
// model only ever produces a 256x256 result, so every real output resolution
// needs this. Bilinear-interpolating the *hard* class labels instead (the
// previous approach) is mathematically meaningless at a boundary — averaging
// hair=1 and face=3 does not produce a valid third class — and produces
// visibly blocky, jagged mask edges once upsampled past a few hundred pixels.
function upsampleClassesBilinear(channels, sw, sh, dw, dh) {
    const numClasses = channels.length;
    const out = new Uint8Array(dw * dh);
    const scaleX = (sw - 1) / Math.max(dw - 1, 1);
    const scaleY = (sh - 1) / Math.max(dh - 1, 1);

    for (let y = 0; y < dh; y++) {
        const sy = y * scaleY;
        const y0 = Math.floor(sy);
        const y1 = Math.min(sh - 1, y0 + 1);
        const fy = sy - y0;
        const rowY0 = y0 * sw;
        const rowY1 = y1 * sw;

        for (let x = 0; x < dw; x++) {
            const sx = x * scaleX;
            const x0 = Math.floor(sx);
            const x1 = Math.min(sw - 1, x0 + 1);
            const fx = sx - x0;

            let bestClass = 0;
            let bestVal = -Infinity;
            for (let c = 0; c < numClasses; c++) {
                const ch = channels[c];
                const top = ch[rowY0 + x0] + (ch[rowY0 + x1] - ch[rowY0 + x0]) * fx;
                const bot = ch[rowY1 + x0] + (ch[rowY1 + x1] - ch[rowY1 + x0]) * fx;
                const val = top + (bot - top) * fy;
                if (val > bestVal) {
                    bestVal = val;
                    bestClass = c;
                }
            }
            out[y * dw + x] = bestClass;
        }
    }
    return out;
}

function paintDisk(buf, w, h, cx, cy, r, val) {
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(w - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(h - 1, Math.ceil(cy + r));
    const r2 = r * r;
    for (let y = y0; y <= y1; y++) {
        const dy = y - cy;
        for (let x = x0; x <= x1; x++) {
            const dx = x - cx;
            if (dx * dx + dy * dy <= r2) buf[y * w + x] = val;
        }
    }
}

function stroke(buf, w, h, x0, y0, x1, y1, radius, val) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy)));
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        paintDisk(buf, w, h, x0 + dx * t, y0 + dy * t, radius, val);
    }
}

function drawConnections(buf, w, h, landmarks, connections, radius, val, minVis = 0.35) {
    if (!connections) return;
    for (const c of connections) {
        const a = landmarks[c.start];
        const b = landmarks[c.end];
        if (!a || !b) continue;
        if ((a.visibility ?? 1) < minVis || (b.visibility ?? 1) < minVis) continue;
        stroke(buf, w, h, a.x * w, a.y * h, b.x * w, b.y * h, radius, val);
    }
}

// useVideoMode: true for a genuinely sequential stream (live camera/screen
// capture, already processed one frame at a time) — uses segmentForVideo/
// detectForVideo with a monotonically increasing timestamp, MediaPipe's own
// documented mode for video frame sequences, which also enables tracking
// continuity between frames. False (default) for batch export and single
// images, where IMAGE mode's per-call independence is what's actually true.
export function inferHuman(image, width, height, settings, useVideoMode = false) {
    const mode = useVideoMode ? 'VIDEO' : 'IMAGE';
    const engines = engineCache[mode];
    if (!engines || !settings.humanAware) return null;
    try {
        const { seg, pose, face } = engines;
        let classMask = new Uint8Array(width * height);
        let rawChannels = null;
        let maskW = 0, maskH = 0;
        let copied = false;
        const onSegmentResult = (result) => {
            const masks = result.confidenceMasks;
            if (!masks || masks.length === 0) return;
            maskW = masks[0].width;
            maskH = masks[0].height;
            rawChannels = masks.map((m) => m.getAsFloat32Array());
            // Bilinear on each class's own confidence, then argmax — correct
            // regardless of whether mw/mh already match width/height, since
            // the interpolation weight is exactly 0 wherever source and
            // target pixels align. See upsampleClassesBilinear.
            classMask = upsampleClassesBilinear(rawChannels, maskW, maskH, width, height);
            copied = true;
        };
        // Use real elapsed wall-clock time for VIDEO mode. MediaPipe's internal
        // temporal tracking uses timestamps to compute motion velocity between
        // frames — a fake +1 ms increment per call makes the tracker think
        // every frame arrives 1 ms after the last, suppressing legitimate
        // motion continuity. performance.now() gives true elapsed ms.
        let tsMs;
        if (useVideoMode) {
            if (!videoSessionStart) videoSessionStart = performance.now();
            tsMs = Math.max(1, Math.round(performance.now() - videoSessionStart + 1));
            seg.segmentForVideo(image, tsMs, onSegmentResult);
        } else {
            seg.segment(image, onSegmentResult);
        }

        if (!copied) {
            // Inference produced no mask this frame (model hiccup, scheduler
            // contention, etc.). Return the temporally-smoothed last-known mask
            // for up to MASK_HOLD_FRAMES consecutive failures so a single bad
            // frame doesn't snap the overlay to nothing.
            if (_prevMask && _prevMaskW === width && _prevMaskH === height && _holdCount < MASK_HOLD_FRAMES) {
                _holdCount++;
                const held = _prevMask.slice(); // copy so caller can't mutate state
                let person = 0;
                for (let i = 0; i < held.length; i++) if (held[i] !== HUMAN_BG) person++;
                const personRatio = person / Math.max(1, held.length);
                return {
                    width, height,
                    classMask: held,
                    extraLines: new Uint8Array(width * height),
                    personRatio,
                    hasPerson: personRatio > 0.010,
                    _held: true,
                };
            }
            return null;
        }
        _holdCount = 0;

        // Temporal smoothing at confidence-channel level (EWA before argmax).
        // Blend each of the 6 per-class float32 channels with the previous
        // frame's smoothed channels at the model's native 256×256 resolution,
        // then re-run bilinear upsample + argmax on the blended result.
        //
        // Why here and not on the output argmax mask?
        // Blending hard labels (the old approach) randomly placed stale body
        // labels in regions the current frame classifies as background, causing
        // body detection to "extend" into non-body areas. Blending floats is
        // safe because background confidence in channel 0 stays high wherever
        // the person has moved away — the EWA-smoothed argmax stays background.
        //
        // Only active in VIDEO mode (live camera / sequential video playback).
        // IMAGE mode (batch export, stills) processes each frame independently.
        let channelsToUpsample = rawChannels;
        if (useVideoMode && _prevChannels &&
            _prevChanW === maskW && _prevChanH === maskH) {
            const n = maskW * maskH;
            channelsToUpsample = rawChannels.map((ch, c) => {
                const prev = _prevChannels[c];
                const out = new Float32Array(n);
                for (let i = 0; i < n; i++) {
                    out[i] = CHAN_ALPHA * ch[i] + (1 - CHAN_ALPHA) * prev[i];
                }
                return out;
            });
        }
        // Store the (possibly smoothed) channels for the next frame.
        _prevChannels = channelsToUpsample.map((ch) => ch.slice());
        _prevChanW = maskW;
        _prevChanH = maskH;

        // If smoothing produced different channels, re-derive classMask from them.
        if (channelsToUpsample !== rawChannels) {
            classMask = upsampleClassesBilinear(channelsToUpsample, maskW, maskH, width, height);
        }
        // Save output-resolution mask for the null-frame hold path.
        _prevMask = classMask;
        _prevMaskW = width;
        _prevMaskH = height;

        let person = 0;
        for (let i = 0; i < classMask.length; i++) {
            if (classMask[i] !== HUMAN_BG) person++;
        }
        const personRatio = person / Math.max(1, classMask.length);
        const extraLines = new Uint8Array(width * height);
        if (settings.poseLines && pose && engines.poseConn.length) {
            const poses = useVideoMode ? pose.detectForVideo(image, tsMs) : pose.detect(image);
            for (const lm of poses.landmarks || []) {
                drawConnections(extraLines, width, height, lm, engines.poseConn, 1.35, 220, 0.4);
            }
        }
        if (settings.faceContours && face && engines.faceContours.length) {
            const faces = useVideoMode ? face.detectForVideo(image, tsMs) : face.detect(image);
            for (const lm of faces.faceLandmarks || []) {
                drawConnections(extraLines, width, height, lm, engines.faceContours, 0.9, 255, 0);
                drawConnections(extraLines, width, height, lm, engines.faceLips, 0.7, 200, 0);
                drawConnections(extraLines, width, height, lm, engines.faceLeft, 0.7, 240, 0);
                drawConnections(extraLines, width, height, lm, engines.faceRight, 0.7, 240, 0);
            }
        }
        // Hysteretic hasPerson thresholds — separate turn-on (2.5%) and turn-off
        // (1.0%) thresholds prevent the flag from toggling on and off when the
        // person only partially fills the frame or moves quickly through it.
        // The caller passes settings.prevHasPerson so we can apply hysteresis.
        const prevHas = settings._prevHasPerson === true;
        const hasPerson = prevHas ? personRatio > 0.010 : personRatio > 0.025;
        return {
            width,
            height,
            classMask,
            extraLines,
            personRatio,
            hasPerson,
        };
    } catch (err) {
        console.warn('Linearty: human inference failed', err);
        return null;
    }
}

// Reset temporal state when starting a new video session or switching sources.
export function resetHumanTemporal() {
    _prevChannels = null;
    _prevChanW = 0;
    _prevChanH = 0;
    _prevMask = null;
    _prevMaskW = 0;
    _prevMaskH = 0;
    _holdCount = 0;
    videoSessionStart = 0;
}



const PAL = [
    [28, 28, 32],
    [196, 146, 72],
    [196, 112, 112],
    [232, 186, 154],
    [92, 124, 164],
    [92, 164, 148],
];

export function colorizeMask(classMask, w, h) {
    const out = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < classMask.length; i++) {
        const c = PAL[classMask[i]] || PAL[0];
        const p = i * 4;
        out[p] = c[0];
        out[p + 1] = c[1];
        out[p + 2] = c[2];
        out[p + 3] = classMask[i] === 0 ? 70 : 200;
    }
    return out;
}

export { HUMAN_BG, HUMAN_HAIR, HUMAN_BODY, HUMAN_FACE, HUMAN_CLOTHES, HUMAN_OTHER };


