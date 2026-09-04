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

let engines = null;
let loadPromise = null;
let wantLandmarks = false;

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

async function createWithDelegate(delegate) {
    const wasm = await FilesetResolver.forVisionTasks(WASM_PATH);
    const common = { delegate };
    const loadPose = wantLandmarks && !skipLandmarks();
    const [seg, pose, face] = await Promise.all([
        ImageSegmenter.createFromOptions(wasm, {
            baseOptions: { modelAssetPath: SEG_MODEL, ...common },
            runningMode: 'IMAGE',
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
                runningMode: 'IMAGE',
                numPoses: 2,
                minPoseDetectionConfidence: 0.35,
                minPosePresenceConfidence: 0.35,
                minTrackingConfidence: 0.4,
            })
            : Promise.resolve(emptyDetect),
        loadPose
            ? FaceLandmarker.createFromOptions(wasm, {
                baseOptions: { modelAssetPath: FACE_MODEL, ...common },
                runningMode: 'IMAGE',
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

export async function ensureHuman(options = {}) {
    if (options.landmarks) wantLandmarks = true;
    if (engines && (!wantLandmarks || engines.poseConn.length)) return true;
    quietLogs();
    if (!loadPromise || (wantLandmarks && engines && !engines.poseConn.length)) {
        engines = null;
        loadPromise = (async () => {
            try {
                engines = await createWithDelegate('GPU');
                return engines;
            } catch {
                try {
                    engines = await createWithDelegate('CPU');
                    return engines;
                } catch (err) {
                    console.warn('Linearty: body maps unavailable', err);
                    engines = null;
                    return null;
                }
            }
        })();
    }
    return !!(await loadPromise);
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

export function inferHuman(image, width, height, settings) {
    if (!engines || !settings.humanAware) return null;
    try {
        const { seg, pose, face } = engines;
        let classMask = new Uint8Array(width * height);
        let copied = false;
        seg.segment(image, (result) => {
            const masks = result.confidenceMasks;
            if (!masks || masks.length === 0) return;
            const mw = masks[0].width;
            const mh = masks[0].height;
            const channels = masks.map((m) => m.getAsFloat32Array());
            // Bilinear on each class's own confidence, then argmax — correct
            // regardless of whether mw/mh already match width/height, since
            // the interpolation weight is exactly 0 wherever source and
            // target pixels align. See upsampleClassesBilinear.
            classMask = upsampleClassesBilinear(channels, mw, mh, width, height);
            copied = true;
        });
        if (!copied) return null;
        let person = 0;
        for (let i = 0; i < classMask.length; i++) {
            if (classMask[i] !== HUMAN_BG) person++;
        }
        const personRatio = person / Math.max(1, classMask.length);
        const extraLines = new Uint8Array(width * height);
        if (settings.poseLines && pose && engines.poseConn.length) {
            const poses = pose.detect(image);
            for (const lm of poses.landmarks || []) {
                drawConnections(extraLines, width, height, lm, engines.poseConn, 1.35, 220, 0.4);
            }
        }
        if (settings.faceContours && face && engines.faceContours.length) {
            const faces = face.detect(image);
            for (const lm of faces.faceLandmarks || []) {
                drawConnections(extraLines, width, height, lm, engines.faceContours, 0.9, 255, 0);
                drawConnections(extraLines, width, height, lm, engines.faceLips, 0.7, 200, 0);
                drawConnections(extraLines, width, height, lm, engines.faceLeft, 0.7, 240, 0);
                drawConnections(extraLines, width, height, lm, engines.faceRight, 0.7, 240, 0);
            }
        }
        return {
            width,
            height,
            classMask,
            extraLines,
            personRatio,
            hasPerson: personRatio > 0.012,
        };
    } catch (err) {
        console.warn('Linearty: human inference failed', err);
        return null;
    }
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
