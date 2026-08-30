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

function skipLandmarks() {
    return typeof window !== 'undefined' && window.matchMedia('(max-width: 640px)').matches;
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
            outputCategoryMask: true,
            outputConfidenceMasks: false,
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

function upsampleNearest(src, sw, sh, dw, dh) {
    const out = new Uint8Array(dw * dh);
    for (let y = 0; y < dh; y++) {
        const sy = Math.min(sh - 1, Math.round((y * (sh - 1)) / Math.max(dh - 1, 1)));
        for (let x = 0; x < dw; x++) {
            const sx = Math.min(sw - 1, Math.round((x * (sw - 1)) / Math.max(dw - 1, 1)));
            out[y * dw + x] = src[sy * sw + sx];
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
        let mw = width;
        let mh = height;
        let copied = false;
        seg.segment(image, (result) => {
            const mask = result.categoryMask;
            if (!mask) return;
            const data = mask.getAsUint8Array();
            mw = mask.width;
            mh = mask.height;
            classMask = new Uint8Array(data);
            copied = true;
        });
        if (!copied) return null;
        if (mw !== width || mh !== height) {
            classMask = upsampleNearest(classMask, mw, mh, width, height);
        }
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
