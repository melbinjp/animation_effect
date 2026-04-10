'use strict';

let processorInstance = null;

console.log('[Worker] Starting OpenCV load...');
try {
    importScripts('vendor/opencv.js');
    console.log('[Worker] OpenCV script imported, waiting for runtime initialization...');

    cv.onRuntimeInitialized = () => {
        console.log('[Worker] OpenCV runtime initialized successfully');
        processorInstance = new WorkerProcessor();
        self.postMessage({ type: 'cv-ready' });
    };
} catch (error) {
    console.error('[Worker] Failed to import OpenCV script:', error);
    self.postMessage({ type: 'cv-error', message: 'Failed to load OpenCV: ' + error.message });
}

class WorkerProcessor {
    constructor() {
        this.width = 0;
        this.height = 0;
        this.src = null;
        this.smoothed = null;
        this.gray = null;
        this.edges = null;
    }

    reset() {
        [this.src, this.rgb, this.smoothed, this.gray, this.edges].forEach((mat) => {
            if (mat) {
                mat.delete();
            }
        });
        this.src = null;
        this.rgb = null;
        this.smoothed = null;
        this.gray = null;
        this.edges = null;
        this.width = 0;
        this.height = 0;
    }

    ensureSize(width, height) {
        if (this.width === width && this.height === height) {
            return;
        }

        this.reset();
        this.width = width;
        this.height = height;
        this.src = new cv.Mat(height, width, cv.CV_8UC4);
        this.rgb = new cv.Mat(height, width, cv.CV_8UC3);
        this.smoothed = new cv.Mat(height, width, cv.CV_8UC3);
        this.gray = new cv.Mat(height, width, cv.CV_8UC1);
        this.edges = new cv.Mat(height, width, cv.CV_8UC1);
    }

    process(rgbaData, width, height, settings) {
        this.ensureSize(width, height);
        this.src.data.set(rgbaData);
        cv.cvtColor(this.src, this.rgb, cv.COLOR_RGBA2RGB);

        const detailFactor = settings.detail / 62;
        const lowThreshold = Math.max(12, Math.round(settings.preset.lowThreshold / detailFactor));
        const highThreshold = Math.max(lowThreshold + 24, Math.round(settings.preset.highThreshold / detailFactor));
        const sigma = Math.max(20, Math.round(settings.preset.sigma * (0.75 + (settings.detail - 35) / 100)));
        const d = settings.preset.bilateralDiameter;

        if (settings.fastMode) {
            // Fast mode: replace the expensive multi-pass bilateral filter with a
            // single cheap Gaussian blur on the RGB image.  The bilateral filter's
            // only role for line art is pre-smoothing the colour frame so that
            // noise (skin texture, fabric grain, etc.) doesn't produce spurious
            // Canny edges in the final line art.  A Gaussian blur achieves the
            // same noise reduction at a fraction of the cost — the bilateral
            // filter's edge-preserving property only matters for flat-colour
            // rendering, not for edge extraction.
            // A 5×5 kernel gives enough suppression for line-art work while
            // being roughly 15–20× faster than three bilateral passes at d=9.
            cv.GaussianBlur(this.rgb, this.smoothed, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
            cv.cvtColor(this.smoothed, this.gray, cv.COLOR_RGB2GRAY);
        } else {
            // First bilateral pass — smooths flat areas while preserving hard edges.
            cv.bilateralFilter(this.rgb, this.smoothed, d, sigma, sigma, cv.BORDER_DEFAULT);

            // Second bilateral pass with a reduced sigma — refines smoothing without
            // over-blurring, giving cartoonier flat regions and cleaner edge boundaries.
            if (settings.preset.smoothPasses >= 2) {
                // Keep refineSigma >= 15 to avoid destroying the edge-preserving
                // property of the bilateral filter at very low sigma values.
                const refineSigma = Math.max(15, Math.round(sigma * 0.5));
                // Ping-pong between this.smoothed and this.rgb so each call writes
                // to a different buffer — bilateral filter requires src ≠ dst.
                cv.bilateralFilter(this.smoothed, this.rgb, d, refineSigma, refineSigma, cv.BORDER_DEFAULT);
                cv.bilateralFilter(this.rgb, this.smoothed, d, refineSigma, refineSigma, cv.BORDER_DEFAULT);
            }

            cv.cvtColor(this.smoothed, this.gray, cv.COLOR_RGB2GRAY);
        }

        // Light Gaussian blur on the grayscale image to suppress high-frequency
        // noise that would otherwise generate spurious thin Canny edges.
        cv.GaussianBlur(this.gray, this.gray, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT);

        // Optional: CLAHE (Contrast Limited Adaptive Histogram Equalisation) lifts
        // the local contrast in dark regions so that Canny can see edges that would
        // otherwise be buried in shadow — bringing out subject detail that is lost
        // when the clip is underexposed or back-lit.  The clip limit and tile grid
        // size are held constant; the intensity slider controls only the edge-merge
        // step below.
        if (settings.darkBoost) {
            // clipLimit 2.5 is a well-tested mid-range value: it suppresses noise
            // amplification (clip limit prevents the histogram redistribution from
            // running away in flat regions) while still noticeably lifting contrast
            // in dark areas.  The 8×8 tile grid matches the OpenCV default and gives
            // good locality without creating visible tile boundaries at typical
            // video resolutions.
            const clahe = new cv.CLAHE(2.5, new cv.Size(8, 8));
            clahe.apply(this.gray, this.gray);
            clahe.delete();
        }

        cv.Canny(this.gray, this.edges, lowThreshold, highThreshold, 3, false);

        // Morphological closing bridges tiny gaps between nearby edge segments,
        // producing closed, cartoon-style contours around subjects.
        const closeKernel = cv.Mat.ones(3, 3, cv.CV_8U);
        cv.morphologyEx(this.edges, this.edges, cv.MORPH_CLOSE, closeKernel);
        closeKernel.delete();

        // Optional: morphological opening with a cross-shaped kernel removes isolated
        // edge fragments (single pixels, tiny specks from skin pores, fabric grain,
        // etc.).  Only enabled in Custom/Experiment mode via the "Clean speckles"
        // checkbox because the cross erosion can destroy thin continuous Canny edges
        // that are essential for subject visibility in standard modes.
        if (settings.cleanSpeckles) {
            const openKernel = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(3, 3));
            cv.morphologyEx(this.edges, this.edges, cv.MORPH_OPEN, openKernel);
            openKernel.delete();
        }

        // Optional: Merge double-edges — bridges the closely-spaced parallel edges
        // that arise when Canny detects both sides of a thick line (e.g. a clothing
        // seam or facial feature), then thins the merged region back to a single pixel.
        // This removes the "double-line" or closed-loop tube artefact that is
        // especially visible in fast mode where Gaussian pre-smoothing creates soft
        // gradients on both sides of thick features.  A larger close kernel bridges
        // the gap between the two parallel edges; a follow-up erosion trims the merged
        // blob back toward single-pixel width.
        // Intensity 1–5 controls the close kernel size (5×5 → 13×13).
        if (settings.mergeDoubleEdge) {
            const DEFAULT_MERGE_INTENSITY = 2;
            const intensity = Math.max(1, Math.min(5, settings.mergeDoubleEdgeIntensity || DEFAULT_MERGE_INTENSITY));
            // Kernel grows: intensity 1 → 5×5, 2 → 7×7, 3 → 9×9, 4 → 11×11, 5 → 13×13
            const mergeSize = 3 + intensity * 2;
            const mergeKernel = cv.Mat.ones(mergeSize, mergeSize, cv.CV_8U);
            cv.morphologyEx(this.edges, this.edges, cv.MORPH_CLOSE, mergeKernel);
            mergeKernel.delete();
            // One erosion pass to thin the merged blobs back toward single-pixel edges.
            const thinKernel = cv.Mat.ones(3, 3, cv.CV_8U);
            cv.erode(this.edges, this.edges, thinKernel);
            thinKernel.delete();
        }

        if (settings.lineWeight > 1) {
            const kernel = cv.Mat.ones(settings.lineWeight, settings.lineWeight, cv.CV_8U);
            cv.dilate(this.edges, this.edges, kernel);
            kernel.delete();
        }

        cv.bitwise_not(this.edges, this.edges);

        const bg = settings.preset.background;
        const ink = settings.preset.ink;
        // Pack RGBA into 32-bit integers for fast bulk writes.
        // ImageData stores pixels as [R, G, B, A, R, G, B, A, ...] in memory.
        // All modern CPUs, ARM devices, and WASM runtimes are little-endian, meaning
        // the least-significant byte of a Uint32 sits at the lowest memory address.
        // So writing the 32-bit value 0xAABBGGRR places byte R at offset+0, G at
        // offset+1, B at offset+2, and A at offset+3 — exactly what ImageData needs.
        const bgPx = ((255 << 24) | (bg[2] << 16) | (bg[1] << 8) | bg[0]) >>> 0;
        const inkPx = ((255 << 24) | (ink[2] << 16) | (ink[1] << 8) | ink[0]) >>> 0;
        const mask = this.edges.data;
        const out = new Uint8ClampedArray(width * height * 4);
        const out32 = new Uint32Array(out.buffer);

        for (let i = 0, len = mask.length; i < len; i++) {
            out32[i] = mask[i] > 127 ? bgPx : inkPx;
        }

        return out;
    }
}

self.onmessage = function ({ data: msg }) {
    if (msg.type === 'process') {
        if (!processorInstance) {
            self.postMessage({ type: 'error', id: msg.id, message: 'OpenCV not ready.' });
            return;
        }

        try {
            const result = processorInstance.process(msg.rgbaData, msg.width, msg.height, msg.settings);
            // Transfer the buffer (zero-copy) back to the main thread.
            self.postMessage({ type: 'result', id: msg.id, data: result }, [result.buffer]);
        } catch (error) {
            console.error('[Worker] Error during process:', error);
            let errMsg = error && error.message ? error.message : String(error);
            if (typeof error === 'number' && cv && typeof cv.exceptionFromPtr === 'function') {
                errMsg = cv.exceptionFromPtr(error).msg;
            }
            self.postMessage({ type: 'error', id: msg.id, message: errMsg });
        }
    } else if (msg.type === 'reset') {
        if (processorInstance) {
            processorInstance.reset();
        }
    }
};
