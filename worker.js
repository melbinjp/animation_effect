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

        if (!settings.customMode) {
            // Standard presets always use bilateral pre-smoothing; pass count from preset.
            cv.bilateralFilter(this.rgb, this.smoothed, d, sigma, sigma, cv.BORDER_DEFAULT);
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
        } else {
            // Custom mode: each smooth method is independently toggled and can be
            // combined in any order — bilateral first (RGB), then Gaussian/median (gray).

            if (settings.useBilateral) {
                // N bilateral passes.  Pass 1 uses the full sigma; subsequent passes
                // use a reduced sigma that refines smoothing without over-blurring.
                // Ping-pong between this.smoothed and this.rgb (bilateral requires src ≠ dst).
                const bilateralPasses = Math.max(1, Math.min(5, settings.bilateralPasses || 2));
                cv.bilateralFilter(this.rgb, this.smoothed, d, sigma, sigma, cv.BORDER_DEFAULT);
                let bilateralResult = this.smoothed;
                let bilateralAlt = this.rgb;
                if (bilateralPasses > 1) {
                    const refineSigma = Math.max(15, Math.round(sigma * 0.5));
                    for (let p = 1; p < bilateralPasses; p++) {
                        cv.bilateralFilter(bilateralResult, bilateralAlt, d, refineSigma, refineSigma, cv.BORDER_DEFAULT);
                        const tmp = bilateralResult;
                        bilateralResult = bilateralAlt;
                        bilateralAlt = tmp;
                    }
                }
                cv.cvtColor(bilateralResult, this.gray, cv.COLOR_RGB2GRAY);
            } else {
                // No bilateral — convert directly to grayscale.
                cv.cvtColor(this.rgb, this.gray, cv.COLOR_RGB2GRAY);
            }

            // Gaussian smooth runs on the grayscale image.  Multiple passes
            // progressively strengthen the blur, preserving fine edges that
            // bilateral would suppress.
            if (settings.useGaussian) {
                const gaussianPasses = Math.max(1, Math.min(5, settings.gaussianPasses || 1));
                for (let p = 0; p < gaussianPasses; p++) {
                    cv.GaussianBlur(this.gray, this.gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
                }
            }

            // Median smooth: excellent at removing salt-and-pepper noise while
            // keeping hard edges sharp.  Each pass uses a 3×3 kernel; multiple
            // passes compound the noise-removal effect.
            if (settings.useMedian) {
                const medianPasses = Math.max(1, Math.min(3, settings.medianPasses || 1));
                for (let p = 0; p < medianPasses; p++) {
                    cv.medianBlur(this.gray, this.gray, 3);
                }
            }
        }

        // Per-frame adaptive normalization — enabled for all standard presets and
        // controllable in Custom / Experiment mode via the "Auto-normalize frames"
        // checkbox.
        //
        // The goal is to transform whatever the camera captured into a tonal
        // distribution that maximises contrast at actual edges, regardless of scene
        // exposure, lighting, or the number of scene changes in the video.
        //
        // Three cascading stages:
        //
        //   Stage 1 — Gamma lift (dark frames, mean < 80)
        //     Applies a power-law brightness boost via a 256-entry LUT so that
        //     mid-tones are raised without touching pure black/white.  Gamma 1.5
        //     gives a mild lift; gamma 3.0 is used for near-black frames.
        //
        //   Stage 2 — Histogram stretch (low-contrast/flat frames, std dev < 45)
        //     cv.normalize(NORM_MINMAX) remaps the darkest pixel to 0 and the
        //     brightest to 255, spreading the full dynamic range.  This recovers
        //     edges buried in a narrow tonal band — foggy scenes, overcast outdoor
        //     shots, monitor-lit subjects, etc.
        //
        //   Stage 3 — Adaptive CLAHE (all frames)
        //     Boosts local contrast in every 8×8 tile.  The clip limit scales
        //     inversely with mean brightness: dark frames need stronger lift (higher
        //     clip); bright/well-lit frames get a conservative clip to avoid
        //     amplifying noise.  clip = clamp(150 / mean, 1.5, 4.5).
        //
        // The three stages cascade: a very dark AND flat frame gets gamma lift first,
        // then histogram stretch, then CLAHE — all compounding to fully restore edges.
        if (settings.autoNormalize) {
            const meanMat = new cv.Mat();
            const stdMat = new cv.Mat();
            cv.meanStdDev(this.gray, meanMat, stdMat);
            const mean = meanMat.data64F[0];
            const std = stdMat.data64F[0];
            meanMat.delete();
            stdMat.delete();

            // Stage 1: gamma lift for under-exposed frames.
            if (mean < 80) {
                // gamma > 1 brightens: output = 255 * (input/255)^(1/gamma)
                // Threshold 80: frames below this are considered under-exposed.
                // Formula: mean 80 → gamma 1.5 (mild lift); mean 0 → gamma 3.0 (strong lift).
                // Divisor 40 = (80 - 0) / (3.0 - 1.0 - 1.0) maps [0,80] onto [3.0,1.5].
                const gamma = Math.max(1.5, Math.min(3.0, 1.0 + (80 - mean) / 40));
                const lut = new cv.Mat(1, 256, cv.CV_8U);
                for (let i = 0; i < 256; i++) {
                    lut.data[i] = Math.round(Math.pow(i / 255, 1 / gamma) * 255);
                }
                cv.LUT(this.gray, lut, this.gray);
                lut.delete();
            }

            // Stage 2: histogram stretch for low-contrast / flat scenes.
            if (std < 45) {
                cv.normalize(this.gray, this.gray, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
            }

            // Stage 3: adaptive CLAHE — clip limit inversely proportional to mean.
            const adaptiveClip = Math.max(1.5, Math.min(4.5, 150 / Math.max(mean, 1)));
            const adaptiveClahe = new cv.CLAHE(adaptiveClip, new cv.Size(8, 8));
            adaptiveClahe.apply(this.gray, this.gray);
            adaptiveClahe.delete();
        }

        // Light Gaussian blur on the grayscale image to suppress high-frequency
        // noise that would otherwise generate spurious thin Canny edges.
        cv.GaussianBlur(this.gray, this.gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

        // Optional: manual CLAHE boost (Custom / Experiment mode only).
        // Applies an additional round of Contrast Limited Adaptive Histogram
        // Equalisation on top of the auto-normalize stage (or alone when
        // autoNormalize is off).  The clip limit is user-tunable via the
        // "CLAHE clip limit" slider (1–6, default 2.5): a higher value lifts
        // shadow contrast more aggressively at the cost of amplifying noise.
        if (settings.darkBoost) {
            const clipLimit = Math.max(1.0, Math.min(6.0, settings.darkBoostClip || 2.5));
            const clahe = new cv.CLAHE(clipLimit, new cv.Size(8, 8));
            clahe.apply(this.gray, this.gray);
            clahe.delete();
        }

        cv.Canny(this.gray, this.edges, lowThreshold, highThreshold, 3, true);

        // Morphological closing bridges tiny gaps between nearby edge segments,
        // producing closed, cartoon-style contours around subjects.
        const closeKernel = cv.Mat.ones(3, 3, cv.CV_8U);
        cv.morphologyEx(this.edges, this.edges, cv.MORPH_CLOSE, closeKernel);
        closeKernel.delete();

        // Optional: connected-component dot removal — removes isolated edge fragments
        // (single pixels, tiny specks from skin pores, fabric grain, etc.) without
        // harming thin continuous Canny edges.  Only enabled in Custom/Experiment mode.
        //
        // Why connected components instead of morphological OPEN?
        // Morphological OPEN with a cross-shaped kernel erodes in 4 directions, which
        // destroys thin diagonal lines along with the dots it targets.  Connected
        // component analysis works differently: it labels every contiguous group of
        // white pixels, then discards only the groups whose total pixel count falls
        // below a minimum area threshold.  A continuous line — however thin — always
        // accumulates enough pixels to survive; an isolated dot or tiny speckle does
        // not.  The result is a strictly cleaner removal of dot-art artefacts with no
        // collateral loss of subject lines.
        //
        // Intensity → minimum component area (pixel count):
        //   1 (fine)   →  4 px  — removes single pixels and 2–3-pixel blobs
        //   2 (medium) → 12 px  — removes clusters up to roughly 3 × 4 pixels
        //   3 (coarse) → 30 px  — removes larger speckle patches
        if (settings.cleanSpeckles) {
            const speckleIntensity = Math.max(1, Math.min(3, settings.cleanSpecklesIntensity || 1));
            // Minimum connected-component area to survive (pixels).
            const MIN_AREA = { 1: 4, 2: 12, 3: 30 };
            const minArea = MIN_AREA[speckleIntensity];
            const labels = new cv.Mat();
            const stats = new cv.Mat();
            const centroids = new cv.Mat();
            const numLabels = cv.connectedComponentsWithStats(
                this.edges, labels, stats, centroids, 8, cv.CV_32S
            );
            const statsData = stats.data32S;
            const labelsData = labels.data32S;
            const edgeData = this.edges.data;
            // CC_STAT_AREA is the 5th column (index 4) in the stats matrix.
            // Each row in the stats matrix has CC_STATS_COLS = 5 entries:
            // [left, top, width, height, area].
            // Label 0 is the background; skip it.
            const CC_STATS_COLS = 5;
            const CC_STAT_AREA = 4;
            for (let i = 0, len = labelsData.length; i < len; i++) {
                const label = labelsData[i];
                if (label !== 0 && statsData[label * CC_STATS_COLS + CC_STAT_AREA] < minArea) {
                    edgeData[i] = 0;
                }
            }
            labels.delete();
            stats.delete();
            centroids.delete();
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
            const lwSize = new cv.Size(settings.lineWeight + 1, settings.lineWeight + 1);
            const lwKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, lwSize);
            cv.dilate(this.edges, this.edges, lwKernel);
            lwKernel.delete();
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
