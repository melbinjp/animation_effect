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
        [this.src, this.smoothed, this.gray, this.edges].forEach((mat) => {
            if (mat) {
                mat.delete();
            }
        });
        this.src = null;
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
        this.smoothed = new cv.Mat(height, width, cv.CV_8UC4);
        this.gray = new cv.Mat(height, width, cv.CV_8UC1);
        this.edges = new cv.Mat(height, width, cv.CV_8UC1);
    }

    process(rgbaData, width, height, settings) {
        this.ensureSize(width, height);
        this.src.data.set(rgbaData);

        const detailFactor = settings.detail / 62;
        const lowThreshold = Math.max(12, Math.round(settings.preset.lowThreshold / detailFactor));
        const highThreshold = Math.max(lowThreshold + 24, Math.round(settings.preset.highThreshold / detailFactor));
        const sigma = Math.max(20, Math.round(settings.preset.sigma * (0.75 + (settings.detail - 35) / 100)));

        cv.bilateralFilter(
            this.src,
            this.smoothed,
            settings.preset.bilateralDiameter,
            sigma,
            sigma,
            cv.BORDER_DEFAULT
        );
        cv.cvtColor(this.smoothed, this.gray, cv.COLOR_RGBA2GRAY);
        cv.Canny(this.gray, this.edges, lowThreshold, highThreshold, 3, false);

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
            self.postMessage({ type: 'error', id: msg.id, message: error.message });
        }
    } else if (msg.type === 'reset') {
        if (processorInstance) {
            processorInstance.reset();
        }
    }
};
