const elements = {
    fileInput: document.getElementById('fileInput'),
    cancelBtn: document.getElementById('cancelBtn'),
    pauseBtn: document.getElementById('pauseBtn'),
    advisoryCard: document.getElementById('advisoryCard'),
    advisoryText: document.getElementById('advisoryText'),
    progressCard: document.getElementById('progressCard'),
    progressFill: document.getElementById('progressFill'),
    progressPercent: document.getElementById('progressPercent'),
    progressText: document.getElementById('progressText'),
    preset: document.getElementById('preset'),
    detail: document.getElementById('detail'),
    lineWeight: document.getElementById('lineWeight'),
    scale: document.getElementById('scale'),
    videoFps: document.getElementById('videoFps'),
    customVideoFps: document.getElementById('customVideoFps'),
    previewBtn: document.getElementById('previewBtn'),
    renderBtn: document.getElementById('renderBtn'),
    resetBtn: document.getElementById('resetBtn'),
    downloadLink: document.getElementById('download'),
    downloadCard: document.getElementById('downloadCard'),
    sourceCanvas: document.getElementById('sourceCanvas'),
    outputCanvas: document.getElementById('canvasOutput'),
    outputCard: document.querySelector('.emphasis-card'),
    outputVideo: document.getElementById('outputVideo'),
    videoResult: document.getElementById('videoResult'),
    fileMeta: document.getElementById('fileMeta'),
    dropZone: document.getElementById('dropZone'),
    workerThreadsInput: document.getElementById('workerThreads'),
    workerThreadsManual: document.getElementById('workerThreadsManual'),
    workerThreadsVal: document.getElementById('workerThreadsVal'),
    workerThreadsMax: document.getElementById('workerThreadsMax'),
    workerThreadsHint: document.getElementById('workerThreadsHint')
};

const STYLE_PRESETS = {
    manga: {
        label: 'Manga Contrast',
        background: [255, 255, 255],
        ink: [0, 0, 0],
        lowThreshold: 28,
        highThreshold: 96,
        bilateralDiameter: 9,
        sigma: 72,
        smoothPasses: 2
    },
    studio: {
        label: 'Studio Ink',
        background: [248, 245, 237],
        ink: [23, 33, 47],
        lowThreshold: 38,
        highThreshold: 118,
        bilateralDiameter: 7,
        sigma: 52,
        smoothPasses: 2
    },
    neon: {
        label: 'Neon Pop',
        background: [10, 8, 22],
        ink: [0, 230, 200],
        lowThreshold: 22,
        highThreshold: 80,
        bilateralDiameter: 9,
        sigma: 68,
        smoothPasses: 2
    },
    warm: {
        label: 'Warm Sketch',
        background: [255, 248, 232],
        ink: [102, 48, 14],
        lowThreshold: 32,
        highThreshold: 104,
        bilateralDiameter: 7,
        sigma: 58,
        smoothPasses: 2
    },
    vivid: {
        label: 'Vivid Toon',
        background: [255, 255, 255],
        ink: [26, 26, 180],
        lowThreshold: 18,
        highThreshold: 66,
        bilateralDiameter: 11,
        sigma: 96,
        smoothPasses: 2
    },
    blueprint: {
        label: 'Blueprint Draft',
        background: [225, 237, 245],
        ink: [19, 57, 92],
        lowThreshold: 48,
        highThreshold: 144,
        bilateralDiameter: 5,
        sigma: 44,
        smoothPasses: 1
    },
    custom: {
        label: 'Custom / Experiment',
        background: [255, 255, 255],
        ink: [0, 0, 0],
        lowThreshold: 28,
        highThreshold: 96,
        bilateralDiameter: 9,
        sigma: 72,
        smoothPasses: 2
    }
};

const state = {
    cvReady: false,
    ffmpegReady: false,
    ffmpeg: null,
    selectedFile: null,
    fileKind: null,
    sourceImage: null,
    sourceVideo: null,
    sourceUrl: '',
    outputUrl: '',
    videoPreviewUrl: '',
    processing: false,
    cancelRequested: false,
    pauseRequested: false,
    activeJobId: '',
    beforeUnloadAttached: false,
    mediaWidth: null,
    mediaHeight: null,
    activeDecodePool: null,
    // Progress-bar mapping for FFmpeg 'progress' events.  renderVideoExport
    // updates these before each ffmpeg.exec() call so the bar moves in the
    // correct sub-range (encoding vs. final concat/mux).
    encodePhaseOffset: 90,
    encodePhaseScale: 0.1
};

const PRODUCTION_LIMITS = {
    recommendedVideoDurationSeconds: 20,
    hardVideoDurationSeconds: 90,
    recommendedVideoFrames: 540,
    hardVideoFrames: 1800,
    recommendedImageMegaPixels: 8,
    recommendedVideoMegaPixels: 3,
    hardFileSizeBytes: 250 * 1024 * 1024
};

// Safe default cap on worker count.  Even on a 16-core machine spawning 16
// WebGPU workers simultaneously saturates the GPU command queue and causes
// device-lost errors.  Users can always raise the limit manually.
const DEFAULT_SAFE_WORKER_CAP = 4;

// Compute the recommended number of Web Workers based on the number of logical
// CPU cores and the amount of RAM reported by the browser.
//
// Core budget: reserve 2 logical cores (one for the main thread / UI, one for
// the FFmpeg encoder) so the UI stays responsive during rendering.
//
// Memory budget: each worker loads a separate OpenCV WASM instance (~30 MB) and
// holds several CV matrices per in-flight frame (~50 MB at typical resolutions),
// so we budget ~80 MB per worker.  We reserve ~300 MB for the browser, main
// thread, and FFmpeg.  navigator.deviceMemory is capped at 8 by the spec; when
// it reports 8 GB we assume memory is not the bottleneck.
//
// Returns { auto, max, cores, memoryGB }
//   auto      — recommended worker count
//   max       — core-based upper limit (used as the slider ceiling)
//   cores     — raw navigator.hardwareConcurrency value
//   memoryGB  — navigator.deviceMemory (null if unavailable)
function computeOptimalWorkers() {
    const hw = navigator.hardwareConcurrency || 2;
    // Reserve 2 cores for UI/FFmpeg; always allow at least 1 worker.
    const maxFromCores = Math.max(1, hw - 2);

    const reportedMemoryGB = (typeof navigator.deviceMemory === 'number') ? navigator.deviceMemory : null;

    let maxFromMemory = maxFromCores; // default: memory constraint not tighter than cores
    if (reportedMemoryGB !== null && reportedMemoryGB < 8) {
        const deviceMemoryMB = reportedMemoryGB * 1024;
        // Use up to 65 % of available RAM; subtract a fixed overhead for the
        // browser runtime, main thread, and FFmpeg (~300 MB).
        const availableForWorkersMB = Math.max(0, deviceMemoryMB * 0.65 - 300);
        maxFromMemory = Math.max(1, Math.floor(availableForWorkersMB / 80));
    }

    // Cap at DEFAULT_SAFE_WORKER_CAP to prevent GPU device-lost crashes.
    // The user can always raise this via the slider.
    const auto = Math.min(maxFromCores, maxFromMemory, DEFAULT_SAFE_WORKER_CAP);
    return { auto, max: maxFromCores, cores: hw, memoryGB: reportedMemoryGB };
}

// Compute per-worker memory requirement and recommended worker count for a
// specific media resolution.  Call this after loading a file when dimensions
// are known to produce a more accurate recommendation than the device-only
// estimate in computeOptimalWorkers().
//
// Returns { auto, max, perWorkerMB }
//   auto        — recommended worker count for this resolution
//   max         — upper safe limit based on available RAM and cores
//   perWorkerMB — estimated RAM each worker will use for this resolution
function computeWorkersForResolution(width, height) {
    const frameSizeMB = (width * height * 4) / (1024 * 1024);
    // OpenCV processing uses ~5 intermediate matrices (RGBA source, greyscale,
    // blur, edges, output) plus the WASM module (~30 MB overhead per worker).
    const perWorkerMB = Math.ceil(frameSizeMB * 5 + 30);
    const { max, memoryGB } = workerConfig;
    let maxFromMemory = max;
    if (memoryGB !== null && memoryGB < 8) {
        const deviceMemoryMB = memoryGB * 1024;
        const availableForWorkersMB = Math.max(0, deviceMemoryMB * 0.65 - 300);
        maxFromMemory = Math.max(1, Math.floor(availableForWorkersMB / perWorkerMB));
    }
    const safeMax = Math.min(max, maxFromMemory);
    const auto = Math.min(safeMax, DEFAULT_SAFE_WORKER_CAP);
    return { auto, max: safeMax, perWorkerMB };
}

// Semaphore used to cap the number of concurrently in-flight frame renders
// so that the worker pool and main-thread memory usage stay bounded.
class Semaphore {
    constructor(n) {
        this._count = n;
        this._queue = [];
    }
    acquire() {
        if (this._count > 0) {
            this._count--;
            return Promise.resolve();
        }
        return new Promise((resolve) => this._queue.push(resolve));
    }
    release() {
        if (this._queue.length > 0) {
            this._queue.shift()();
        } else {
            this._count++;
        }
    }
}

// LineArtProcessor manages a pool of Web Workers so that multiple video frames
// can be processed in parallel, fully utilising all available CPU cores.
// Pixel data is transferred as zero-copy ArrayBuffer Transferables.
class LineArtProcessor {
    constructor(concurrency) {
        this._concurrency = Math.max(1, concurrency || 1);

        this._pending = new Map();
        this._idCounter = 0;
        this._initTimeout = null;
        this._loadingStartTime = Date.now();
        this._readyCount = 0;

        // Pool management: indices of currently idle workers.
        this._freePool = [];
        // Queue of resolve-fns waiting for a free worker index.
        this._waitQueue = [];
        // Indices of workers that should be terminated instead of returned to
        // the pool when they next complete a task (used by resize() scale-down).
        this._drainSet = new Set();

        console.log(`[Main] Spawning ${this._concurrency} worker(s)`);

        this._workers = Array.from({ length: this._concurrency }, (_, i) =>
            this._spawnWorker(i)
        );

        // Timeout in case no worker ever signals cv-ready.
        this._initTimeout = setTimeout(() => {
            if (!state.cvReady) {
                const elapsed = Math.round((Date.now() - this._loadingStartTime) / 1000);
                console.error(`[Main] OpenCV initialization timeout after ${elapsed}s`);
                setAdvisory('Processing engine failed to load. Please check your internet connection and reload the page.', 'error');
                elements.dropZone.classList.remove('is-loading');
            }
        }, 30000);
    }

    get concurrency() { return this._concurrency; }

    _spawnWorker(index) {
        // gpu-worker.js tries WebGPU first and transparently falls back to the
        // OpenCV CPU pipeline when WebGPU is unavailable, so no branch is needed
        // here.  Replace with 'worker.js' to force CPU-only processing.
        const worker = new Worker('gpu-worker.js');

        worker.onmessage = (event) => {
            const msg = event && event.data;
            if (!msg) return;

            if (msg.type === 'cv-ready') {
                this._readyCount++;
                if (this._readyCount === 1) {
                    // First worker ready → enable the UI.
                    const elapsed = Math.round((Date.now() - this._loadingStartTime) / 1000);
                    console.log(`[Main] Worker 0 ready after ${elapsed}s`);
                    if (this._initTimeout) {
                        clearTimeout(this._initTimeout);
                        this._initTimeout = null;
                    }
                    state.cvReady = true;
                    refreshActions();
                }
                console.log(`[Main] Worker ${index} ready (${this._readyCount}/${this._concurrency})`);
                this._releaseWorker(index);
                return;
            }

            if (msg.type === 'cv-error') {
                console.error(`[Main] Worker ${index} cv-error:`, msg.message);
                if (this._initTimeout) {
                    clearTimeout(this._initTimeout);
                    this._initTimeout = null;
                }
                if (this._readyCount === 0) {
                    setAdvisory('Failed to load processing engine: ' + msg.message, 'error');
                    elements.dropZone.classList.remove('is-loading');
                }
                return;
            }

            if (msg.id !== undefined) {
                const entry = this._pending.get(msg.id);
                this._pending.delete(msg.id);
                // Free the worker before resolving so the pool can immediately
                // accept the next queued task.
                this._releaseWorker(index);
                if (!entry) return;
                if (msg.type === 'result') {
                    entry.resolve(msg.data);
                } else if (msg.type === 'error') {
                    entry.reject(new Error(msg.message));
                }
            }
        };

        worker.onerror = (error) => {
            console.error(`[Main] Worker ${index} error:`, error);
            if (this._initTimeout) {
                clearTimeout(this._initTimeout);
                this._initTimeout = null;
            }
            for (const [, { reject }] of this._pending) {
                reject(new Error('Processing worker error.'));
            }
            this._pending.clear();
            setAdvisory('Processing worker error. Please reload the page.', 'error');
        };

        return worker;
    }

    // Add a worker back to the idle pool (or hand it straight to a waiter).
    _releaseWorker(index) {
        // Guard against double-release (e.g. racing reset + normal completion).
        if (this._freePool.includes(index)) return;
        // If this worker was marked for removal by resize(), terminate it now.
        if (this._drainSet.has(index)) {
            this._drainSet.delete(index);
            this._workers[index].terminate();
            return;
        }
        if (this._waitQueue.length > 0) {
            this._waitQueue.shift()(index);
        } else {
            this._freePool.push(index);
        }
    }

    // Return a Promise that resolves to a free worker index.
    _acquireWorker() {
        if (this._freePool.length > 0) {
            return Promise.resolve(this._freePool.shift());
        }
        return new Promise((resolve) => this._waitQueue.push(resolve));
    }

    reset() {
        for (const [, { reject }] of this._pending) {
            reject(new Error('Render cancelled.'));
        }
        this._pending.clear();
        // Unblock any tasks that are waiting for a free worker.
        const waiting = this._waitQueue.splice(0);
        waiting.forEach((resolve) => resolve(-1));
        this._workers.forEach((w) => w.postMessage({ type: 'reset' }));
        // Busy workers will call _releaseWorker when their in-progress frame
        // completes; idle workers are already in _freePool.
    }

    // Hard-stop all workers and clear all internal state.  Call before
    // recreating the pool with a different concurrency level.
    terminate() {
        if (this._initTimeout) {
            clearTimeout(this._initTimeout);
            this._initTimeout = null;
        }
        for (const [, { reject }] of this._pending) {
            reject(new Error('Render cancelled.'));
        }
        this._pending.clear();
        this._waitQueue.splice(0).forEach((resolve) => resolve(-1));
        this._freePool = [];
        this._drainSet.clear();
        this._workers.forEach((w) => w.terminate());
        this._workers = [];
    }

    // Dynamically change the number of active workers without interrupting
    // any in-flight processing.
    //
    // Scale-up  — new workers are spawned immediately and added to the pool
    //             once their WebGPU pipeline is ready.
    // Scale-down — excess workers are terminated as soon as they finish their
    //              current task (they are added to _drainSet and never returned
    //              to the free pool).  Idle excess workers are stopped at once.
    resize(n) {
        n = Math.max(1, n);
        if (n === this._concurrency) return;

        if (n > this._concurrency) {
            // Scale up: spawn additional workers.
            const startIdx = this._workers.length;
            const toAdd = n - this._concurrency;
            this._concurrency = n;
            for (let i = 0; i < toAdd; i++) {
                this._workers.push(this._spawnWorker(startIdx + i));
            }
        } else {
            // Scale down: drain the highest-indexed workers.
            const toRemove = this._concurrency - n;
            this._concurrency = n;
            for (let i = 0; i < toRemove; i++) {
                const workerIdx = this._workers.length - 1 - i;
                const freeIdx = this._freePool.indexOf(workerIdx);
                if (freeIdx !== -1) {
                    // Worker is currently idle — terminate it immediately.
                    this._freePool.splice(freeIdx, 1);
                    this._workers[workerIdx].terminate();
                } else {
                    // Worker is busy — mark for termination after its task.
                    this._drainSet.add(workerIdx);
                }
            }
        }
    }

    // Returns Promise<Uint8ClampedArray> — the raw RGBA output without writing
    // to any canvas.  Pixels are extracted from sourceCanvas synchronously
    // (before the first await), so the caller may safely overwrite sourceCanvas
    // on the very next iteration of an async loop.
    async renderToData(sourceCanvas, settings) {
        const width = sourceCanvas.width;
        const height = sourceCanvas.height;
        const imageData = sourceCanvas
            .getContext('2d', { willReadFrequently: true })
            .getImageData(0, 0, width, height); // synchronous pixel copy

        const id = this._idCounter++;
        const workerIndex = await this._acquireWorker();
        if (workerIndex === -1) {
            throw new Error('Render cancelled.');
        }

        // `await` the inner promise so V8 attaches a rejection handler to it
        // before any microtask checkpoint.  Without `await`, returning a bare
        // Promise from an async function can trigger spurious "Unhandled
        // promise rejection" reports in Chrome/V8 when reset() rejects the
        // entry while the microtask queue is being drained.
        return await new Promise((resolve, reject) => {
            this._pending.set(id, { resolve, reject });
            // Transfer the pixel buffer zero-copy to the chosen worker.
            this._workers[workerIndex].postMessage(
                { type: 'process', id, rgbaData: imageData.data, width, height, settings },
                [imageData.data.buffer]
            );
        });
    }

    // Convenience wrapper: render to a visible canvas (used for single-frame preview).
    async render(sourceCanvas, destinationCanvas, settings) {
        const width = sourceCanvas.width;
        const height = sourceCanvas.height;
        const rawData = await this.renderToData(sourceCanvas, settings);
        destinationCanvas.width = width;
        destinationCanvas.height = height;
        destinationCanvas.getContext('2d').putImageData(
            new ImageData(rawData, width, height), 0, 0
        );
    }
}

const workerConfig = computeOptimalWorkers();
let processor = new LineArtProcessor(workerConfig.auto);

function setBusy(isBusy, isVideo = false) {
    state.processing = isBusy;
    elements.cancelBtn.hidden = !isBusy;
    elements.cancelBtn.disabled = !isBusy;
    // Pause is only meaningful during video renders (has a multi-frame loop).
    elements.pauseBtn.hidden = !(isBusy && isVideo);
    elements.pauseBtn.disabled = !isBusy;
    if (!isBusy) {
        state.pauseRequested = false;
        elements.pauseBtn.textContent = 'Pause';
    }
    refreshActions();
}

function refreshActions() {
    const hasFile = Boolean(state.selectedFile);
    const notReady = !state.cvReady || state.processing;
    elements.previewBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.renderBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.fileInput.disabled = notReady;
    // Worker controls stay enabled during processing so the user can adjust
    // concurrency dynamically mid-render without a full pool rebuild.
    elements.workerThreadsInput.disabled = !state.cvReady;
    elements.workerThreadsManual.disabled = !state.cvReady;
    elements.dropZone.classList.toggle('is-loading', !state.cvReady);
}

function updateUnloadProtection() {
    if (!state.beforeUnloadAttached && state.processing) {
        window.addEventListener('beforeunload', beforeUnloadHandler);
        state.beforeUnloadAttached = true;
        return;
    }

    if (state.beforeUnloadAttached && !state.processing) {
        window.removeEventListener('beforeunload', beforeUnloadHandler);
        state.beforeUnloadAttached = false;
    }
}

function beforeUnloadHandler(event) {
    if (!state.processing) {
        return;
    }

    event.preventDefault();
    event.returnValue = '';
}

function revokeUrl(key) {
    if (state[key]) {
        URL.revokeObjectURL(state[key]);
        state[key] = '';
    }
}

function setResultGlow(active) {
    if (active) {
        elements.outputCard.classList.add('has-result');
    } else {
        elements.outputCard.classList.remove('has-result');
    }
}

function clearRenderedOutput() {
    revokeUrl('outputUrl');
    revokeUrl('videoPreviewUrl');
    elements.downloadCard.hidden = true;
    elements.downloadLink.removeAttribute('href');
    elements.downloadLink.removeAttribute('download');
    elements.videoResult.hidden = true;
    elements.outputVideo.removeAttribute('src');
    elements.outputVideo.load();
}

function updateFileMeta(message) {
    elements.fileMeta.textContent = message;
}

function setAdvisory(message, tone = 'info') {
    elements.advisoryText.textContent = message;
    elements.advisoryCard.dataset.tone = tone;
}

function setProgress(value, message) {
    const clamped = Math.min(100, Math.max(0, Math.round(value)));
    elements.progressCard.hidden = false;
    elements.progressFill.style.width = `${clamped}%`;
    elements.progressPercent.textContent = `${clamped}%`;
    elements.progressText.textContent = message;
}

function resetProgress() {
    elements.progressCard.hidden = true;
    elements.progressFill.style.width = '0%';
    elements.progressPercent.textContent = '0%';
    elements.progressText.textContent = 'Idle';
}

function getMediaType(file) {
    if (file.type.startsWith('image/')) {
        return 'image';
    }

    if (file.type.startsWith('video/')) {
        return 'video';
    }

    return null;
}

function hexToRgb(hex) {
    const n = parseInt(hex.replace('#', ''), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function getCustomPreset() {
    return {
        label: 'Custom / Experiment',
        background: hexToRgb(document.getElementById('customBg').value),
        ink: hexToRgb(document.getElementById('customInk').value),
        lowThreshold: Number(document.getElementById('customLowThresh').value),
        highThreshold: Number(document.getElementById('customHighThresh').value),
        bilateralDiameter: Number(document.getElementById('customBilateral').value),
        sigma: Number(document.getElementById('customSigma').value)
    };
}

function getSettings() {
    const presetKey = elements.preset.value;
    const preset = presetKey === 'custom' ? getCustomPreset() : STYLE_PRESETS[presetKey];

    let fps = 18;
    if (elements.videoFps.value === 'original') {
        // Fallback to 30 if we can't extract original, though usually
        // FFmpeg extraction just handles the source frames, but for simplicity
        // here we assume a reasonable default or 30 if original isn't accessible
        // natively via the HTML video element. We will attempt to rely on duration/frames if possible.
        fps = 30; // Native video FPS is not accessible in standard JS, so we use 30 as a default "original" proxy for computations.
    } else if (elements.videoFps.value === 'custom') {
        fps = Number(elements.customVideoFps.value) || 30;
    } else {
        fps = Number(elements.videoFps.value);
    }

    return {
        preset,
        detail: Number(elements.detail.value),
        lineWeight: Number(elements.lineWeight.value),
        scale: Number(elements.scale.value),
        videoFps: fps,
        isOriginalFps: elements.videoFps.value === 'original',
        customMode: presetKey === 'custom',
        useBilateral: presetKey === 'custom' && document.getElementById('customUseBilateral').checked,
        bilateralPasses: Number(document.getElementById('customBilateralPasses').value),
        useGaussian: presetKey === 'custom' && document.getElementById('customUseGaussian').checked,
        gaussianPasses: Number(document.getElementById('customGaussianPasses').value),
        useMedian: presetKey === 'custom' && document.getElementById('customUseMedian').checked,
        medianPasses: Number(document.getElementById('customMedianPasses').value),
        cleanSpeckles: presetKey === 'custom' && document.getElementById('customCleanSpeckles').checked,
        cleanSpecklesIntensity: Number(document.getElementById('customCleanSpecklesIntensity').value),
        autoNormalize: presetKey !== 'custom' || document.getElementById('customAutoNormalize').checked,
        darkBoost: presetKey === 'custom' && document.getElementById('customDarkBoost').checked,
        darkBoostClip: Number(document.getElementById('customDarkBoostClip').value),
        mergeDoubleEdge: presetKey === 'custom' && document.getElementById('customMergeDoubleEdge').checked,
        mergeDoubleEdgeIntensity: Number(document.getElementById('customMergeDoubleEdgeIntensity').value)
    };
}

function computeScaledSize(width, height, scale, noCap = false) {
    const largestSide = Math.max(width, height);
    const dimensionCap = 1600;
    const capRatio = (!noCap && largestSide > dimensionCap) ? dimensionCap / largestSide : 1;
    const ratio = Math.min(1, scale * capRatio);

    return {
        width: Math.max(1, Math.round(width * ratio)),
        height: Math.max(1, Math.round(height * ratio))
    };
}

function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return '0 B';
    }

    const units = ['B', 'KB', 'MB', 'GB'];
    const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / (1024 ** exponent);
    return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function summarizeWorkload() {
    if (!state.selectedFile) {
        setAdvisory('Select a file to estimate browser workload.', 'info');
        return;
    }

    const settings = getSettings();

    if (state.fileKind === 'image' && state.sourceImage) {
        const size = computeScaledSize(state.sourceImage.naturalWidth, state.sourceImage.naturalHeight, settings.scale, settings.customMode);
        const megaPixels = (size.width * size.height) / 1000000;
        const tone = megaPixels > PRODUCTION_LIMITS.recommendedImageMegaPixels ? 'warn' : 'success';
        const guidance = megaPixels > PRODUCTION_LIMITS.recommendedImageMegaPixels
            ? 'Large image. Consider 75% or 50% render size for faster exports.'
            : 'Image workload is within the browser-friendly range.';
        setAdvisory(
            `${guidance} Render target: ${size.width}×${size.height} at ${megaPixels.toFixed(1)} MP. Source file: ${formatFileSize(state.selectedFile.size)}.`,
            tone
        );
        return;
    }

    if (state.fileKind === 'video' && state.sourceVideo) {
        const size = computeScaledSize(state.sourceVideo.videoWidth, state.sourceVideo.videoHeight, settings.scale, settings.customMode);
        const totalFrames = Math.max(1, Math.floor(state.sourceVideo.duration * settings.videoFps));
        const frameMegaPixels = (size.width * size.height) / 1000000;
        let tone = 'success';
        let guidance = 'Workload looks reasonable for browser-side rendering.';

        if (
            state.sourceVideo.duration > PRODUCTION_LIMITS.recommendedVideoDurationSeconds ||
            totalFrames > PRODUCTION_LIMITS.recommendedVideoFrames ||
            frameMegaPixels > PRODUCTION_LIMITS.recommendedVideoMegaPixels
        ) {
            tone = 'warn';
            guidance = 'Heavy render. Use lower FPS or render size for better reliability.';
        }

        if (
            state.sourceVideo.duration > PRODUCTION_LIMITS.hardVideoDurationSeconds ||
            totalFrames > PRODUCTION_LIMITS.hardVideoFrames ||
            state.selectedFile.size > PRODUCTION_LIMITS.hardFileSizeBytes
        ) {
            tone = 'warn';
            guidance = 'This clip is near the practical browser limit. Expect long processing time and high memory use.';
        }

        setAdvisory(
            `${guidance} Render target: ${size.width}×${size.height}, ${totalFrames} frames at ${settings.videoFps} FPS, ${state.sourceVideo.duration.toFixed(1)}s duration, file size ${formatFileSize(state.selectedFile.size)}.`,
            tone
        );
        return;
    }

    setAdvisory('Select a supported image or video file.', 'warn');
}

function assertWithinOperationalLimits(file) {
    const isCustomMode = elements.preset.value === 'custom';
    if (!isCustomMode && file.size > PRODUCTION_LIMITS.hardFileSizeBytes) {
        throw new Error('File is too large for reliable browser-side processing. Keep uploads under 250 MB. Switch to the "Custom / Experiment" preset to bypass this limit.');
    }
}

function drawEmptyCanvas(canvas, label) {
    const context = canvas.getContext('2d');
    const width = 960;
    const height = 720;
    canvas.width = width;
    canvas.height = height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#f7f1e8';
    context.fillRect(0, 0, width, height);
    context.fillStyle = '#425061';
    context.font = '600 28px Segoe UI';
    context.textAlign = 'center';
    context.fillText(label, width / 2, height / 2);
}

function drawMediaToCanvas(media, canvas, scale, noCap = false) {
    const naturalWidth = media.videoWidth || media.naturalWidth || media.width;
    const naturalHeight = media.videoHeight || media.naturalHeight || media.height;
    const size = computeScaledSize(naturalWidth, naturalHeight, scale, noCap);
    const context = canvas.getContext('2d');

    canvas.width = size.width;
    canvas.height = size.height;
    context.clearRect(0, 0, size.width, size.height);
    context.drawImage(media, 0, 0, size.width, size.height);

    return size;
}

function canvasToBlob(canvas, type) {
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
                return;
            }

            reject(new Error('Canvas export failed.'));
        }, type);
    });
}

function waitForMediaEvent(media, eventName) {
    return new Promise((resolve, reject) => {
        const onReady = () => {
            media.removeEventListener('error', onError);
            resolve();
        };

        const onError = () => {
            media.removeEventListener(eventName, onReady);
            reject(new Error(`Failed while waiting for ${eventName}.`));
        };

        media.addEventListener(eventName, onReady, { once: true });
        media.addEventListener('error', onError, { once: true });
    });
}

async function seekVideo(video, timeInSeconds) {
    // Cloned video elements in the decode pool may not have loaded their
    // metadata yet the first time they are acquired.  videoWidth/videoHeight
    // remain 0 until readyState >= HAVE_METADATA (1), which causes
    // computeScaledSize to return 1×1 and FFmpeg to fail.  Waiting here
    // ensures dimensions are available before drawMediaToCanvas is called.
    if (video.readyState < 1) {
        await waitForMediaEvent(video, 'loadedmetadata');
    }

    if (Math.abs(video.currentTime - timeInSeconds) < 0.001) {
        return;
    }

    const seekPromise = waitForMediaEvent(video, 'seeked');
    video.currentTime = timeInSeconds;
    await seekPromise;
}

function sanitizeBaseName(fileName) {
    return fileName.replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-').replace(/^-+|-+$/g, '') || 'line-art';
}

function setDownload(blob, fileName) {
    revokeUrl('outputUrl');
    state.outputUrl = URL.createObjectURL(blob);
    elements.downloadLink.href = state.outputUrl;
    elements.downloadLink.download = fileName;
    elements.downloadCard.hidden = false;
}

function requestCancel() {
    if (!state.processing) {
        return;
    }

    state.cancelRequested = true;
    state.pauseRequested = false;
    elements.pauseBtn.textContent = 'Pause';
    console.log('Cancel requested. Finishing the current step...', 'warn');
    setProgress(getCurrentProgress(), 'Stopping current job...');

    processor.reset();
    // Do NOT call ffmpeg.terminate() here — terminating the FFmpeg worker
    // while operations are in-flight causes unhandled promise rejections
    // inside ffmpeg.js that cannot be caught from user code.  Instead we
    // rely on throwIfCancelled() checks in the render loop to abort before
    // the encode step starts, and let any already-started writeFile calls
    // finish gracefully so the FFmpeg instance stays healthy for the next render.
}

function throwIfCancelled() {
    if (state.cancelRequested) {
        throw new Error('Render cancelled.');
    }
}

// Suspend the caller until the user resumes (or cancels).
// Polls every PAUSE_POLL_INTERVAL_MS to keep CPU use negligible while paused.
const PAUSE_POLL_INTERVAL_MS = 200;
async function waitIfPaused() {
    while (state.pauseRequested && !state.cancelRequested) {
        await new Promise((resolve) => setTimeout(resolve, PAUSE_POLL_INTERVAL_MS));
    }
}

function getCurrentProgress() {
    return Number(elements.progressPercent.textContent.replace('%', '')) || 0;
}

async function safeDeleteFile(ffmpeg, filePath) {
    try {
        await ffmpeg.deleteFile(filePath);
    } catch (error) {
        return;
    }
}

async function cleanupFfmpegJob(ffmpeg, jobId, segCount, inputPath, outputPath) {
    if (!ffmpeg || !jobId) {
        return;
    }

    // Delete per-segment MP4 files and the concat list written by the parallel
    // encode path.  safeDeleteFile silently ignores files that do not exist, so
    // this is safe whether parallel encoding ran or not.
    const fileDels = [];
    for (let s = 0; s < segCount; s++) {
        fileDels.push(safeDeleteFile(ffmpeg, `${jobId}/seg${s}.mp4`));
    }
    if (segCount > 1) {
        fileDels.push(safeDeleteFile(ffmpeg, `${jobId}/concat.txt`));
    }
    fileDels.push(safeDeleteFile(ffmpeg, inputPath));
    fileDels.push(safeDeleteFile(ffmpeg, outputPath));
    await Promise.all(fileDels);

    try {
        await ffmpeg.deleteDir(jobId);
    } catch (_) {}
}

function getFFmpegClass() {
    const namespace = window.FFmpegWASM || window.FFmpeg;
    return namespace?.FFmpeg || namespace;
}

async function loadFFmpeg() {
    if (state.ffmpegReady) {
        return state.ffmpeg;
    }

    const FFmpegClass = getFFmpegClass();
    if (typeof FFmpegClass !== 'function') {
        throw new Error('FFmpeg runtime was not found in vendor/ffmpeg.js.');
    }

    if (!state.ffmpeg) {
        state.ffmpeg = new FFmpegClass();
        state.ffmpeg.on('log', ({ message }) => {
            if (message.startsWith('frame=')) {
                console.log(`Encoding video... ${message}`, 'info');
            }
        });
        state.ffmpeg.on('progress', ({ progress }) => {
            if (typeof progress === 'number') {
                const pct = Math.min(100, Math.max(0, Math.round(progress * 100)));
                const mapped = state.encodePhaseOffset + Math.round(pct * state.encodePhaseScale);
                setProgress(mapped, 'Encoding...');
            }
        });
    }

    console.log('Loading FFmpeg video export engine...', 'info');
    const getAssetUrl = (path) => new URL(path, window.location.href).href;
    await state.ffmpeg.load({
        coreURL: getAssetUrl('vendor/ffmpeg-core.js'),
        wasmURL: getAssetUrl('vendor/ffmpeg-core.wasm')
    });

    state.ffmpegReady = true;
    refreshActions();
    console.log('OpenCV ready. Video export engine loaded.', 'success');
    return state.ffmpeg;
}

// VideoDecodePool — manages a pool of cloned HTMLVideoElement instances so
// that multiple frames can be sought and drawn in parallel.  Each caller
// calls acquire(), which resolves to { video, release }.  The caller seeks,
// draws, then calls release() to return the element to the pool.
//
// All elements share the same object-URL src and are created from the single
// `sourceVideo` element stored in state, so no extra network fetch occurs.
class VideoDecodePool {
    constructor(size, srcVideo) {
        this._size = size;
        this._srcVideo = srcVideo;
        this._free = [];
        this._waiting = [];

        for (let i = 0; i < size; i++) {
            const v = document.createElement('video');
            v.preload = 'auto';
            v.muted = true;
            v.playsInline = true;
            v.src = srcVideo.src;
            this._free.push(v);
        }
    }

    acquire() {
        if (this._free.length > 0) {
            const video = this._free.shift();
            const release = () => this._release(video);
            return Promise.resolve({ video, release });
        }
        return new Promise((resolve) => {
            this._waiting.push((video) => {
                resolve({ video, release: () => this._release(video) });
            });
        });
    }

    _release(video) {
        if (this._waiting.length > 0) {
            this._waiting.shift()(video);
        } else {
            this._free.push(video);
        }
    }

    destroy() {
        this._free.forEach((v) => {
            v.src = '';
            v.load();
        });
        this._free = [];
    }

    // Dynamically add or remove video elements from the pool.
    // Scale-up adds immediately-usable elements.
    // Scale-down removes idle elements; busy elements are left running
    // (they will simply not be returned to the pool if the pool is destroyed).
    resize(n) {
        n = Math.max(1, n);
        if (n === this._size) return;
        if (n > this._size) {
            const toAdd = n - this._size;
            for (let i = 0; i < toAdd; i++) {
                const v = document.createElement('video');
                v.preload = 'auto';
                v.muted = true;
                v.playsInline = true;
                v.src = this._srcVideo.src;
                this._free.push(v);
            }
        } else {
            const toRemove = this._size - n;
            for (let i = 0; i < toRemove && this._free.length > 0; i++) {
                const v = this._free.pop();
                v.src = '';
                v.load();
            }
        }
        this._size = n;
    }
}

async function readSelectedFile(file) {
    assertWithinOperationalLimits(file);
    revokeUrl('sourceUrl');
    state.sourceUrl = URL.createObjectURL(file);
    state.fileKind = getMediaType(file);
    state.sourceImage = null;
    state.sourceVideo = null;

    if (state.fileKind === 'image') {
        const image = new Image();
        image.decoding = 'async';
        image.src = state.sourceUrl;
        await waitForMediaEvent(image, 'load');
        state.sourceImage = image;
        state.mediaWidth = image.naturalWidth;
        state.mediaHeight = image.naturalHeight;
        updateFileMeta(`${file.name} · ${image.naturalWidth}×${image.naturalHeight} image`);
        summarizeWorkload();
        return;
    }

    if (state.fileKind === 'video') {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.muted = true;
        video.playsInline = true;
        video.src = state.sourceUrl;
        await waitForMediaEvent(video, 'loadedmetadata');
        state.sourceVideo = video;
        state.mediaWidth = video.videoWidth;
        state.mediaHeight = video.videoHeight;
        updateFileMeta(
            `${file.name} · ${video.videoWidth}×${video.videoHeight} video · ${video.duration.toFixed(1)}s`
        );
        summarizeWorkload();
        // Eagerly load the video export engine in the background so the user
        // does not have to click the button manually before rendering.
        if (!state.ffmpegReady) {
            loadFFmpeg().catch((err) => {
                console.warn('Auto-load of video export engine failed:', err.message);
            });
        }
        return;
    }

    throw new Error('Unsupported file type. Use an image or video file.');
}

async function drawCurrentSource() {
    const settings = getSettings();

    if (state.fileKind === 'image' && state.sourceImage) {
        drawMediaToCanvas(state.sourceImage, elements.sourceCanvas, settings.scale, settings.customMode);
        return;
    }

    if (state.fileKind === 'video' && state.sourceVideo) {
        const previewTime = Math.min(Math.max(state.sourceVideo.duration * 0.2, 0), Math.max(0, state.sourceVideo.duration - 0.05));
        await seekVideo(state.sourceVideo, previewTime);
        drawMediaToCanvas(state.sourceVideo, elements.sourceCanvas, settings.scale, settings.customMode);
    }
}

async function renderPreview() {
    throwIfCancelled();
    await drawCurrentSource();
    throwIfCancelled();
    await processor.render(elements.sourceCanvas, elements.outputCanvas, getSettings());
    clearRenderedOutput();
    setResultGlow(true);
    console.log(
        state.fileKind === 'video'
            ? 'Preview ready. Use Render final to process the full clip.'
            : 'Preview ready. Use Render final to export the PNG.',
        'success'
    );
}

async function renderImageExport() {
    setProgress(20, 'Rendering image line art...');
    await renderPreview();
    const fileName = `${sanitizeBaseName(state.selectedFile.name)}-lineart.png`;
    const blob = await canvasToBlob(elements.outputCanvas, 'image/png');
    setDownload(blob, fileName);
    setProgress(100, 'PNG export ready.');
    console.log('Image render complete. Download your PNG.', 'success');
}

// ── BMP encoder ──────────────────────────────────────────────────────────────
// Encode a Uint8ClampedArray of RGBA pixels to an uncompressed 24-bit BMP.
// BMP has zero deflate overhead, making it ~100× faster to produce than PNG
// for the intermediate frame files.  FFmpeg reads BMP natively.
// A negative biHeight in the header signals top-down row order so we never
// need to flip rows before writing.
function rgbaToBmp(rgba, width, height) {
    const rowBytes = Math.ceil(width * 3 / 4) * 4; // rows padded to 4 bytes
    const pixelBytes = rowBytes * height;
    const fileSize = 54 + pixelBytes;
    const buf = new Uint8Array(fileSize);
    const v = new DataView(buf.buffer);
    // BITMAPFILEHEADER
    buf[0] = 0x42; buf[1] = 0x4D;         // 'BM'
    v.setUint32(2,  fileSize,      true);  // bfSize
    v.setUint32(6,  0,             true);  // bfReserved
    v.setUint32(10, 54,            true);  // bfOffBits
    // BITMAPINFOHEADER
    v.setUint32(14, 40,            true);  // biSize
    v.setInt32 (18, width,         true);  // biWidth
    v.setInt32 (22, -height,       true);  // biHeight (negative = top-down)
    v.setUint16(26, 1,             true);  // biPlanes
    v.setUint16(28, 24,            true);  // biBitCount
    v.setUint32(30, 0,             true);  // biCompression (BI_RGB)
    v.setUint32(34, pixelBytes,    true);  // biSizeImage
    v.setUint32(38, 2835,          true);  // biXPelsPerMeter (~72 dpi)
    v.setUint32(42, 2835,          true);  // biYPelsPerMeter
    v.setUint32(46, 0,             true);  // biClrUsed
    v.setUint32(50, 0,             true);  // biClrImportant
    // Pixel data: RGBA → BGR, row by row (top-to-bottom matches negative biHeight)
    for (let y = 0; y < height; y++) {
        const rowBase = 54 + y * rowBytes;
        const srcBase = y * width * 4;
        for (let x = 0; x < width; x++) {
            const s = srcBase + x * 4;
            const d = rowBase + x * 3;
            buf[d]     = rgba[s + 2]; // B
            buf[d + 1] = rgba[s + 1]; // G
            buf[d + 2] = rgba[s];     // R
        }
    }
    return buf;
}

// Load and return a fresh FFmpeg instance independent of state.ffmpeg.
// Each instance gets its own WASM heap, allowing true parallel encoding.
// The WASM binary is cached by the browser after the first load, so subsequent
// calls are fast (compile from cache, ~100–300 ms) rather than a full download.
async function createFreshFFmpeg() {
    const FFmpegClass = getFFmpegClass();
    if (typeof FFmpegClass !== 'function') {
        throw new Error('FFmpeg runtime was not found in vendor/ffmpeg.js.');
    }
    const inst = new FFmpegClass();
    const getAssetUrl = (path) => new URL(path, window.location.href).href;
    await inst.load({
        coreURL: getAssetUrl('vendor/ffmpeg-core.js'),
        wasmURL: getAssetUrl('vendor/ffmpeg-core.wasm'),
    });
    return inst;
}

// Determine how many parallel FFmpeg encode workers to use.
// Each worker encodes a separate segment of the video, so N workers gives
// roughly N× speedup on the final encode step.
//   • Require at least MIN_FRAMES_PER_SEGMENT frames per segment to avoid
//     header/concat overhead swamping the actual encode work.
//   • Cap at MAX_ENCODE_WORKERS to stay within browser memory budget
//     (~250 MB per FFmpeg WASM instance).
//   • Respect navigator.deviceMemory: leave at least 1 GB for the rest of
//     the page, budget 250 MB per extra FFmpeg instance.
const MIN_FRAMES_PER_SEGMENT = 60;
const MAX_ENCODE_WORKERS = 4;
// Memory budget per extra FFmpeg WASM instance (MB).
const FFMPEG_WORKER_MB = 250;
// RAM we leave free for the browser runtime, main thread, and the page itself (MB).
const FFMPEG_RESERVED_MB = 1024; // 1 GB
function computeEncodeWorkers(totalFrames) {
    if (totalFrames < MIN_FRAMES_PER_SEGMENT) return 1;
    const mem = (typeof navigator.deviceMemory === 'number') ? navigator.deviceMemory : 4;
    const maxByMem   = Math.max(1, Math.floor((mem * 1024 - FFMPEG_RESERVED_MB) / FFMPEG_WORKER_MB));
    const maxByCores = Math.max(1, (navigator.hardwareConcurrency || 2) - 1);
    const maxByFrames = Math.floor(totalFrames / MIN_FRAMES_PER_SEGMENT);
    return Math.min(MAX_ENCODE_WORKERS, maxByMem, maxByCores, maxByFrames);
}

async function renderVideoExport() {
    const ffmpeg = await loadFFmpeg();
    const settings = getSettings();
    let fps = settings.videoFps;
    const video = state.sourceVideo;
    const jobId = `job-${Date.now()}`;
    const extension = state.selectedFile.name.includes('.')
        ? state.selectedFile.name.slice(state.selectedFile.name.lastIndexOf('.'))
        : '.mp4';
    const safeBaseName = sanitizeBaseName(state.selectedFile.name);
    const inputPath  = `${jobId}/input${extension}`;
    const outputPath = `${jobId}/output.mp4`;
    state.activeJobId = jobId;

    // Declared at function scope so the finally block can always read them.
    let totalFrames      = 1;
    let segCount         = 1;
    let framesPerSegment = 1;
    let decodePool       = null;
    // segInstances[0] is always state.ffmpeg (main); [1..N-1] are fresh workers.
    const segInstances = [];

    try {
        await ffmpeg.createDir(jobId);
        await ffmpeg.writeFile(
            inputPath,
            new Uint8Array(await state.selectedFile.arrayBuffer())
        );

        // ── FPS detection ──────────────────────────────────────────────────────
        if (settings.isOriginalFps) {
            setProgress(2, 'Detecting original framerate...');
            let detectedFps = null;
            const logHandler = ({ message }) => {
                const m = message.match(/(\d+(?:\.\d+)?) fps/);
                if (m) detectedFps = parseFloat(m[1]);
            };
            ffmpeg.on('log', logHandler);
            try { await ffmpeg.exec(['-i', inputPath]); } catch (_) { /* expected */ }
            ffmpeg.off('log', logHandler);
            if (detectedFps && detectedFps > 0) {
                fps = detectedFps;
                console.log(`Detected original FPS: ${fps}`);
            } else {
                console.warn('Could not detect original FPS, falling back to 30');
                fps = 30;
            }
        }

        totalFrames      = Math.max(1, Math.floor(video.duration * fps));
        framesPerSegment = totalFrames; // default (single-segment path)

        // ── Parallel encode worker setup ───────────────────────────────────────
        // Try to load N-1 extra FFmpeg instances for parallel segment encoding.
        // If loading fails (e.g. memory pressure), we fall back to a single instance
        // so the render still completes correctly — just without parallelism.
        const desiredSeg = computeEncodeWorkers(totalFrames);
        if (desiredSeg > 1) {
            try {
                setProgress(4, `Loading ${desiredSeg} parallel encode workers...`);
                console.log(
                    `Parallel encoding: requesting ${desiredSeg} FFmpeg workers...`, 'info'
                );
                const freshInsts = await Promise.all(
                    Array.from({ length: desiredSeg - 1 }, () => createFreshFFmpeg())
                );
                segInstances.push(ffmpeg, ...freshInsts);
                segCount         = desiredSeg;
                framesPerSegment = Math.ceil(totalFrames / segCount);
                console.log(
                    `Parallel encoding ready: ${segCount} workers, ` +
                    `${framesPerSegment} frames/segment.`, 'info'
                );
            } catch (loadErr) {
                console.warn(
                    'Could not load extra FFmpeg workers; using single instance. ' +
                    loadErr.message
                );
                segInstances.length = 0;
            }
        }
        if (segInstances.length === 0) {
            segInstances.push(ffmpeg);
            segCount         = 1;
            framesPerSegment = totalFrames;
        }

        // Create a working directory in each segment instance's virtual FS.
        for (let s = 0; s < segCount; s++) {
            await segInstances[s].createDir(`seg${s}`);
        }

        // ── Max-bitrate guard — output file will never exceed input size ───────
        // Line-art compresses far better than the source video, so in practice
        // the output is always smaller.  This cap is a safety net for edge cases
        // (e.g. already-compressed input, very short clips).
        // Floor at MIN_OUTPUT_BITRATE_KBPS so the encoder never gets an impossibly
        // tight budget (e.g. for very short clips where the arithmetic gives <100).
        const MIN_OUTPUT_BITRATE_KBPS = 200;
        const inputKbps = Math.max(
            MIN_OUTPUT_BITRATE_KBPS,
            Math.ceil((state.selectedFile.size * 8) / (video.duration * 1000))
        );

        // ---- Parallel frame pipeline ------------------------------------------------
        // Each frame is processed concurrently (seek → OpenCV → BMP encode → write)
        // and written to its assigned segment's FFmpeg FS.
        //
        // A pipeline-wide Semaphore caps the number of frames simultaneously alive
        // in the seek→process→BMP→write pipeline.  Without this cap, all frames
        // are queued upfront; each holds ~8 MB of canvas/pixel data, totalling
        // several GB for long/hi-res videos and causing RangeError OOM crashes.
        //
        // Batching (BATCH_SIZE = 2 × concurrency) adds a GC checkpoint between
        // batches so completed frame closures and their pixel buffers are reclaimed.
        //
        // STAGGER: frames 1…N-1 in the first pool-sized batch start with a small
        // per-frame delay so GPU workers receive tasks at offset times, smoothing
        // the sawtooth GPU utilisation pattern.
        // Delay (ms) between consecutive frame-decode starts in the first batch.
        // Large enough to break lock-step GPU submission; small enough to keep
        // the pipeline full with no idle time between workers.
        const FRAME_STAGGER_MS = 30;
        const decodePoolSize   = processor.concurrency;
        decodePool = new VideoDecodePool(decodePoolSize, video);
        state.activeDecodePool = decodePool;

        // +1 slot lets FFmpeg writeFile for frame N overlap with decoding of
        // frame N+concurrency, hiding I/O latency without extra memory cost.
        const inFlightSem = new Semaphore(processor.concurrency + 1);
        const BATCH_SIZE  = Math.max(1, processor.concurrency * 2);

        let completedFrames   = 0;
        let latestShownFrame  = -1;
        let latestOutputFrame = -1;

        for (let batchStart = 0; batchStart < totalFrames; batchStart += BATCH_SIZE) {
            throwIfCancelled();
            await waitIfPaused();
            throwIfCancelled();

            const batchEnd = Math.min(batchStart + BATCH_SIZE, totalFrames);
            const batchPromises = [];

            for (let fi = batchStart; fi < batchEnd; fi++) {
                const capturedFi = fi;
                // Cap at duration-1ms so the last seek never overshoots.
                const frameTime  = Math.min(video.duration - 0.001, capturedFi / fps);
                // Route each frame to the correct segment instance.
                const segIdx     = Math.min(segCount - 1, Math.floor(capturedFi / framesPerSegment));
                const localIdx   = capturedFi - segIdx * framesPerSegment;
                const segFfmpeg  = segInstances[segIdx];

                const p = (async () => {
                    // Stagger the first pool-sized batch to break lock-step seek starts.
                    if (capturedFi > 0 && capturedFi < decodePoolSize) {
                        throwIfCancelled();
                        await new Promise(r => setTimeout(r, capturedFi * FRAME_STAGGER_MS));
                        throwIfCancelled();
                    }

                    // Bound total in-flight frames to prevent OOM on long/hi-res videos.
                    await inFlightSem.acquire();
                    try {
                        // Phase 1: seek + draw to a private off-screen canvas so
                        // concurrent frames never overwrite each other's pixels.
                        let capturedW, capturedH, offCanvas;
                        {
                            const { video: vEl, release: releaseVideo } =
                                await decodePool.acquire();
                            try {
                                throwIfCancelled();
                                offCanvas = document.createElement('canvas');
                                await seekVideo(vEl, frameTime);
                                ({ width: capturedW, height: capturedH } =
                                    drawMediaToCanvas(
                                        vEl, offCanvas,
                                        settings.scale, settings.customMode
                                    ));
                                if (capturedFi > latestShownFrame) {
                                    latestShownFrame = capturedFi;
                                    drawMediaToCanvas(
                                        vEl, elements.sourceCanvas,
                                        settings.scale, settings.customMode
                                    );
                                }
                            } finally {
                                releaseVideo();
                            }
                        }

                        // Phase 2: OpenCV/GPU processing → BMP encode → write to
                        // the assigned segment's FFmpeg WASM filesystem.
                        throwIfCancelled();
                        let rawData = await processor.renderToData(offCanvas, settings);
                        offCanvas = null; // release canvas backing store immediately

                        throwIfCancelled();

                        // Live output preview (forward-progress only).
                        if (capturedFi > latestOutputFrame) {
                            latestOutputFrame = capturedFi;
                            elements.outputCanvas.width  = capturedW;
                            elements.outputCanvas.height = capturedH;
                            elements.outputCanvas.getContext('2d').putImageData(
                                new ImageData(rawData, capturedW, capturedH), 0, 0
                            );
                        }

                        // BMP has zero deflate overhead (~100× faster than PNG).
                        // The raw bytes are larger but the Semaphore ensures only
                        // O(concurrency) frames reside in the WASM heap at once.
                        let frameBytes = rgbaToBmp(rawData, capturedW, capturedH);
                        rawData = null; // GC: preview done, BMP encoded

                        throwIfCancelled();
                        await segFfmpeg.writeFile(
                            `seg${segIdx}/frame-${String(localIdx).padStart(5, '0')}.bmp`,
                            frameBytes
                        );
                        frameBytes = null; // GC: written to WASM FS

                        completedFrames++;
                        setProgress(
                            5 + (completedFrames / totalFrames) * 85,
                            `Processed frame ${completedFrames} of ${totalFrames}...`
                        );
                    } finally {
                        inFlightSem.release();
                    }
                })();

                batchPromises.push(p);
            }

            // Await the full batch before scheduling the next one.
            // This gives the JS garbage collector a definite checkpoint to reclaim
            // completed frame closures and their large pixel buffers — essential
            // for long, high-resolution videos that would otherwise accumulate GBs.
            const batchResults = await Promise.allSettled(batchPromises);
            const firstFailure = batchResults.find(r => r.status === 'rejected');
            if (firstFailure) throw firstFailure.reason;
        }

        decodePool.destroy();
        decodePool = null;
        state.activeDecodePool = null;
        // ---- End parallel frame pipeline -------------------------------------------

        throwIfCancelled();

        // ── Parallel encode ────────────────────────────────────────────────────
        // All N segment instances encode their frames simultaneously.
        // Thread budget: floor(total-cores / segCount) per segment keeps all
        // segments encoding in parallel without excessive thread contention.
        const threadsPerSeg = Math.max(
            1, Math.floor((navigator.hardwareConcurrency || 2) / segCount)
        );

        setProgress(
            90,
            segCount > 1
                ? `Encoding ${segCount} video segments in parallel...`
                : 'Encoding video...'
        );
        console.log(
            segCount > 1
                ? `Parallel encode: ${segCount} segments × ${threadsPerSeg} threads`
                : `Encoding video (${threadsPerSeg} threads)...`,
            'info'
        );

        // Map state.ffmpeg progress events (fired for segment 0) to 90–96 %.
        state.encodePhaseOffset = 90;
        state.encodePhaseScale  = 0.06;

        // encodeOneSegment: encode all BMP frames in seg${s}/ to seg${s}/out.mp4
        // (video-only), read the result, then immediately delete frame files from
        // the WASM heap to free memory before the concat step.
        const encodeOneSegment = async (segFfmpeg, s) => {
            const segFrameCount = Math.min(
                framesPerSegment, totalFrames - s * framesPerSegment
            );
            const segArgs = [
                '-y',
                '-framerate', String(fps),
                '-start_number', '0',
                '-i', `seg${s}/frame-%05d.bmp`,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'animation',
                // CRF 28: perceptually lossless for binary line art, ~30 % faster
                // than the default CRF 23 with no visible quality loss.
                '-crf', '28',
                // GOP = 1 second: cheap I-frames for 2-colour content; enables
                // clean seeking.
                '-g', String(Math.max(1, Math.round(fps))),
                '-threads', String(threadsPerSeg),
                '-pix_fmt', 'yuv420p',
                // Ensure even dimensions required by yuv420p.
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                // Hard bitrate cap so the output segment never exceeds the input
                // bitrate — guarantees the final file is no larger than the source.
                '-maxrate', `${inputKbps}k`,
                '-bufsize', `${inputKbps * 2}k`,
                `seg${s}/out.mp4`,
            ];
            const segLog  = [];
            const segLogH = ({ message }) => segLog.push(message);
            segFfmpeg.on('log', segLogH);
            let exitCode;
            try {
                exitCode = await segFfmpeg.exec(segArgs);
            } finally {
                segFfmpeg.off('log', segLogH);
            }
            if (exitCode !== 0) {
                throw new Error(
                    `Segment ${s} encode failed (exit ${exitCode}).\n` +
                    segLog.slice(-20).join('\n')
                );
            }

            // Read the encoded segment before freeing WASM heap.
            const mp4Data = await segFfmpeg.readFile(`seg${s}/out.mp4`);

            // Delete frame BMPs and the encoded segment from this instance's WASM
            // heap immediately — critical for long videos where frames linger until
            // the outer finally block and exhaust the WASM address space.
            const frameDels = Array.from({ length: segFrameCount }, (_, i) =>
                safeDeleteFile(
                    segFfmpeg,
                    `seg${s}/frame-${String(i).padStart(5, '0')}.bmp`
                )
            );
            await Promise.all(frameDels);
            await safeDeleteFile(segFfmpeg, `seg${s}/out.mp4`);
            try { await segFfmpeg.deleteDir(`seg${s}`); } catch (_) {}

            return mp4Data;
        };

        const segMp4Bufs = await Promise.all(
            segInstances.map((inst, s) => encodeOneSegment(inst, s))
        );

        throwIfCancelled();

        // ── Final assembly: merge all segments + mux original audio ───────────
        setProgress(96, 'Assembling final video with audio...');
        console.log('Assembling final video...', 'info');

        // Map progress events to the 96–100 % band during the final mux.
        state.encodePhaseOffset = 96;
        state.encodePhaseScale  = 0.04;

        // Write segment MP4s to the main instance for concat.
        for (let s = 0; s < segCount; s++) {
            await ffmpeg.writeFile(`${jobId}/seg${s}.mp4`, segMp4Bufs[s]);
            segMp4Bufs[s] = null; // GC: safely written to main FS
        }

        let finalArgs;
        if (segCount === 1) {
            // Single segment: mux audio onto the already-encoded video.
            // -c:v copy avoids a re-encode; -c:a aac re-encodes audio to AAC.
            finalArgs = [
                '-y',
                '-i', `${jobId}/seg0.mp4`,
                '-i', inputPath,
                '-map', '0:v:0',
                '-map', '1:a?',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-movflags', '+faststart',
                outputPath,
            ];
        } else {
            // Multiple segments: write a concat list, then stream-copy + mux audio.
            // -safe 0 allows paths containing the jobId prefix.
            const concatContent = Array.from(
                { length: segCount }, (_, s) => `file '${jobId}/seg${s}.mp4'`
            ).join('\n');
            await ffmpeg.writeFile(
                `${jobId}/concat.txt`,
                new TextEncoder().encode(concatContent)
            );
            finalArgs = [
                '-y',
                '-f', 'concat', '-safe', '0',
                '-i', `${jobId}/concat.txt`,
                '-i', inputPath,
                '-map', '0:v:0',
                '-map', '1:a?',
                '-c:v', 'copy',   // all segments share codec/params — stream-copy is safe
                '-c:a', 'aac',
                '-shortest',
                '-movflags', '+faststart',
                outputPath,
            ];
        }

        const finalLog  = [];
        const finalLogH = ({ message }) => finalLog.push(message);
        ffmpeg.on('log', finalLogH);
        let finalCode;
        try {
            finalCode = await ffmpeg.exec(finalArgs);
        } finally {
            ffmpeg.off('log', finalLogH);
        }
        if (finalCode !== 0) {
            throw new Error(
                `Final video assembly failed (exit ${finalCode}).\n` +
                finalLog.slice(-30).join('\n')
            );
        }

        throwIfCancelled();
        const outputData = await ffmpeg.readFile(outputPath);
        if (!outputData || outputData.byteLength === 0) {
            throw new Error(
                'FFmpeg produced an empty output file. The video encoding may have failed.'
            );
        }
        const videoBlob = new Blob([outputData], { type: 'video/mp4' });
        setDownload(videoBlob, `${safeBaseName}-lineart.mp4`);
        // Separate blob URL for the video preview element so Chrome's range-request
        // errors against the media element don't taint the download link URL.
        revokeUrl('videoPreviewUrl');
        state.videoPreviewUrl = URL.createObjectURL(videoBlob);
        elements.outputVideo.src    = state.videoPreviewUrl;
        elements.videoResult.hidden = false;
        setProgress(100, 'MP4 export ready.');
        console.log('Video render complete. Download or review the MP4.', 'success');

    } finally {
        // Always destroy the decode pool even if the render was cancelled or threw.
        if (decodePool) {
            try { decodePool.destroy(); } catch (_) {}
        }
        state.activeDecodePool = null;

        // Best-effort cleanup of any BMP frames / segment outputs that may still
        // reside in a segment instance's WASM heap (handles cancel + mid-render
        // errors).  encodeOneSegment already deletes on success, so these
        // safeDeleteFile calls are harmless no-ops for already-deleted files.
        await Promise.allSettled(segInstances.map(async (segFfmpeg, s) => {
            const n = Math.max(1, framesPerSegment);
            const frameDels = Array.from({ length: n }, (_, i) =>
                safeDeleteFile(
                    segFfmpeg,
                    `seg${s}/frame-${String(i).padStart(5, '0')}.bmp`
                )
            );
            await Promise.allSettled(frameDels);
            await safeDeleteFile(segFfmpeg, `seg${s}/out.mp4`);
            try { await segFfmpeg.deleteDir(`seg${s}`); } catch (_) {}
        }));

        // Clean up the main instance's job directory (input, segment MP4s, concat list, output).
        await cleanupFfmpegJob(ffmpeg, jobId, segCount, inputPath, outputPath);

        // Terminate extra segment workers (index 0 is state.ffmpeg — never terminate).
        for (let s = 1; s < segInstances.length; s++) {
            try {
                if (typeof segInstances[s].terminate === 'function') {
                    segInstances[s].terminate();
                }
            } catch (_) {}
        }

        // Reset progress-phase state for the next render.
        state.encodePhaseOffset = 90;
        state.encodePhaseScale  = 0.1;
        state.activeJobId = '';
    }
}

async function handleFileSelection(file) {
    if (!file) {
        return;
    }

    if (!state.cvReady) {
        console.log('Processing engine still loading — please wait a moment and try again.', 'warn');
        return;
    }

    try {
        setBusy(true);
        state.cancelRequested = false;
        resetProgress();
        clearRenderedOutput();
        state.selectedFile = file;
        await readSelectedFile(file);
        await renderPreview();
    } catch (error) {
        console.error(error);
        state.selectedFile = null;
        state.fileKind = null;
        updateFileMeta('No file selected.');
        drawEmptyCanvas(elements.sourceCanvas, 'Source preview');
        drawEmptyCanvas(elements.outputCanvas, 'Line-art preview');
        setAdvisory('Select a file to estimate browser workload.', 'info');
        console.log(error.message, 'error');
    } finally {
        setBusy(false);
        updateUnloadProtection();
    }
}

function resetWorkspace() {
    processor.reset();
    state.selectedFile = null;
    state.fileKind = null;
    state.sourceImage = null;
    state.sourceVideo = null;
    state.mediaWidth = null;
    state.mediaHeight = null;
    revokeUrl('sourceUrl');
    clearRenderedOutput();
    setResultGlow(false);
    resetProgress();
    elements.fileInput.value = '';
    updateFileMeta('No file selected.');
    setAdvisory('Select a file to estimate browser workload.', 'info');
    drawEmptyCanvas(elements.sourceCanvas, 'Source preview');
    drawEmptyCanvas(elements.outputCanvas, 'Line-art preview');
    if (!state.cvReady) {
        setAdvisory('Loading processing engine… This may take 10–30 seconds on first load.', 'info');
    } else {
        setAdvisory('Ready. Drop a photo or video clip to get started.', 'success');
    }
    refreshActions();
    updateUnloadProtection();
}

// Update the hint text below the worker threads slider.
// currentWorkers — the value the slider is currently set to.
function updateWorkerThreadsHint(currentWorkers) {
    const { auto, cores, memoryGB } = workerConfig;
    const memStr = memoryGB !== null ? `${memoryGB} GB RAM` : 'RAM unknown';
    let baseInfo = `Auto: ${auto} (${cores} cores, ${memStr})`;

    // Append a resolution-aware recommendation when media dimensions are known.
    if (state.mediaWidth && state.mediaHeight) {
        const resInfo = computeWorkersForResolution(state.mediaWidth, state.mediaHeight);
        baseInfo += ` · ${state.mediaWidth}×${state.mediaHeight}: ≤${resInfo.max} workers (~${resInfo.perWorkerMB} MB each)`;
    }

    if (currentWorkers > auto) {
        const extraNote = currentWorkers > Number(elements.workerThreadsInput.max)
            ? ' — ⚠ manually above detected core count, monitor memory carefully'
            : ' — ⚠ above recommended, may cause memory pressure';
        elements.workerThreadsHint.textContent = `${baseInfo}${extraNote}`;
        elements.workerThreadsHint.dataset.tone = 'warn';
    } else {
        elements.workerThreadsHint.textContent = baseInfo;
        delete elements.workerThreadsHint.dataset.tone;
    }
}

// Tear down the current processor pool and replace it with a new one of
// the requested size.  The UI is temporarily locked while workers reload.
function rebuildProcessor(n) {
    processor.terminate();
    state.cvReady = false;
    processor = new LineArtProcessor(n);
    refreshActions();
    updateWorkerThreadsHint(n);
}

// Absolute upper bound for the manual thread count entry.
// Users can type up to this value; there is no slider stop here — they
// deliberately type a number they know their device can handle.
const WORKER_THREADS_ABSOLUTE_MAX = 64;

// Initialise the worker-threads slider with the auto-detected safe values.
function initWorkerThreadsControl() {
    const { auto, max } = workerConfig;
    // Slider ceiling: core-based max (always reserves 2 cores for UI/FFmpeg).
    // Users who want more can increase beyond the memory-safe auto value and
    // will see an ⚠ warning; the slider still won't allow starving the UI.
    elements.workerThreadsInput.max = max;
    elements.workerThreadsMax.textContent = max;
    elements.workerThreadsInput.value = auto;
    elements.workerThreadsVal.textContent = auto;
    elements.workerThreadsManual.max = WORKER_THREADS_ABSOLUTE_MAX;
    elements.workerThreadsManual.min = 1;
    elements.workerThreadsManual.value = auto;
    updateWorkerThreadsHint(auto);
}

async function onPreviewClick() {
    if (!state.selectedFile || state.processing) {
        return;
    }

    try {
        setBusy(true);
        state.cancelRequested = false;
        updateUnloadProtection();
        await renderPreview();
    } catch (error) {
        console.error(error);
        console.log(error.message, error.message === 'Render cancelled.' ? 'warn' : 'error');
    } finally {
        setBusy(false);
        updateUnloadProtection();
    }
}

function onPauseClick() {
    if (!state.processing) {
        return;
    }

    state.pauseRequested = !state.pauseRequested;
    if (state.pauseRequested) {
        elements.pauseBtn.textContent = 'Resume';
        setProgress(
            getCurrentProgress(),
            'Paused — click Resume to continue...'
        );
        console.log('Render paused. Click Resume to continue.', 'warn');
    } else {
        elements.pauseBtn.textContent = 'Pause';
        setProgress(
            getCurrentProgress(),
            'Resuming render...'
        );
        console.log('Render resumed.', 'info');
    }
}

async function onRenderClick() {
    if (!state.selectedFile || state.processing) {
        return;
    }

    try {
        setBusy(true);
        state.cancelRequested = false;
        updateUnloadProtection();
        resetProgress();

        if (state.fileKind === 'image') {
            await renderImageExport();
            return;
        }

        if (state.fileKind === 'video') {
            setBusy(true, true); // show pause button for video renders
            await renderVideoExport();
            return;
        }

        throw new Error('Unsupported file type.');
    } catch (error) {
        console.error(error);
        console.log(error.message || 'Rendering failed.', error.message === 'Render cancelled.' ? 'warn' : 'error');
    } finally {
        setBusy(false);
        state.cancelRequested = false;
        updateUnloadProtection();
    }
}

function attachDropZone() {
    ['dragenter', 'dragover'].forEach((eventName) => {
        elements.dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            elements.dropZone.classList.add('is-dragover');
        });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        elements.dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            elements.dropZone.classList.remove('is-dragover');
        });
    });

    elements.dropZone.addEventListener('drop', async (event) => {
        const [file] = event.dataTransfer.files;
        await handleFileSelection(file);
    });
}

// OpenCV is loaded inside worker.js; cv-ready is signalled via the worker message handler above.

elements.fileInput.addEventListener('change', async (event) => {
    const [file] = event.target.files;
    await handleFileSelection(file);
});

elements.previewBtn.addEventListener('click', onPreviewClick);
elements.renderBtn.addEventListener('click', onRenderClick);
elements.cancelBtn.addEventListener('click', requestCancel);
elements.pauseBtn.addEventListener('click', onPauseClick);
elements.resetBtn.addEventListener('click', resetWorkspace);

['preset', 'detail', 'lineWeight', 'scale', 'videoFps', 'customVideoFps'].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', async () => {
        if (id === 'preset') {
            document.getElementById('customControls').hidden = elements.preset.value !== 'custom';
        }
        if (id === 'videoFps') {
            elements.customVideoFps.hidden = elements.videoFps.value !== 'custom';
        }
        summarizeWorkload();

        if (!state.selectedFile || state.processing) {
            return;
        }

        await onPreviewClick();
    });
});

// Custom preset controls: update live labels and re-preview on change
const customInputIds = ['customBg', 'customInk', 'customLowThresh', 'customHighThresh', 'customBilateral', 'customSigma', 'customUseBilateral', 'customBilateralPasses', 'customUseGaussian', 'customGaussianPasses', 'customUseMedian', 'customMedianPasses', 'customCleanSpeckles', 'customCleanSpecklesIntensity', 'customAutoNormalize', 'customDarkBoost', 'customDarkBoostClip', 'customMergeDoubleEdge', 'customMergeDoubleEdgeIntensity'];
const customValueSpans = {
    customLowThresh: document.getElementById('customLowThreshVal'),
    customHighThresh: document.getElementById('customHighThreshVal'),
    customBilateral: document.getElementById('customBilateralVal'),
    customSigma: document.getElementById('customSigmaVal'),
    customBilateralPasses: document.getElementById('customBilateralPassesVal'),
    customGaussianPasses: document.getElementById('customGaussianPassesVal'),
    customMedianPasses: document.getElementById('customMedianPassesVal'),
    customCleanSpecklesIntensity: document.getElementById('customCleanSpecklesIntensityVal'),
    customDarkBoostClip: document.getElementById('customDarkBoostClipVal'),
    customMergeDoubleEdgeIntensity: document.getElementById('customMergeDoubleEdgeIntensityVal')
};

// Grey out bilateral diameter/sigma/passes sliders when bilateral smooth is disabled.
function syncBilateralRows() {
    const enabled = document.getElementById('customUseBilateral').checked;
    ['customBilateral', 'customSigma'].forEach((id) => {
        const row = document.getElementById(id).closest('.custom-slider-row');
        if (row) {
            row.classList.toggle('is-disabled', !enabled);
            document.getElementById(id).disabled = !enabled;
        }
    });
    document.getElementById('bilateralPassesRow').classList.toggle('is-disabled', !enabled);
    document.getElementById('customBilateralPasses').disabled = !enabled;
}

// Show/hide Gaussian passes slider when Gaussian smooth is toggled.
function syncGaussianPassesRow() {
    const enabled = document.getElementById('customUseGaussian').checked;
    document.getElementById('gaussianPassesRow').hidden = !enabled;
}

// Show/hide Median passes slider when Median smooth is toggled.
function syncMedianPassesRow() {
    const enabled = document.getElementById('customUseMedian').checked;
    document.getElementById('medianPassesRow').hidden = !enabled;
}

// Show/hide clean speckles intensity slider when clean speckles is toggled.
function syncCleanSpecklesRow() {
    const enabled = document.getElementById('customCleanSpeckles').checked;
    document.getElementById('cleanSpecklesIntensityRow').hidden = !enabled;
}

// Show the intensity slider only when merge double-edges is enabled.
function syncMergeDoubleEdgeRow() {
    const enabled = document.getElementById('customMergeDoubleEdge').checked;
    document.getElementById('mergeDoubleEdgeIntensityRow').hidden = !enabled;
}

// Show/hide the CLAHE clip-limit slider when the extra shadow boost is toggled.
function syncDarkBoostRow() {
    const enabled = document.getElementById('customDarkBoost').checked;
    document.getElementById('darkBoostClipRow').hidden = !enabled;
}

customInputIds.forEach((id) => {
    document.getElementById(id).addEventListener('input', () => {
        if (customValueSpans[id]) {
            customValueSpans[id].textContent = document.getElementById(id).value;
        }
        // Keep hex text inputs in sync when the color picker wheel moves
        if (id === 'customBg') {
            document.getElementById('customBgHex').value = document.getElementById('customBg').value;
        } else if (id === 'customInk') {
            document.getElementById('customInkHex').value = document.getElementById('customInk').value;
        }
    });
    document.getElementById(id).addEventListener('change', async () => {
        // Keep dependent UI states in sync for controls that have visual side-effects.
        if (id === 'customUseBilateral') { syncBilateralRows(); }
        if (id === 'customUseGaussian') { syncGaussianPassesRow(); }
        if (id === 'customUseMedian') { syncMedianPassesRow(); }
        if (id === 'customCleanSpeckles') { syncCleanSpecklesRow(); }
        if (id === 'customDarkBoost') { syncDarkBoostRow(); }
        if (id === 'customMergeDoubleEdge') { syncMergeDoubleEdgeRow(); }
        if (elements.preset.value !== 'custom' || !state.selectedFile || state.processing) {
            return;
        }
        await onPreviewClick();
    });
});

// Parse a user-supplied colour string (hex or rgb) into a valid #rrggbb hex value.
// Returns null when the input cannot be interpreted.
function parseColorInput(raw) {
    const s = raw.trim();
    // Already a valid 6- or 3-digit hex
    if (/^#?[0-9a-fA-F]{6}$/.test(s)) {
        return s.startsWith('#') ? s : `#${s}`;
    }
    if (/^#?[0-9a-fA-F]{3}$/.test(s)) {
        const hex = s.replace('#', '');
        return `#${hex[0]}${hex[0]}${hex[1]}${hex[1]}${hex[2]}${hex[2]}`;
    }
    // rgb(r, g, b) or rgb(r g b)
    const rgbMatch = s.match(/^rgb\s*\(\s*(\d{1,3})\s*(?:,\s*|\s+)(\d{1,3})\s*(?:,\s*|\s+)(\d{1,3})\s*\)$/i);
    if (rgbMatch) {
        const toHex = (n) => Math.min(255, Math.max(0, Number(n))).toString(16).padStart(2, '0');
        return `#${toHex(rgbMatch[1])}${toHex(rgbMatch[2])}${toHex(rgbMatch[3])}`;
    }
    return null;
}

function attachHexInput(hexInputId, colorPickerId) {
    const hexEl = document.getElementById(hexInputId);
    const colorEl = document.getElementById(colorPickerId);

    function applyHex() {
        const parsed = parseColorInput(hexEl.value);
        if (!parsed) {
            hexEl.value = colorEl.value;
            return;
        }
        hexEl.value = parsed;
        colorEl.value = parsed;
        if (elements.preset.value === 'custom' && state.selectedFile && !state.processing) {
            onPreviewClick();
        }
    }

    hexEl.addEventListener('blur', applyHex);
    hexEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            applyHex();
        }
    });
}

attachHexInput('customBgHex', 'customBg');
attachHexInput('customInkHex', 'customInk');

// Worker threads slider — update label and hint in real time (input); rebuild
// the pool only when the user commits a value (change) to avoid spawning many
// short-lived pools while dragging.
elements.workerThreadsInput.addEventListener('input', () => {
    const n = Number(elements.workerThreadsInput.value);
    elements.workerThreadsVal.textContent = n;
    elements.workerThreadsManual.value = n;
    updateWorkerThreadsHint(n);
});

elements.workerThreadsInput.addEventListener('change', () => {
    const n = Number(elements.workerThreadsInput.value);
    if (state.processing) {
        processor.resize(n);
        if (state.activeDecodePool) state.activeDecodePool.resize(n);
    } else {
        rebuildProcessor(n);
    }
});

// Manual number input — lets users type any value up to WORKER_THREADS_ABSOLUTE_MAX.
// The slider is updated to reflect the new value (clamped to its own max).
elements.workerThreadsManual.addEventListener('input', () => {
    const raw = Number(elements.workerThreadsManual.value);
    if (!Number.isFinite(raw) || raw < 1) return;
    const n = Math.min(WORKER_THREADS_ABSOLUTE_MAX, Math.max(1, Math.round(raw)));
    elements.workerThreadsVal.textContent = n;
    // Clamp the slider to its own max — the slider cannot represent values
    // above its max, but the manual input can go higher.
    elements.workerThreadsInput.value = Math.min(n, Number(elements.workerThreadsInput.max));
    updateWorkerThreadsHint(n);
});

elements.workerThreadsManual.addEventListener('change', () => {
    const raw = Number(elements.workerThreadsManual.value);
    if (!Number.isFinite(raw) || raw < 1) {
        // Restore to the current processor count on invalid input.
        elements.workerThreadsManual.value = processor.concurrency;
        return;
    }
    const n = Math.min(WORKER_THREADS_ABSOLUTE_MAX, Math.max(1, Math.round(raw)));
    elements.workerThreadsManual.value = n;
    elements.workerThreadsInput.value = Math.min(n, Number(elements.workerThreadsInput.max));
    elements.workerThreadsVal.textContent = n;
    if (state.processing) {
        processor.resize(n);
        if (state.activeDecodePool) state.activeDecodePool.resize(n);
    } else {
        rebuildProcessor(n);
    }
});

attachDropZone();
initWorkerThreadsControl();
// Pre-create the source canvas context with willReadFrequently so that
// repeated getImageData calls in renderToData do not trigger a browser warning.
elements.sourceCanvas.getContext('2d', { willReadFrequently: true });
resetWorkspace();
