const elements = {
    fileInput: document.getElementById('fileInput'),
    cancelBtn: document.getElementById('cancelBtn'),
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
    dropZone: document.getElementById('dropZone')
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
    processing: false,
    cancelRequested: false,
    activeJobId: '',
    beforeUnloadAttached: false
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

// LineArtProcessor delegates all heavy OpenCV work to a dedicated Web Worker so
// the main thread (and the UI) stays responsive during processing.  Pixel data
// is transferred between threads as zero-copy ArrayBuffer Transferables.
class LineArtProcessor {
    constructor() {
        this._worker = new Worker('worker.js');
        this._pending = new Map();
        this._idCounter = 0;
        this._initTimeout = null;
        this._loadingStartTime = Date.now();

        console.log('[Main] LineArtProcessor: Worker created, waiting for OpenCV to initialize...');

        // Set a timeout for OpenCV initialization (30 seconds)
        this._initTimeout = setTimeout(() => {
            if (!state.cvReady) {
                const elapsedSeconds = Math.round((Date.now() - this._loadingStartTime) / 1000);
                console.error(`[Main] OpenCV initialization timeout after ${elapsedSeconds} seconds - worker did not signal ready`);
                setAdvisory('Processing engine failed to load. Please check your internet connection and reload the page.', 'error');
                elements.dropZone.classList.remove('is-loading');
            }
        }, 30000);

        this._worker.onmessage = (event) => {
            const msg = event && event.data;
            if (!msg) return;
            if (msg.type === 'cv-ready') {
                const elapsedSeconds = Math.round((Date.now() - this._loadingStartTime) / 1000);
                console.log(`[Main] Received cv-ready message from worker after ${elapsedSeconds} seconds`);
                if (this._initTimeout) {
                    clearTimeout(this._initTimeout);
                    this._initTimeout = null;
                }
                state.cvReady = true;
                refreshActions();
                return;
            }

            if (msg.type === 'cv-error') {
                console.error('[Main] Received cv-error from worker:', msg.message);
                if (this._initTimeout) {
                    clearTimeout(this._initTimeout);
                    this._initTimeout = null;
                }
                setAdvisory('Failed to load processing engine: ' + msg.message, 'error');
                elements.dropZone.classList.remove('is-loading');
                return;
            }

            if (msg.id !== undefined) {
                const entry = this._pending.get(msg.id);
                this._pending.delete(msg.id);
                if (!entry) {
                    return;
                }

                if (msg.type === 'result') {
                    const { resolve, destinationCanvas, width, height } = entry;
                    destinationCanvas.width = width;
                    destinationCanvas.height = height;
                    const ctx = destinationCanvas.getContext('2d');
                    const clampedArray = new Uint8ClampedArray(msg.data);
                    ctx.putImageData(new ImageData(clampedArray, width, height), 0, 0);
                    resolve();
                } else if (msg.type === 'error') {
                    entry.reject(new Error(msg.message));
                }
            }
        };

        this._worker.onerror = (error) => {
            console.error('[Main] OpenCV worker error:', error);
            if (this._initTimeout) {
                clearTimeout(this._initTimeout);
                this._initTimeout = null;
            }
            for (const [, { reject }] of this._pending) {
                reject(new Error('Processing worker error.'));
            }
            this._pending.clear();
            setAdvisory('Processing worker error. Please reload the page.', 'error');
            elements.dropZone.classList.remove('is-loading');
        };
    }

    reset() {
        for (const [, { reject }] of this._pending) {
            reject(new Error('Render cancelled.'));
        }
        this._pending.clear();
        this._worker.postMessage({ type: 'reset' });
    }

    // Returns a Promise that resolves once the output canvas has been updated.
    render(sourceCanvas, destinationCanvas, settings) {
        const width = sourceCanvas.width;
        const height = sourceCanvas.height;
        const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
        const imageData = sourceContext.getImageData(0, 0, width, height);
        const id = this._idCounter++;

        return new Promise((resolve, reject) => {
            this._pending.set(id, { resolve, reject, destinationCanvas, width, height });
            // Transfer the pixel buffer (zero-copy) to the worker.
            this._worker.postMessage(
                { type: 'process', id, rgbaData: imageData.data, width, height, settings },
                [imageData.data.buffer]
            );
        });
    }
}

const processor = new LineArtProcessor();

function setBusy(isBusy) {
    state.processing = isBusy;
    elements.cancelBtn.hidden = !isBusy;
    elements.cancelBtn.disabled = !isBusy;
    refreshActions();
}

function refreshActions() {
    const hasFile = Boolean(state.selectedFile);
    const notReady = !state.cvReady || state.processing;
    elements.previewBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.renderBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.fileInput.disabled = notReady;
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
        darkBoost: presetKey === 'custom' && document.getElementById('customDarkBoost').checked,
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
    console.log('Cancel requested. Finishing the current step...', 'warn');
    setProgress(Number(elements.progressPercent.textContent.replace('%', '')) || 0, 'Stopping current job...');

    processor.reset();

    if (state.ffmpeg) {
        state.ffmpeg.terminate();
        state.ffmpeg = null;
        state.ffmpegReady = false;
    }
}

function throwIfCancelled() {
    if (state.cancelRequested) {
        throw new Error('Render cancelled.');
    }
}

async function safeDeleteFile(ffmpeg, filePath) {
    try {
        await ffmpeg.deleteFile(filePath);
    } catch (error) {
        return;
    }
}

async function cleanupFfmpegJob(ffmpeg, jobId, totalFrames, inputPath, outputPath) {
    if (!ffmpeg || !jobId) {
        return;
    }

    for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        await safeDeleteFile(ffmpeg, `${jobId}/frame-${String(frameIndex).padStart(5, '0')}.png`);
    }

    await safeDeleteFile(ffmpeg, inputPath);
    await safeDeleteFile(ffmpeg, outputPath);

    try {
        await ffmpeg.deleteDir(jobId);
    } catch (error) {
        return;
    }
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
                const percent = Math.min(100, Math.max(0, Math.round(progress * 100)));
                console.log(`Encoding video... ${percent}%`, 'info');
                setProgress(92 + Math.round(percent * 0.08), 'Encoding final MP4...');
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
    const inputPath = `${jobId}/input${extension}`;
    const framePattern = `${jobId}/frame-%05d.png`;
    const outputPath = `${jobId}/output.mp4`;
    state.activeJobId = jobId;

    let totalFrames = 1;

    try {
        await ffmpeg.createDir(jobId);
        await ffmpeg.writeFile(inputPath, new Uint8Array(await state.selectedFile.arrayBuffer()));

        if (settings.isOriginalFps) {
            setProgress(5, 'Detecting original framerate...');
            let detectedFps = null;
            const logHandler = ({ message }) => {
                const fpsMatch = message.match(/(\d+(?:\.\d+)?) fps/);
                if (fpsMatch) {
                    detectedFps = parseFloat(fpsMatch[1]);
                }
            };
            ffmpeg.on('log', logHandler);
            try {
                await ffmpeg.exec(['-i', inputPath]);
            } catch (e) {
                // ffmpeg exits with code 1 if no output file is provided, which is expected here
            }
            ffmpeg.off('log', logHandler);

            if (detectedFps && detectedFps > 0) {
                fps = detectedFps;
                console.log(`Detected original FPS: ${fps}`);
            } else {
                console.warn('Could not detect original FPS, falling back to 30');
                fps = 30;
            }
        }
        totalFrames = Math.max(1, Math.floor(video.duration * fps));

        for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
            throwIfCancelled();
            const frameTime = Math.min(video.duration, frameIndex / fps);
            await seekVideo(video, frameTime);
            drawMediaToCanvas(video, elements.sourceCanvas, settings.scale, settings.customMode);
            await processor.render(elements.sourceCanvas, elements.outputCanvas, settings);

            const frameBlob = await canvasToBlob(elements.outputCanvas, 'image/png');
            const frameBytes = new Uint8Array(await frameBlob.arrayBuffer());
            const frameName = `${jobId}/frame-${String(frameIndex).padStart(5, '0')}.png`;
            await ffmpeg.writeFile(frameName, frameBytes);
            const framePercent = ((frameIndex + 1) / totalFrames) * 92;
            setProgress(framePercent, `Rendering frame ${frameIndex + 1} of ${totalFrames}...`);
            console.log(`Rendering frame ${frameIndex + 1} of ${totalFrames}...`, 'info');
        }

        throwIfCancelled();
        setProgress(92, 'Encoding final MP4 in the browser...');
        console.log('Encoding final MP4 in the browser...', 'info');
        const encodeArgs = [
            '-y',
            '-framerate', String(fps),
            '-i', framePattern,
            '-i', inputPath,
            '-map', '0:v:0',
            '-map', '1:a?',
            '-c:v', 'libx264',
            // ultrafast preset encodes 4–5× faster than the default medium preset with
            // no visible quality difference for pure black-and-white line art.
            // animation tune tells x264 the content has large flat areas (which is true
            // for line art), improving both compression ratio and encode speed.
            '-preset', 'ultrafast',
            '-tune', 'animation',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-shortest'
        ];

        if (settings.isOriginalFps) {
            // If using original FPS, we can just let ffmpeg use the source frame rate for the video
            // by not specifying -framerate for the input or mapping it differently. But since we extract frames based on time,
            // we have to reconstruct. However, FFmpeg can just read the original file and apply an OpenCV filter natively,
            // but since we render in browser canvas, we must specify framerate. We fallback to 30.
            // A truly native way to match FPS is to use the original video stream's framerate.
            // `-framerate` applies to the image sequence. We'll use our computed fps proxy.
            // We can also copy the timebase and fps from the original if we processed it frame by frame using ffmpeg native extract,
            // but for canvas readback, the above is sufficient.
        }
        encodeArgs.push(outputPath);

        await ffmpeg.exec(encodeArgs);

        throwIfCancelled();
        const outputData = await ffmpeg.readFile(outputPath);
        const videoBlob = new Blob([outputData.buffer], { type: 'video/mp4' });
        setDownload(videoBlob, `${safeBaseName}-lineart.mp4`);
        elements.outputVideo.src = state.outputUrl;
        elements.videoResult.hidden = false;
        setProgress(100, 'MP4 export ready.');
        console.log('Video render complete. Download or review the MP4.', 'success');
    } finally {
        await cleanupFfmpegJob(ffmpeg, jobId, totalFrames, inputPath, outputPath);
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
        console.log('Loading processing engine... This may take 10-30 seconds on first load.', 'info');
    } else {
        console.log('Ready. Drop a photo or video clip to get started.', 'success');
    }
    refreshActions();
    updateUnloadProtection();
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
const customInputIds = ['customBg', 'customInk', 'customLowThresh', 'customHighThresh', 'customBilateral', 'customSigma', 'customUseBilateral', 'customBilateralPasses', 'customUseGaussian', 'customGaussianPasses', 'customUseMedian', 'customMedianPasses', 'customCleanSpeckles', 'customCleanSpecklesIntensity', 'customDarkBoost', 'customMergeDoubleEdge', 'customMergeDoubleEdgeIntensity'];
const customValueSpans = {
    customLowThresh: document.getElementById('customLowThreshVal'),
    customHighThresh: document.getElementById('customHighThreshVal'),
    customBilateral: document.getElementById('customBilateralVal'),
    customSigma: document.getElementById('customSigmaVal'),
    customBilateralPasses: document.getElementById('customBilateralPassesVal'),
    customGaussianPasses: document.getElementById('customGaussianPassesVal'),
    customMedianPasses: document.getElementById('customMedianPassesVal'),
    customCleanSpecklesIntensity: document.getElementById('customCleanSpecklesIntensityVal'),
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

attachDropZone();
resetWorkspace();
