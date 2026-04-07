const elements = {
    fileInput: document.getElementById('fileInput'),
    loadFFmpegBtn: document.getElementById('load-ffmpeg'),
    cancelBtn: document.getElementById('cancelBtn'),
    status: document.getElementById('status'),
    statusCard: document.querySelector('.status-card'),
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
    previewBtn: document.getElementById('previewBtn'),
    renderBtn: document.getElementById('renderBtn'),
    resetBtn: document.getElementById('resetBtn'),
    downloadLink: document.getElementById('download'),
    downloadCard: document.getElementById('downloadCard'),
    sourceCanvas: document.getElementById('sourceCanvas'),
    outputCanvas: document.getElementById('canvasOutput'),
    outputVideo: document.getElementById('outputVideo'),
    videoResult: document.getElementById('videoResult'),
    fileMeta: document.getElementById('fileMeta'),
    dropZone: document.getElementById('dropZone')
};

const STYLE_PRESETS = {
    studio: {
        label: 'Studio Ink',
        background: [248, 245, 237],
        ink: [23, 33, 47],
        lowThreshold: 38,
        highThreshold: 118,
        bilateralDiameter: 7,
        sigma: 52
    },
    manga: {
        label: 'Manga Contrast',
        background: [255, 255, 255],
        ink: [0, 0, 0],
        lowThreshold: 28,
        highThreshold: 96,
        bilateralDiameter: 9,
        sigma: 72
    },
    blueprint: {
        label: 'Blueprint Draft',
        background: [225, 237, 245],
        ink: [19, 57, 92],
        lowThreshold: 48,
        highThreshold: 144,
        bilateralDiameter: 5,
        sigma: 44
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

        this._worker.onmessage = ({ data: msg }) => {
            if (msg.type === 'cv-ready') {
                state.cvReady = true;
                refreshActions();
                setStatus('OpenCV ready. Drop an image or video to create line art.', 'success');
                return;
            }

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
                ctx.putImageData(new ImageData(msg.data, width, height), 0, 0);
                resolve();
            } else if (msg.type === 'error') {
                entry.reject(new Error(msg.message));
            }
        };

        this._worker.onerror = (error) => {
            console.error('OpenCV worker error:', error);
            for (const [, { reject }] of this._pending) {
                reject(new Error('Processing worker error.'));
            }
            this._pending.clear();
            setStatus('Processing worker error. Please reload the page.', 'error');
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

function setStatus(message, tone = 'info') {
    elements.status.textContent = message;
    elements.statusCard.dataset.tone = tone;
}

function setBusy(isBusy) {
    state.processing = isBusy;
    elements.cancelBtn.hidden = !isBusy;
    elements.cancelBtn.disabled = !isBusy;
    refreshActions();
}

function refreshActions() {
    const hasFile = Boolean(state.selectedFile);
    elements.previewBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.renderBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.fileInput.disabled = !state.cvReady || state.processing;
    elements.loadFFmpegBtn.disabled = !state.cvReady || state.processing || state.ffmpegReady;
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

function getSettings() {
    return {
        preset: STYLE_PRESETS[elements.preset.value],
        detail: Number(elements.detail.value),
        lineWeight: Number(elements.lineWeight.value),
        scale: Number(elements.scale.value),
        videoFps: Number(elements.videoFps.value)
    };
}

function computeScaledSize(width, height, scale) {
    const largestSide = Math.max(width, height);
    const dimensionCap = 1600;
    const capRatio = largestSide > dimensionCap ? dimensionCap / largestSide : 1;
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
        const size = computeScaledSize(state.sourceImage.naturalWidth, state.sourceImage.naturalHeight, settings.scale);
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
        const size = computeScaledSize(state.sourceVideo.videoWidth, state.sourceVideo.videoHeight, settings.scale);
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
    if (file.size > PRODUCTION_LIMITS.hardFileSizeBytes) {
        throw new Error('File is too large for reliable browser-side processing. Keep uploads under 250 MB.');
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

function drawMediaToCanvas(media, canvas, scale) {
    const naturalWidth = media.videoWidth || media.naturalWidth || media.width;
    const naturalHeight = media.videoHeight || media.naturalHeight || media.height;
    const size = computeScaledSize(naturalWidth, naturalHeight, scale);
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
    setStatus('Cancel requested. Finishing the current step...', 'warn');
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
                setStatus(`Encoding video... ${message}`, 'info');
            }
        });
        state.ffmpeg.on('progress', ({ progress }) => {
            if (typeof progress === 'number') {
                const percent = Math.min(100, Math.max(0, Math.round(progress * 100)));
                setStatus(`Encoding video... ${percent}%`, 'info');
                setProgress(92 + Math.round(percent * 0.08), 'Encoding final MP4...');
            }
        });
    }

    setStatus('Loading FFmpeg video export engine...', 'info');
    await state.ffmpeg.load({
        coreURL: 'vendor/ffmpeg-core.js',
        wasmURL: 'vendor/ffmpeg-core.wasm'
    });

    state.ffmpegReady = true;
    refreshActions();
    setStatus('OpenCV ready. Video export engine loaded.', 'success');
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
        return;
    }

    throw new Error('Unsupported file type. Use an image or video file.');
}

async function drawCurrentSource() {
    const settings = getSettings();

    if (state.fileKind === 'image' && state.sourceImage) {
        drawMediaToCanvas(state.sourceImage, elements.sourceCanvas, settings.scale);
        return;
    }

    if (state.fileKind === 'video' && state.sourceVideo) {
        const previewTime = Math.min(Math.max(state.sourceVideo.duration * 0.2, 0), Math.max(0, state.sourceVideo.duration - 0.05));
        await seekVideo(state.sourceVideo, previewTime);
        drawMediaToCanvas(state.sourceVideo, elements.sourceCanvas, settings.scale);
    }
}

async function renderPreview() {
    throwIfCancelled();
    await drawCurrentSource();
    throwIfCancelled();
    await processor.render(elements.sourceCanvas, elements.outputCanvas, getSettings());
    clearRenderedOutput();
    setStatus(
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
    setStatus('Image render complete. Download your PNG.', 'success');
}

async function renderVideoExport() {
    const ffmpeg = await loadFFmpeg();
    const settings = getSettings();
    const fps = settings.videoFps;
    const video = state.sourceVideo;
    const totalFrames = Math.max(1, Math.floor(video.duration * fps));
    const jobId = `job-${Date.now()}`;
    const extension = state.selectedFile.name.includes('.')
        ? state.selectedFile.name.slice(state.selectedFile.name.lastIndexOf('.'))
        : '.mp4';
    const safeBaseName = sanitizeBaseName(state.selectedFile.name);
    const inputPath = `${jobId}/input${extension}`;
    const framePattern = `${jobId}/frame-%05d.png`;
    const outputPath = `${jobId}/output.mp4`;
    state.activeJobId = jobId;

    try {
        await ffmpeg.createDir(jobId);
        await ffmpeg.writeFile(inputPath, new Uint8Array(await state.selectedFile.arrayBuffer()));

        for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
            throwIfCancelled();
            const frameTime = Math.min(video.duration, frameIndex / fps);
            await seekVideo(video, frameTime);
            drawMediaToCanvas(video, elements.sourceCanvas, settings.scale);
            await processor.render(elements.sourceCanvas, elements.outputCanvas, settings);

            const frameBlob = await canvasToBlob(elements.outputCanvas, 'image/png');
            const frameBytes = new Uint8Array(await frameBlob.arrayBuffer());
            const frameName = `${jobId}/frame-${String(frameIndex).padStart(5, '0')}.png`;
            await ffmpeg.writeFile(frameName, frameBytes);
            const framePercent = ((frameIndex + 1) / totalFrames) * 92;
            setProgress(framePercent, `Rendering frame ${frameIndex + 1} of ${totalFrames}...`);
            setStatus(`Rendering frame ${frameIndex + 1} of ${totalFrames}...`, 'info');
        }

        throwIfCancelled();
        setProgress(92, 'Encoding final MP4 in the browser...');
        setStatus('Encoding final MP4 in the browser...', 'info');
        await ffmpeg.exec([
            '-y',
            '-framerate', String(fps),
            '-i', framePattern,
            '-i', inputPath,
            '-map', '0:v:0',
            '-map', '1:a?',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-shortest',
            outputPath
        ]);

        throwIfCancelled();
        const outputData = await ffmpeg.readFile(outputPath);
        const videoBlob = new Blob([outputData.buffer], { type: 'video/mp4' });
        setDownload(videoBlob, `${safeBaseName}-lineart.mp4`);
        elements.outputVideo.src = state.outputUrl;
        elements.videoResult.hidden = false;
        setProgress(100, 'MP4 export ready.');
        setStatus('Video render complete. Download or review the MP4.', 'success');
    } finally {
        await cleanupFfmpegJob(ffmpeg, jobId, totalFrames, inputPath, outputPath);
        state.activeJobId = '';
    }
}

async function handleFileSelection(file) {
    if (!file) {
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
        setStatus(error.message, 'error');
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
    resetProgress();
    elements.fileInput.value = '';
    updateFileMeta('No file selected.');
    setAdvisory('Select a file to estimate browser workload.', 'info');
    drawEmptyCanvas(elements.sourceCanvas, 'Source preview');
    drawEmptyCanvas(elements.outputCanvas, 'Line-art preview');
    setStatus(
        state.cvReady
            ? 'Ready. Drop an image or video to create line art.'
            : 'Loading OpenCV engine...',
        'info'
    );
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
        setStatus(error.message, error.message === 'Render cancelled.' ? 'warn' : 'error');
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
        setStatus(error.message || 'Rendering failed.', error.message === 'Render cancelled.' ? 'warn' : 'error');
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

elements.loadFFmpegBtn.addEventListener('click', async () => {
    try {
        setBusy(true);
        await loadFFmpeg();
    } catch (error) {
        console.error(error);
        setStatus(error.message, 'error');
    } finally {
        setBusy(false);
    }
});

elements.fileInput.addEventListener('change', async (event) => {
    const [file] = event.target.files;
    await handleFileSelection(file);
});

elements.previewBtn.addEventListener('click', onPreviewClick);
elements.renderBtn.addEventListener('click', onRenderClick);
elements.cancelBtn.addEventListener('click', requestCancel);
elements.resetBtn.addEventListener('click', resetWorkspace);

['preset', 'detail', 'lineWeight', 'scale', 'videoFps'].forEach((id) => {
    document.getElementById(id).addEventListener('change', async () => {
        summarizeWorkload();

        if (!state.selectedFile || state.processing) {
            return;
        }

        await onPreviewClick();
    });
});

attachDropZone();
resetWorkspace();
