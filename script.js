const elements = {
    fileInput: document.getElementById('fileInput'),
    loadFFmpegBtn: document.getElementById('load-ffmpeg'),
    status: document.getElementById('status'),
    statusCard: document.querySelector('.status-card'),
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
    processing: false
};

class LineArtProcessor {
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

    render(sourceCanvas, destinationCanvas, settings) {
        this.ensureSize(sourceCanvas.width, sourceCanvas.height);

        const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
        const sourceImageData = sourceContext.getImageData(0, 0, this.width, this.height);
        this.src.data.set(sourceImageData.data);

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

        destinationCanvas.width = this.width;
        destinationCanvas.height = this.height;

        const outputContext = destinationCanvas.getContext('2d');
        const outputImageData = outputContext.createImageData(this.width, this.height);
        const mask = this.edges.data;
        const output = outputImageData.data;

        for (let index = 0; index < mask.length; index += 1) {
            const offset = index * 4;
            const color = mask[index] > 127 ? settings.preset.background : settings.preset.ink;
            output[offset] = color[0];
            output[offset + 1] = color[1];
            output[offset + 2] = color[2];
            output[offset + 3] = 255;
        }

        outputContext.putImageData(outputImageData, 0, 0);
    }
}

const processor = new LineArtProcessor();

function setStatus(message, tone = 'info') {
    elements.status.textContent = message;
    elements.statusCard.dataset.tone = tone;
}

function setBusy(isBusy) {
    state.processing = isBusy;
    refreshActions();
}

function refreshActions() {
    const hasFile = Boolean(state.selectedFile);
    elements.previewBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.renderBtn.disabled = !state.cvReady || !hasFile || state.processing;
    elements.loadFFmpegBtn.disabled = !state.cvReady || state.processing || state.ffmpegReady;
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
    await drawCurrentSource();
    processor.render(elements.sourceCanvas, elements.outputCanvas, getSettings());
    clearRenderedOutput();
    setStatus(
        state.fileKind === 'video'
            ? 'Preview ready. Use Render final to process the full clip.'
            : 'Preview ready. Use Render final to export the PNG.',
        'success'
    );
}

async function renderImageExport() {
    await renderPreview();
    const fileName = `${sanitizeBaseName(state.selectedFile.name)}-lineart.png`;
    const blob = await canvasToBlob(elements.outputCanvas, 'image/png');
    setDownload(blob, fileName);
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

    await ffmpeg.createDir(jobId);
    await ffmpeg.writeFile(inputPath, new Uint8Array(await state.selectedFile.arrayBuffer()));

    for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
        const frameTime = Math.min(video.duration, frameIndex / fps);
        await seekVideo(video, frameTime);
        drawMediaToCanvas(video, elements.sourceCanvas, settings.scale);
        processor.render(elements.sourceCanvas, elements.outputCanvas, settings);

        const frameBlob = await canvasToBlob(elements.outputCanvas, 'image/png');
        const frameBytes = new Uint8Array(await frameBlob.arrayBuffer());
        const frameName = `${jobId}/frame-${String(frameIndex).padStart(5, '0')}.png`;
        await ffmpeg.writeFile(frameName, frameBytes);
        setStatus(`Rendering frame ${frameIndex + 1} of ${totalFrames}...`, 'info');
    }

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

    const outputData = await ffmpeg.readFile(outputPath);
    const videoBlob = new Blob([outputData.buffer], { type: 'video/mp4' });
    setDownload(videoBlob, `${safeBaseName}-lineart.mp4`);
    elements.outputVideo.src = state.outputUrl;
    elements.videoResult.hidden = false;
    setStatus('Video render complete. Download or review the MP4.', 'success');
}

async function handleFileSelection(file) {
    if (!file) {
        return;
    }

    try {
        setBusy(true);
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
        setStatus(error.message, 'error');
    } finally {
        setBusy(false);
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
    elements.fileInput.value = '';
    updateFileMeta('No file selected.');
    drawEmptyCanvas(elements.sourceCanvas, 'Source preview');
    drawEmptyCanvas(elements.outputCanvas, 'Line-art preview');
    setStatus(
        state.cvReady
            ? 'Ready. Drop an image or video to create line art.'
            : 'Loading OpenCV engine...',
        'info'
    );
    refreshActions();
}

async function onPreviewClick() {
    if (!state.selectedFile || state.processing) {
        return;
    }

    try {
        setBusy(true);
        await renderPreview();
    } catch (error) {
        console.error(error);
        setStatus(error.message, 'error');
    } finally {
        setBusy(false);
    }
}

async function onRenderClick() {
    if (!state.selectedFile || state.processing) {
        return;
    }

    try {
        setBusy(true);

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
        setStatus(error.message || 'Rendering failed.', 'error');
    } finally {
        setBusy(false);
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

window.onOpenCvReady = function onOpenCvReady() {
    state.cvReady = true;
    elements.fileInput.disabled = false;
    refreshActions();
    setStatus('OpenCV ready. Drop an image or video to create line art.', 'success');
};

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
elements.resetBtn.addEventListener('click', resetWorkspace);

['preset', 'detail', 'lineWeight', 'scale'].forEach((id) => {
    document.getElementById(id).addEventListener('change', async () => {
        if (!state.selectedFile || state.processing) {
            return;
        }

        await onPreviewClick();
    });
});

attachDropZone();
resetWorkspace();
