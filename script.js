const fileInput = document.getElementById('fileInput');
const canvas = document.getElementById('canvasOutput');
const downloadLink = document.getElementById('download');
const statusElement = document.getElementById('status');
const loadFFmpegBtn = document.getElementById('load-ffmpeg');
const ctx = canvas.getContext('2d');

let opencvReady = false;
let ffmpegReady = false;

const ffmpeg = new FFmpeg.FFmpeg();

function onOpenCvReady() {
    statusElement.innerHTML = 'OpenCV.js is ready.';
    opencvReady = true;
    loadFFmpegBtn.disabled = false;
}

loadFFmpegBtn.addEventListener('click', async () => {
    statusElement.innerHTML = 'Loading FFmpeg-core...';
    loadFFmpegBtn.disabled = true;
    const baseURL = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.15/dist/umd'
    await ffmpeg.load({
        coreURL: await FFmpegUtil.toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
        wasmURL: await FFmpegUtil.toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    });
    ffmpegReady = true;
    fileInput.disabled = false;
    statusElement.innerHTML = 'Ready to cartoonify!';
});


fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            if (file.type.startsWith('image/')) {
                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    cartoonizeImage();
                };
                img.src = event.target.result;
            } else if (file.type.startsWith('video/')) {
                const video = document.createElement('video');
                video.muted = true;
                video.src = event.target.result;
                video.addEventListener('loadeddata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    cartoonizeVideo(video);
                });
            }
        };
        reader.readAsDataURL(file);
    }
});

function cartoonizeImage() {
    let src = cv.imread(canvas);
    let dst = new cv.Mat();
    let gray = new cv.Mat();
    let edges = new cv.Mat();
    let color = new cv.Mat();

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    cv.medianBlur(gray, gray, 5);
    cv.adaptiveThreshold(gray, edges, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9);
    cv.bitwise_not(edges, edges);
    cv.bilateralFilter(src, color, 9, 250, 250, cv.BORDER_DEFAULT);
    cv.bitwise_and(color, color, dst, edges);

    cv.imshow(canvas, dst);
    src.delete();
    dst.delete();
    gray.delete();
    edges.delete();
    color.delete();

    downloadLink.href = canvas.toDataURL();
    downloadLink.style.display = 'block';
    downloadLink.download = 'cartoonized_image.png';
}

function cartoonizeVideo(video) {
    // This function will be rewritten in the next step
    // For now, I will just leave the old code here as a placeholder
    const stream = canvas.captureStream();
    const audioContext = new AudioContext();
    const audioSource = audioContext.createMediaElementSource(video);
    const dest = audioContext.createMediaStreamDestination();
    audioSource.connect(dest);
    stream.addTrack(dest.stream.getAudioTracks()[0]);

    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });
    const chunks = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        downloadLink.href = url;
        downloadLink.style.display = 'block';
        downloadLink.download = 'cartoonized_video.webm';
    };

    const processVideo = () => {
        if (video.paused || video.ended) {
            recorder.stop();
            return;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        let src = cv.imread(canvas);
        let dst = new cv.Mat();
        let gray = new cv.Mat();
        let edges = new cv.Mat();
        let color = new cv.Mat();

        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        cv.medianBlur(gray, gray, 5);
        cv.adaptiveThreshold(gray, edges, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9);
        cv.bitwise_not(edges, edges);
        cv.bilateralFilter(src, color, 9, 250, 250, cv.BORDER_DEFAULT);
        cv.bitwise_and(color, color, dst, edges);

        cv.imshow(canvas, dst);
        src.delete();
        dst.delete();
        gray.delete();
        edges.delete();
        color.delete();
        requestAnimationFrame(processVideo);
    };

    video.play();
    recorder.start();
    processVideo();
}
