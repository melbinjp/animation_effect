const fileInput = document.getElementById('fileInput');
const canvas = document.getElementById('canvasOutput');
const downloadLink = document.getElementById('download');
const statusElement = document.getElementById('status');
const loadFFmpegBtn = document.getElementById('load-ffmpeg');
const ctx = canvas.getContext('2d');

let opencvReady = false;
let ffmpegReady = false;

const { FFmpeg, toBlobURL, fetchFile } = FFmpeg;
const ffmpeg = new FFmpeg();

ffmpeg.on('log', ({ message }) => {
    console.log(message);
    statusElement.innerHTML = message;
});

window.onOpenCvReady = function() {
    statusElement.innerHTML = 'OpenCV.js is ready.';
    opencvReady = true;
    loadFFmpegBtn.disabled = false;
}

loadFFmpegBtn.addEventListener('click', async () => {
    statusElement.innerHTML = 'Loading FFmpeg-core...';
    loadFFmpegBtn.disabled = true;
    const baseURL = 'vendor';
    await ffmpeg.load({
        coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'application/javascript'),
        wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
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
                video.src = event.target.result;
                cartoonizeVideo(video, file);
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

async function cartoonizeVideo(video, file) {
    if (!ffmpegReady) {
        statusElement.innerHTML = 'FFmpeg is not loaded yet. Please click the "Load FFmpeg" button.';
        return;
    }
    statusElement.innerHTML = 'Processing video... this might take a while.';

    await ffmpeg.writeFile('input.mp4', await fetchFile(file));

    const videoDuration = video.duration;
    const frameRate = 30; // A reasonable frame rate
    const totalFrames = Math.floor(videoDuration * frameRate);

    for (let i = 0; i < totalFrames; i++) {
        video.currentTime = i / frameRate;
        await new Promise(resolve => video.addEventListener('seeked', resolve, { once: true }));

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
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

        const frameData = canvas.toDataURL('image/png');
        const frameBlob = await (await fetch(frameData)).blob();
        await ffmpeg.writeFile(`frame-${i.toString().padStart(5, '0')}.png`, new Uint8Array(await frameBlob.arrayBuffer()));

        src.delete();
        dst.delete();
        gray.delete();
        edges.delete();
        color.delete();

        statusElement.innerHTML = `Processing frame ${i + 1} of ${totalFrames}`;
    }

    statusElement.innerHTML = 'Combining frames and audio...';

    await ffmpeg.exec([
        '-framerate', `${frameRate}`,
        '-i', 'frame-%05d.png',
        '-i', 'input.mp4',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        'output.mp4'
    ]);

    const data = await ffmpeg.readFile('output.mp4');
    const url = URL.createObjectURL(new Blob([data.buffer], { type: 'video/mp4' }));
    downloadLink.href = url;
    downloadLink.style.display = 'block';
    downloadLink.download = 'cartoonized_video.mp4';
    statusElement.innerHTML = 'Done! Your video is ready for download.';
}
