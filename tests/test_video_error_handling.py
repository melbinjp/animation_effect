import pytest
from playwright.sync_api import Page, expect
import threading
import http.server
import socketserver
import time
import os

PORT = 8000
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Suppress logging for cleaner test output
        pass

def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

@pytest.fixture(scope="session", autouse=True)
def setup_server():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1) # Wait for server to start
    yield

def test_video_processing_error_handling(page: Page):
    page.goto(f"http://localhost:{PORT}")

    # Mock FFmpeg and OpenCV
    mock_script = """
    window.onOpenCvReady = function() {
        console.log("Mock OpenCV ready");
        window.opencvReady = true;
        document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
        document.getElementById('load-ffmpeg').disabled = false;
    };

    // Simulate cv being present to avoid undefined errors during cartoonizeVideo
    window.cv = {
        imread: () => ({ delete: () => {} }),
        cvtColor: () => {},
        medianBlur: () => {},
        adaptiveThreshold: () => {},
        bitwise_not: () => {},
        bilateralFilter: () => {},
        bitwise_and: () => {},
        imshow: () => {},
        Mat: function() { this.delete = () => {}; },
        COLOR_RGBA2GRAY: 0,
        ADAPTIVE_THRESH_MEAN_C: 0,
        THRESH_BINARY: 0,
        BORDER_DEFAULT: 0
    };

    window.FFmpeg = {
        FFmpeg: class {
            constructor() {
                this.callbacks = {};
            }
            on(event, callback) {
                this.callbacks[event] = callback;
            }
            async load() {
                console.log("Mock FFmpeg loaded");
            }
            async writeFile() {
                console.log("Mock FFmpeg writeFile");
            }
            async exec() {
                console.log("Mock FFmpeg exec - throwing error");
                throw new Error("Simulated FFmpeg execution error");
            }
        },
        toBlobURL: async () => 'mock_url',
        fetchFile: async () => new Blob(['mock_data'])
    };

    // We also need to mock HTMLVideoElement.duration since cartoonizeVideo uses it
    Object.defineProperty(HTMLVideoElement.prototype, 'duration', {
        get: () => 0.01
    });

        // Mocking canvas.toDataURL to avoid issues
        HTMLCanvasElement.prototype.toDataURL = function() {
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==";
        };
    """

    # Expose a mock fetch to avoid hitting network
    def handle_route(route):
        url = route.request.url
        if "script.js" in url:
            # Modify script.js to initialize ffmpegReady to true
            with open("script.js", "r") as f:
                content = f.read()
            content = content.replace("let ffmpegReady = false;", "let ffmpegReady = true;")

            # We need to make sure the script uses the mocked FFmpeg properly.
            content = content.replace("const { FFmpeg, toBlobURL, fetchFile } = FFmpeg;", "const { FFmpeg: FFmpegClass, toBlobURL, fetchFile } = window.FFmpeg;\nconst FFmpeg = FFmpegClass;")

            # Make cartoonizeVideo globally accessible for the test
            content += "\nwindow.cartoonizeVideo = cartoonizeVideo;\nwindow.ffmpeg = ffmpeg;\n"

            route.fulfill(body=content, content_type="application/javascript")
        elif "ffmpeg" in url or "blob" in url:
            route.fulfill(body=b"console.log('mock ffmpeg js')", content_type="application/javascript")
        elif url.startswith("data:"):
            # Data URLs are usually handled by the browser directly, but just in case
            route.continue_()
        else:
            route.continue_()

    page.route("**/*", handle_route)

    page.add_init_script(script=mock_script)
    page.on("console", lambda msg: print(f"Browser console {msg.type}: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Browser error: {err}"))
    page.goto(f"http://localhost:{PORT}")

    # Manually trigger the onOpenCvReady to simulate it loading
    page.evaluate("window.onOpenCvReady()")

    # Let's ensure our window.ffmpeg object is actually working within script.js by just
    # printing whether it loaded or throw the reference error we saw earlier
    page.evaluate("""() => {
        if (typeof cartoonizeVideo === 'undefined') console.error('cartoonizeVideo is undefined');
        if (typeof ffmpeg === 'undefined') console.error('ffmpeg is undefined');
    }""")

    # In index.html, OpenCV is loaded via async script which takes time to fire onOpenCvReady.
    # Since we mocked it, and triggered it manually, wait for the DOM to reflect it.
    status_element = page.locator("#status")
    expect(status_element).to_have_text("OpenCV.js is ready.")

    # In our mock, ffmpeg.load doesn't actually trigger what script.js expects
    # (specifically setting ffmpegReady=true, changing button state, and updating status)
    # Let's bypass clicking and just manually call the load click handler logic, or
    # overwrite the ffmpegReady flag directly since we already mocked ffmpeg load earlier.
    # Actually, the button click listener calls `ffmpeg.load(...)` then sets `ffmpegReady = true`
    # and updates the status to 'Ready to cartoonify!'.
    # Wait, in the mock, `await ffmpeg.load(...)` returns immediately.
    # Let's see if the listener completes.
    # Wait, the event listener is async.

    # Click the "Load FFmpeg" button
    load_ffmpeg_btn = page.locator("#load-ffmpeg")
    expect(load_ffmpeg_btn).to_be_enabled()

    # To bypass `ffmpegReady = false` scoping issues, let's intercept `script.js`
    # and just remove the check completely or set `ffmpegReady = true` initially.
    # Actually, we can just intercept the request for script.js and modify it!

    # We also need to mock video playback because cartoonizeVideo awaits `seeked`
    # and relies on video playback. Let's force that event in page context.
    # We trigger cartoonizeVideo directly with a mock video object to test the
    # specific failure within ffmpeg.exec.

    page.evaluate("""() => {
        const mockVideo = {
            duration: 0.01,
            videoWidth: 10,
            videoHeight: 10,
            currentTime: 0,
            addEventListener: function(event, callback) {
                if (event === 'seeked') {
                        setTimeout(() => callback({}), 0);
                }
            }
        };
        const mockFile = new File(["dummy"], "dummy.mp4", { type: "video/mp4" });
            // Don't wait on it here, it will run async
            cartoonizeVideo(mockVideo, mockFile).catch(e => console.error("Unhandled in cartoonizeVideo", e));
    }""")

    # Since cartoonizeVideo calls ffmpeg.exec which we mocked to throw an error,
    # the status element should eventually display the error message.
    # Note: Using textContent matches what we updated in script.js
    expect(status_element).to_have_text("Error processing video. Please try again or use a different file.", timeout=10000)
