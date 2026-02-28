import pytest
from playwright.sync_api import Page, expect

def test_cartoonize_video_ffmpeg_not_ready(page: Page):
    page.goto("http://localhost:8000")

    # Ensure page is loaded
    expect(page.locator("h1")).to_have_text("Cartoonizer")

    # Set ffmpegReady to false (default state)

    # Trigger cartoonizeVideo directly via evaluate
    page.evaluate("""() => {
        const video = document.createElement('video');
        const file = new File([''], 'dummy.mp4', { type: 'video/mp4' });
        cartoonizeVideo(video, file);
    }""")

    # Verify status message
    expect(page.locator("#status")).to_have_text('FFmpeg is not loaded yet. Please click the "Load FFmpeg" button.')


def test_cartoonize_video_success(page: Page):
    # Ok, the absolute foolproof way is to modify the DOM itself before it gets run,
    # or just let it load normally, and THEN override `cartoonizeVideo`.
    # Wait, the task says: "This function has complex dependencies involving `ffmpeg`, `cv`, DOM video events (`seeked`), and promises. Mocking these interactions thoroughly requires significant setup and extensive test code."
    # Let's use Playwright to intercept `script.js` and modify it.

    def handle_script_route(route):
        response = route.fetch()
        original_script = response.text()

        # Replace the `const ffmpeg` declaration with `window.ffmpeg`
        modified_script = original_script.replace(
            "const { FFmpeg, toBlobURL, fetchFile } = FFmpeg;\nconst ffmpeg = new FFmpeg();",
            "const { FFmpeg: FFmpegClass, toBlobURL, fetchFile } = FFmpeg;\nwindow.ffmpeg = new FFmpegClass();\nconst ffmpeg = window.ffmpeg;"
        )

        route.fulfill(
            response=response,
            content_type="application/javascript",
            body=modified_script
        )

    page.route("**/script.js", handle_script_route)

    page.add_init_script("""
        window.cv = {
            imread: () => ({ delete: () => {} }),
            Mat: class { delete() {} },
            cvtColor: () => {},
            medianBlur: () => {},
            adaptiveThreshold: () => {},
            bitwise_not: () => {},
            bilateralFilter: () => {},
            bitwise_and: () => {},
            imshow: () => {},
            COLOR_RGBA2GRAY: 0,
            ADAPTIVE_THRESH_MEAN_C: 0,
            THRESH_BINARY: 0,
            BORDER_DEFAULT: 0
        };

        window.mockFfmpegExecCalled = false;
        window.mockFfmpegWriteFileCount = 0;
    """)

    # Mock FFmpeg completely so it doesn't try to load the heavy wasm
    page.route("**/vendor/ffmpeg.js", lambda route: route.fulfill(
        content_type="application/javascript",
        body="""
            window.FFmpeg = {
                FFmpeg: class {
                    constructor() {
                        this.on = () => {};
                    }
                    async load() { return Promise.resolve(); }
                    async writeFile(name, data) {
                        window.mockFfmpegWriteFileCount++;
                        return Promise.resolve();
                    }
                    async exec(args) {
                        window.mockFfmpegExecCalled = true;
                        return Promise.resolve();
                    }
                    async readFile(name) {
                        return new Uint8Array([0, 1, 2, 3]);
                    }
                },
                toBlobURL: async () => 'blob:dummy',
                fetchFile: async () => new Uint8Array([0])
            };
        """
    ))

    page.route("**/opencv.js", lambda route: route.fulfill(body="window.onOpenCvReady();"))

    page.goto("http://localhost:8000")

    # Wait for the page to be ready
    expect(page.locator("h1")).to_have_text("Cartoonizer")

    # Bypass the click and just set ready directly
    page.evaluate("() => { ffmpegReady = true; }")

    # We need a dummy video with a non-zero duration to test the loop
    # Create an in-memory video and trigger cartoonizeVideo
    page.evaluate(r"""async () => {
        try {
            const video = document.createElement('video');
            // Use a proper minimal video data URL
            video.src = 'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAAAM21kYXQAAAHXQAAABpIBAAAB//+AAEAQQEBAQEA=';

            // Override duration and video dimensions for testing
            Object.defineProperty(video, 'duration', { value: 0.1 });
            Object.defineProperty(video, 'videoWidth', { value: 100 });
            Object.defineProperty(video, 'videoHeight', { value: 100 });

            // We need to trigger the seeked event manually because the dummy video won't actually seek
            video.addEventListener('seeked', () => {
                console.log('Seeked triggered');
            });

            const originalCurrentTime = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'currentTime');
            Object.defineProperty(video, 'currentTime', {
                set: function(val) {
                    originalCurrentTime.set.call(this, val);
                    // Dispatch seeked event asynchronously to mimic real behavior
                    setTimeout(() => this.dispatchEvent(new Event('seeked')), 0);
                }
            });

            const file = new File([''], 'dummy.mp4', { type: 'video/mp4' });

            // Set canvas size manually to prevent errors in read/draw
            document.getElementById('canvasOutput').width = 100;
            document.getElementById('canvasOutput').height = 100;

            // Call the function and await it
            await cartoonizeVideo(video, file);
        } catch(e) {
            console.error("Error in evaluate block:", e);
            throw e;
        }
    }""")

    # Verify status changed to Done!
    expect(page.locator("#status")).to_have_text('Done! Your video is ready for download.', timeout=10000)

    # Verify download link is visible and correct
    expect(page.locator("#download")).to_be_visible()
    expect(page.locator("#download")).to_have_attribute("download", "cartoonized_video.mp4")

    # Verify our mocks were actually used
    exec_called = page.evaluate("() => window.mockFfmpegExecCalled")
    assert exec_called is True

    write_count = page.evaluate("() => window.mockFfmpegWriteFileCount")
    # 1 for input.mp4, plus frames (duration 0.1 * frameRate 30 = 3 frames)
    assert write_count == 4
