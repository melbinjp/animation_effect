import pytest
import subprocess
import time
import socket
from playwright.sync_api import Page, expect

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

@pytest.fixture(scope="session")
def server_port():
    port = get_free_port()
    # Start the local HTTP server in a subprocess
    proc = subprocess.Popen(["python3", "-m", "http.server", str(port)])

    # Wait a moment for the server to start
    time.sleep(1)

    yield port

    # Terminate the server when the session completes
    proc.terminate()
    proc.wait()

@pytest.fixture
def index_url(server_port):
    return f"http://localhost:{server_port}/index.html"

def test_initial_state(page: Page, index_url: str):
    """Test the initial DOM state before OpenCV and FFmpeg are loaded."""
    # We want to intercept the OpenCV script so it doesn't actually load and we can control the state.
    # We also have to prevent the `onload` from triggering by not fulfilling the request until we want.
    page.route("https://docs.opencv.org/4.5.2/opencv.js", lambda route: route.abort())

    page.add_init_script("""
        window.FFmpeg = {
            FFmpeg: class {
                constructor() {}
                on(event, callback) {}
                async load() { return Promise.resolve(); }
            },
            toBlobURL: async (url, type) => url,
            fetchFile: async (file) => new Uint8Array()
        };
        window.FFmpegWASM = window.FFmpeg;
        // Do not auto-execute onOpenCvReady here or intercept the callback
    """)

    page.goto(index_url)

    expect(page.locator("h1")).to_have_text("Cartoonizer")
    expect(page.locator("#status")).to_have_text("Loading OpenCV.js...")
    expect(page.locator("#load-ffmpeg")).to_be_disabled()
    expect(page.locator("#fileInput")).to_be_disabled()
    expect(page.locator("#download")).to_be_hidden()

def test_opencv_ready_enables_ffmpeg_button(page: Page, index_url: str):
    """Test that when OpenCV is ready, the FFmpeg button is enabled."""
    # Prevent real opencv from loading
    page.route("https://docs.opencv.org/4.5.2/opencv.js", lambda route: route.fulfill(status=200, body=""))

    # Add FFmpeg mock before scripts load
    page.add_init_script("""
        window.FFmpeg = {
            FFmpeg: class {
                constructor() {}
                on(event, callback) {}
                async load() { return Promise.resolve(); }
            },
            toBlobURL: async (url, type) => url,
            fetchFile: async (file) => new Uint8Array()
        };
        window.FFmpegWASM = window.FFmpeg;
    """)
    page.route("**/vendor/ffmpeg.js", lambda route: route.fulfill(status=200, body=""))

    # Capture console output to debug
    page.on("console", lambda msg: print(f"Console: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Page Error: {err}"))

    page.goto(index_url)

    # Wait for the page to be fully loaded so window.onOpenCvReady is available
    page.wait_for_load_state("networkidle")

    # Simulate onOpenCvReady being called
    page.evaluate("""
        if (typeof window.onOpenCvReady === 'function') {
            window.onOpenCvReady();
        } else {
            console.error('onOpenCvReady is not defined');
        }
    """)

    expect(page.locator("#status")).to_have_text("OpenCV.js is ready.")
    expect(page.locator("#load-ffmpeg")).to_be_enabled()
    expect(page.locator("#fileInput")).to_be_disabled()


def test_mocked_ffmpeg_flow(page: Page, index_url: str):
    # We need to mock FFmpeg before script.js is loaded
    page.add_init_script("""
        window.FFmpeg = {
            FFmpeg: class {
                constructor() {}
                on(event, callback) {}
                async load() { return Promise.resolve(); }
                async exec() { return Promise.resolve(); }
                async writeFile() { return Promise.resolve(); }
                async readFile() { return new Uint8Array(); }
            },
            toBlobURL: async (url, type) => url,
            fetchFile: async (file) => new Uint8Array()
        };
        window.FFmpegWASM = window.FFmpeg;
    """)
    page.route("**/vendor/ffmpeg.js", lambda route: route.fulfill(status=200, body=""))
    # Intercept OpenCV
    page.route("https://docs.opencv.org/4.5.2/opencv.js", lambda route: route.fulfill(status=200, body=""))

    # Capture console output to debug
    page.on("console", lambda msg: print(f"Console: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Page Error: {err}"))

    page.goto(index_url)

    # Wait for the page to be fully loaded
    page.wait_for_load_state("networkidle")

    # Simulate OpenCV ready
    page.evaluate("""
        if (typeof window.onOpenCvReady === 'function') {
            window.onOpenCvReady();
        } else {
            console.error('onOpenCvReady is not defined');
        }
    """)
    expect(page.locator("#load-ffmpeg")).to_be_enabled()

    # Click load FFmpeg
    page.locator("#load-ffmpeg").click()

    # Wait for the status to update indicating it's ready
    expect(page.locator("#status")).to_have_text("Ready to cartoonify!")
    expect(page.locator("#fileInput")).to_be_enabled()
    expect(page.locator("#load-ffmpeg")).to_be_disabled()
