import pytest
from playwright.sync_api import Page, expect

def test_unsupported_file_type_shows_error(page: Page):
    # Mock OpenCV and FFmpeg to prevent loading large binaries
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
    """)

    ffmpeg_mock = """
        window.FFmpeg = {
            FFmpeg: class {
                constructor() {
                    this.on = () => {};
                    this.load = () => Promise.resolve();
                    this.writeFile = () => Promise.resolve();
                    this.exec = () => Promise.resolve();
                    this.readFile = () => Promise.resolve(new Uint8Array());
                }
            },
            toBlobURL: () => Promise.resolve(''),
            fetchFile: () => Promise.resolve(new Uint8Array())
        };
    """

    page.route("**/ffmpeg.js", lambda route: route.fulfill(body=ffmpeg_mock, status=200, content_type="application/javascript"))
    page.route("**/opencv.js", lambda route: route.fulfill(body="if (window.onOpenCvReady) window.onOpenCvReady();", status=200, content_type="application/javascript"))

    page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
    page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

    # Navigate to the local server
    page.goto("http://localhost:8000/index.html")

    # Get the file input and status element
    file_input = page.locator("#fileInput")
    status = page.locator("#status")

    page.evaluate('document.getElementById("fileInput").disabled = false')

    # Ensure the input is enabled
    expect(file_input).to_be_enabled()

    # Force the accept attribute to allow anything for the test so we can use set_input_files easily
    page.evaluate('document.getElementById("fileInput").removeAttribute("accept")')

    # Wait for the status element to be set by script.js first
    expect(status).to_have_text("OpenCV.js is ready.")

    # Force test file reading via JS and dispatch event so we make sure `change` is properly intercepted
    page.evaluate('''() => {
        const fileInput = document.getElementById('fileInput');
        const file = new File(['Hello world!'], 'test.txt', { type: 'text/plain' });

        // This simulates a file drop instead of regular upload
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }''')

    # Because `fileReader.readAsDataURL` is asynchronous, wait for status to update
    expect(status).to_have_text("Error: Unsupported file type. Please upload an image or video.", timeout=5000)
