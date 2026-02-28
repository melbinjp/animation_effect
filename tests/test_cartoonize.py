import pytest
from playwright.sync_api import Page, expect
import threading
from http.server import SimpleHTTPRequestHandler
import socketserver
import time
import socket

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

PORT = get_free_port()
httpd_server = None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)

def start_server():
    global httpd_server
    try:
        httpd_server = socketserver.TCPServer(("", PORT), Handler)
        httpd_server.serve_forever()
    except OSError:
        pass # Port probably already in use, ignore for tests

@pytest.fixture(scope="session", autouse=True)
def server():
    global httpd_server
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    time.sleep(1) # wait for server to start
    yield
    if httpd_server:
        httpd_server.shutdown()
        httpd_server.server_close()

def test_cartoonizeImage_processes_image_and_cleans_up(page: Page):
    # Mock cv methods and objects to track memory management
    mock_cv_script = """
    window.cv_calls = [];
    window.deleted_mats = 0;
    window.created_mats = 0;

    class MockMat {
        constructor() {
            this.id = window.created_mats++;
            this.deleted = false;
        }
        delete() {
            if (!this.deleted) {
                this.deleted = true;
                window.deleted_mats++;
            }
        }
    }

    window.cv = {
        Mat: MockMat,
        imread: (canvas) => {
            window.cv_calls.push(['imread', canvas.id]);
            return new MockMat();
        },
        COLOR_RGBA2GRAY: 'COLOR_RGBA2GRAY',
        ADAPTIVE_THRESH_MEAN_C: 'ADAPTIVE_THRESH_MEAN_C',
        THRESH_BINARY: 'THRESH_BINARY',
        BORDER_DEFAULT: 'BORDER_DEFAULT',
        cvtColor: (src, dst, code, dstCn) => { window.cv_calls.push(['cvtColor', src.id, dst.id, code, dstCn]); },
        medianBlur: (src, dst, ksize) => { window.cv_calls.push(['medianBlur', src.id, dst.id, ksize]); },
        adaptiveThreshold: (src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C) => {
            window.cv_calls.push(['adaptiveThreshold', src.id, dst.id, maxValue, adaptiveMethod, thresholdType, blockSize, C]);
        },
        bitwise_not: (src, dst, mask) => { window.cv_calls.push(['bitwise_not', src.id, dst.id, mask]); },
        bilateralFilter: (src, dst, d, sigmaColor, sigmaSpace, borderType) => {
            window.cv_calls.push(['bilateralFilter', src.id, dst.id, d, sigmaColor, sigmaSpace, borderType]);
        },
        bitwise_and: (src1, src2, dst, mask) => { window.cv_calls.push(['bitwise_and', src1.id, src2.id, dst.id, mask?.id]); },
        imshow: (canvas, mat) => { window.cv_calls.push(['imshow', canvas.id, mat.id]); }
    };

    // Mock HTMLCanvasElement.toDataURL to prevent base64 encoding overhead
    const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function() {
        window.cv_calls.push(['toDataURL', this.id]);
        return 'data:image/png;base64,mock_data';
    };
    """

    page.add_init_script(mock_cv_script)

    # Intercept and abort requests to real opencv.js
    page.route("**/opencv.js", lambda route: route.abort())

    page.goto(f"http://localhost:{PORT}/index.html")

    # Run the function
    page.evaluate("cartoonizeImage()")

    # Check cv_calls
    cv_calls = page.evaluate("window.cv_calls")

    # Expected call sequence:
    # 0: imread
    # 1: cvtColor
    # 2: medianBlur
    # 3: adaptiveThreshold
    # 4: bitwise_not
    # 5: bilateralFilter
    # 6: bitwise_and
    # 7: imshow
    # 8: toDataURL

    assert len(cv_calls) == 9
    assert cv_calls[0][0] == 'imread'
    assert cv_calls[0][1] == 'canvasOutput'

    assert cv_calls[1][0] == 'cvtColor'
    assert cv_calls[1][3] == 'COLOR_RGBA2GRAY'

    assert cv_calls[2][0] == 'medianBlur'
    assert cv_calls[2][3] == 5

    assert cv_calls[3][0] == 'adaptiveThreshold'
    assert cv_calls[3][3] == 255
    assert cv_calls[3][4] == 'ADAPTIVE_THRESH_MEAN_C'
    assert cv_calls[3][5] == 'THRESH_BINARY'
    assert cv_calls[3][6] == 9
    assert cv_calls[3][7] == 9

    assert cv_calls[4][0] == 'bitwise_not'

    assert cv_calls[5][0] == 'bilateralFilter'
    assert cv_calls[5][3] == 9
    assert cv_calls[5][4] == 250
    assert cv_calls[5][5] == 250
    assert cv_calls[5][6] == 'BORDER_DEFAULT'

    assert cv_calls[6][0] == 'bitwise_and'

    assert cv_calls[7][0] == 'imshow'
    assert cv_calls[7][1] == 'canvasOutput'

    assert cv_calls[8][0] == 'toDataURL'
    assert cv_calls[8][1] == 'canvasOutput'

    # Verify memory management: all created mats should be deleted
    created = page.evaluate("window.created_mats")
    deleted = page.evaluate("window.deleted_mats")

    # src + dst + gray + edges + color = 5 Mats
    assert created == 5, f"Expected 5 Mats to be created, but got {created}"
    assert deleted == 5, f"Expected 5 Mats to be deleted, but got {deleted}"
    assert created == deleted, "Memory leak: Not all created cv.Mats were deleted"

    # Verify download link was updated correctly
    download_display = page.evaluate("document.getElementById('download').style.display")
    download_href = page.evaluate("document.getElementById('download').href")
    download_name = page.evaluate("document.getElementById('download').download")

    assert download_display == 'block'
    assert download_href == 'data:image/png;base64,mock_data'
    assert download_name == 'cartoonized_image.png'
