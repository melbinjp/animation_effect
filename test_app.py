import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # We need to mock things before the scripts run, so add_init_script is proper
        await page.add_init_script("""
            window.ffmpegReady = true;
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

            // Re-mock FFmpeg global object BEFORE script.js uses it
            window.FFmpeg = {
                FFmpeg: class {
                    on() {}
                    async load() {}
                    async writeFile() {}
                    async exec() {}
                    async readFile() { return new Uint8Array([0]); }
                },
                fetchFile: async () => new Blob(),
                toBlobURL: async () => ''
            };

            // To ensure script uses mocked FFmpeg
            window.ffmpeg = new window.FFmpeg.FFmpeg();
        """)

        await page.goto("http://localhost:8000")

        # Wait for script.js to initialize (could be blocked on opencv load)
        # Mock the opencvReady function invocation
        await page.evaluate("if(window.onOpenCvReady) window.onOpenCvReady();")

        result = await page.evaluate("""
            async () => {
                const video = {
                    duration: 0.1, // creates 3 frames
                    videoWidth: 100,
                    videoHeight: 100,
                    addEventListener: (event, callback, options) => {
                        // Simulate seeked event immediately
                        if(event === 'seeked') callback();
                    },
                    currentTime: 0
                };

                const file = new File(["dummy"], "test.mp4", { type: "video/mp4" });

                try {
                    await window.cartoonizeVideo(video, file);
                    return true;
                } catch (e) {
                    console.error("Test error:", e);
                    return false;
                }
            }
        """)
        print("Test successful:", result)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
