import asyncio
from playwright.async_api import async_playwright
import time

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # We evaluate a script that creates a mock video and measures the performance of two loops.
        result = await page.evaluate("""
            async () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                // Mock video properties
                const video = {
                    videoWidth: 1920,
                    videoHeight: 1080
                };

                const totalFrames = 1000;

                // Test 1: Resize inside loop
                const start1 = performance.now();
                for (let i = 0; i < totalFrames; i++) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    ctx.fillRect(0, 0, canvas.width, canvas.height); // simulate drawing
                }
                const end1 = performance.now();

                // Test 2: Resize outside loop
                const start2 = performance.now();
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                for (let i = 0; i < totalFrames; i++) {
                    ctx.fillRect(0, 0, canvas.width, canvas.height); // simulate drawing
                }
                const end2 = performance.now();

                return {
                    insideLoopMs: end1 - start1,
                    outsideLoopMs: end2 - start2
                };
            }
        """)

        print("Performance Benchmark Results:")
        print(f"Resizing INSIDE loop (Current):  {result['insideLoopMs']:.2f} ms")
        print(f"Resizing OUTSIDE loop (Optimized): {result['outsideLoopMs']:.2f} ms")
        print(f"Improvement: {(result['insideLoopMs'] / result['outsideLoopMs']):.2f}x faster")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
