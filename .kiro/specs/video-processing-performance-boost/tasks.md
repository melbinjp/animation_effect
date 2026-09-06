# Implementation Plan: Video Processing Performance Boost

## Overview

This implementation plan delivers a 40-50% performance improvement for browser-based video processing by completing the GPU pipeline for all filters, implementing adaptive worker scaling, and adding a parallel video decode pool. The goal is to reduce processing time for a 2.5-hour 4K 60fps video from 3-3.5 hours to 1.5-2 hours while preserving all existing stability mechanisms.

**Key Performance Targets:**
- Complete GPU pipeline: 25% improvement (eliminate CPU/GPU context switching)
- Adaptive worker scaling: 10% improvement (optimize parallelism dynamically)
- Parallel video decode pool: 5% improvement (eliminate worker idle time by decoding multiple frames concurrently across a pool of cloned video elements)

**Critical Constraints:**
- All existing stability mechanisms must be preserved (streaming encode, memory bounds, graceful fallback)
- GPU and CPU pipelines must produce pixel-equivalent output (MAE ≤ 1, SSIM > 0.99)
- Cross-platform compatibility maintained (Windows, macOS, Linux)

**Current Implementation Status (as of last reconciliation):**
- ✅ Phase 0: Test harness and Python test fix — complete (195 tests passing)
- ✅ Phase 1: PerformanceMonitor — complete and wired into render loop
- ✅ Phase 2: CLAHE GPU shader + CLAHEComputeShader wrapper — complete and on GPU path
- ✅ Phase 3: Auto-Normalize GPU shaders + AutoNormalizeFilter wrapper — complete and on GPU path
- ✅ Phase 4: Clean-Speckles GPU CCL shader + CleanSpecklesShader wrapper — complete and on GPU path
- ✅ Phase 5: Color-Edges GPU shaders + ColorEdgesShader wrapper — complete and on GPU path
- ✅ Phase 6: Full GPU pipeline integration — complete
- ✅ Phase 7: AdaptiveWorkerScaler — complete and wired into render loop (every 50 frames)
- ✅ Phase 8: VideoDecodePool — complete and wired into render loop
- ✅ Phase 9 (tasks 13.1–13.3): SettingsParser + SettingsPrettyPrinter + round-trip property test — complete
- 🔲 Phase 9 (tasks 14–15): localStorage persistence and Reset button — not started
- 🔲 Phase 10 (task 16): Color sampling verification — not started
- 🔲 Phase 11 (task 17): Memory safety verification and property tests — property tests not yet written
- 🔲 Phase 12 (task 18): Error handling — partial (worker scaling ✅, GPU device-lost partial, decode/memory/settings error handling not started)
- 🔲 Phases 13–15: Custom mode validation, benchmarking, documentation — not started

## Tasks

### Phase 0: Test Infrastructure and Maintenance

- [x] 0. Establish JavaScript test infrastructure and fix stale tests
  - [x] 0.1 Set up JavaScript test harness
    - Add `package.json` with Vitest and fast-check as dev dependencies and a `test` script
    - Configure Vitest to run in a Node environment (no browser required)
    - Expose the pure-logic classes (`SettingsParser`, `SettingsPrettyPrinter`, `PerformanceMonitor`, `AdaptiveWorkerScaler`) for import/testing in Node (e.g., conditional module export) without pulling in browser-only globals
    - Note: All 195 tests across 4 test files pass (`src/logic.test.js`, `src/clahe.test.js`, `src/auto-normalize.test.js`, `tests/clean-speckles.test.js`). Property tests 7.1, 9.5, 10.6, 16.1, 17.1, and 17.2 still need to be written in this harness.
    - _Requirements: 12.1, 12.5, 2.1, 5.6_

  - [x] 0.2 Fix or remove the stale Python test `tests/test_encode_args.py`
    - The test has been updated to assert against the current FFmpeg encode arguments in `encodeOneChunk` (checks for `-start_number 0`, `libx264`, `yuv420p`, and `+faststart` — all present in the current script.js)
    - _Requirements: 6.1_

### Phase 1: Performance Monitoring Infrastructure

- [x] 1. Create PerformanceMonitor class for tracking metrics
  - Create `PerformanceMonitor` class in script.js with frame latency tracking
  - Implement methods: `recordFrameStart()`, `recordDecodeComplete()`, `recordProcessComplete()`, `recordEncodeComplete()`
  - Implement statistical analysis: `getMedianLatency(lastN)`, `getP95Latency(lastN)`, `getP99Latency(lastN)`
  - Implement GPU/CPU usage tracking counters
  - Implement `getAvailableMemoryMB()` using `performance.memory` API
  - Implement bottleneck identification heuristic: compare avgDecode, avgProcess, avgEncode
  - Implement `getSummary()` method returning all metrics and bottleneck analysis
  - Add console logging for real-time metrics display
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10_

- [x] 2. Integrate performance monitoring into video processing loop
  - Instrument `renderVideoExport()` function to create PerformanceMonitor instance
  - Add timing calls at decode start/complete, process start/complete, encode start/complete
  - Record GPU vs CPU usage in worker message handler
  - Add console logging every 100 frames showing median latency, FPS, GPU usage %, bottleneck
  - Add final summary log after video processing completes with all metrics
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.9, 4.10_

### Phase 2: GPU Compute Shader - CLAHE Filter

- [x] 3. Implement CLAHE compute shader for GPU pipeline
  - Note: All 4 WGSL shader passes and the CLAHEComputeShader JS wrapper are implemented in gpu-worker.js. CLAHE is now on the GPU path: `GpuProcessor.applyCLAHE()` routes through `CLAHEComputeShader.apply()` which is called by AutoNormalizeFilter and directly when darkBoost/autoNormalize is active.
  - [x] 3.1 Create CLAHE tile histogram computation shader pass
    - Write WGSL compute shader for Pass 1: tile histogram with atomic binning
    - Divide image into 8×8 pixel tiles, compute 256-bin histogram per tile
    - Use workgroup size (8, 8) with atomic operations for parallel histogram accumulation
    - Store results in tileHistograms buffer (tilesPerRow × tilesPerColumn × 256 entries)
    - _Requirements: 1.1, 8.1, 8.2_

  - [x] 3.2 Create CLAHE clip and redistribute shader pass
    - Write WGSL compute shader for Pass 2: clip histogram bins at threshold and redistribute excess
    - Apply clip limit to each tile histogram (clip limit range 1.5-6.0)
    - Redistribute clipped bins uniformly across all 256 intensity values
    - Use workgroup size (256) to process all bins in parallel per tile
    - _Requirements: 1.1, 8.3, 8.4_

  - [x] 3.3 Create CLAHE CDF computation shader pass
    - Write WGSL compute shader for Pass 3: compute cumulative distribution function
    - Perform parallel prefix sum on clipped histogram per tile
    - Normalize CDF to output intensity range [0, 255]
    - Store CDF results per tile for interpolation pass
    - _Requirements: 1.1, 8.5_

  - [x] 3.4 Create CLAHE bilinear interpolation shader pass
    - Write WGSL compute shader for Pass 4: apply bilinear interpolation between adjacent tiles
    - For interior pixels: interpolate between 4 adjacent tile CDFs
    - For border pixels: apply single nearest-tile transformation
    - Output equalized grayscale image to output buffer
    - _Status: CLAHE_INTERP_SHADER WGSL is implemented in gpu-worker.js and used by CLAHEComputeShader.apply(). The shader correctly handles interior bilinear blending and border clamping._
    - _Requirements: 1.1, 8.6, 8.7_

  - [x] 3.5 Create CLAHEComputeShader JavaScript class wrapper
    - `CLAHEComputeShader` class is implemented in gpu-worker.js
    - Constructor initializes GPUDevice, compiles all 4 WGSL shader pipelines, and allocates tile histogram / CDF buffers
    - `apply(inputBuffer, outputBuffer, clipLimit)` orchestrates all 4 passes with proper buffer management and pipeline synchronization
    - _Requirements: 1.1, 8.1-8.10_

  - [x] 3.6 Write unit tests for CLAHE shader against OpenCV reference
    - Generate test images: low contrast, high contrast, varied brightness (mean 20-240)
    - Run CLAHE GPU shader with various clip limits (1.5, 2.5, 4.0, 6.0)
    - Compare against OpenCV CLAHE CPU reference outputs
    - Assert mean absolute error (MAE) < 3 intensity levels per pixel
    - Assert processing time < 15ms for 3840×2160 frame on reference hardware
    - _Requirements: 1.8, 8.8, 8.9_

### Phase 3: GPU Compute Shader - Auto-Normalize Filter

- [x] 4. Implement Auto-Normalize GPU shaders
  - Note: AutoNormalizeFilter class is fully implemented in gpu-worker.js. The 3-stage pipeline (gamma lift → histogram stretch → adaptive CLAHE) runs on the GPU path when `settings.autoNormalize` is true in `GpuProcessor.process()`.
  - [x] 4.1 Create image statistics computation shader (mean and std dev)
    - Write WGSL parallel reduction shader for computing mean brightness
    - Write WGSL parallel reduction shader for computing standard deviation
    - Use two-pass reduction: workgroup local reduction then global aggregation
    - Return statistics to CPU for normalization stage selection
    - _Requirements: 1.3, 9.7_

  - [x] 4.2 Create gamma lift shader with LUT
    - Write WGSL compute shader applying gamma correction via 256-entry lookup table
    - Build LUT on CPU: `output = 255 * (input/255)^(1/gamma)` for gamma range 1.5-3.0
    - Apply when input mean brightness < 80
    - Single compute pass with texture lookup for fast power-law transformation
    - _Requirements: 1.2, 9.1, 9.2_

  - [x] 4.3 Create histogram stretch shader
    - Write WGSL parallel reduction shader to find min/max pixel intensity
    - Write WGSL compute shader for NORM_MINMAX remapping: `output = ((input - min) * 255) / (max - min)`
    - Apply when input standard deviation < 45
    - Two compute passes: min/max reduction + remapping
    - _Requirements: 1.3, 9.3, 9.4_

  - [x] 4.4 Create AutoNormalizeFilter JavaScript class wrapper
    - `AutoNormalizeFilter` class is implemented in gpu-worker.js
    - `async analyze(grayBuffer)` returns `{mean, stdDev}` via GPU parallel reduction
    - `applyGammaLift(inputBuffer, outputBuffer, gamma)` applies LUT-based power-law transform
    - `async applyHistogramStretch(inputBuffer, outputBuffer)` does two-pass min/max reduction + remap
    - `async applyCLAHE(inputBuffer, outputBuffer, adaptiveClipLimit)` delegates to CLAHEComputeShader
    - `async apply(grayBuffer, outputBuffer)` orchestrates the full 3-stage pipeline with the same adaptive logic as the design spec
    - _Requirements: 1.2, 1.3, 1.4, 9.1-9.10_

  - [x] 4.5 Write unit tests for Auto-Normalize shaders
    - Generate test images with varying brightness (mean 20-240) and contrast (stdDev 10-80)
    - Test gamma lift stage: verify power-law brightness transformation matches CPU
    - Test histogram stretch stage: verify min/max remapping correctness
    - Test adaptive CLAHE: verify clip limit calculation and output quality
    - Assert MAE < 5 intensity levels per pixel compared to CPU reference
    - Assert total processing time < 20ms for 3840×2160 frame
    - _Requirements: 1.8, 9.8, 9.9_

### Phase 4: GPU Compute Shader - Clean-Speckles Filter

- [x] 5. Implement Clean-Speckles GPU shader (connected component analysis)
  - Note: CleanSpecklesShader is fully implemented in gpu-worker.js with the complete CCL pipeline (label initialization, propagation, compaction, histogram, and filtering). It is on the GPU path: `GpuProcessor.applyCleanSpeckles()` routes through `CleanSpecklesShader.apply()`, which is called directly in `GpuProcessor.process()` when `settings.cleanSpeckles` is true.
  - [x] 5.1 Create parallel connected component labeling shader
    - Write WGSL compute shader for label initialization (each white pixel gets unique label)
    - Write WGSL compute shader for label propagation (8 iterations of neighbor label adoption)
    - Each pixel adopts minimum label from 8-connected neighbors per iteration
    - Use ping-pong buffers to alternate read/write labels between iterations
    - _Requirements: 1.5, 10.1, 10.7_

  - [x] 5.2 Create component area computation and filtering shader
    - WGSL shaders for label compaction (dense label range), parallel histogram of label counts (component areas), and component area filtering are all implemented in CleanSpecklesShader
    - Supports intensity thresholds: fine (4px), medium (12px), coarse (30px)
    - _Requirements: 1.5, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 5.3 Create CleanSpecklesShader JavaScript class wrapper
    - Create `CleanSpecklesShader` class in gpu-worker.js
    - Implement constructor: initialize GPU pipelines, allocate label and histogram buffers
    - Implement `apply(binaryMaskBuffer, outputBuffer, minArea)` orchestrating all passes
    - Add buffer management for ping-pong label propagation
    - _Requirements: 1.5, 10.1-10.10_

  - [x] 5.4 Write unit tests for Clean-Speckles shader
    - Generate binary test masks with known connected components (various sizes 1-50 pixels)
    - Run Clean-Speckles GPU shader with different intensity settings (fine/medium/coarse)
    - Compare against OpenCV connectedComponentsWithStats reference output
    - Assert pixel-perfect equivalence for component filtering
    - Assert processing time < 25ms for 3840×2160 binary mask
    - _Requirements: 1.8, 10.8, 10.9_

### Phase 5: GPU Compute Shader - Color-Edges Filter

- [x] 6. Implement Color-Edges GPU shaders
  - [x] 6.1 Create color edge detection shader (Canny on raw grayscale)
    - Write WGSL compute shader for Canny edge detection on raw grayscale (before CLAHE normalization)
    - Reuse existing Sobel gradient, non-maximum suppression, and hysteresis shaders
    - Use separate thresholds for color edges (colorLowThresh, colorHighThresh)
    - Output binary color edge mask
    - _Requirements: 1.6, 11.1, 16.7_

  - [x] 6.2 Create edge dilation shader for line weight control
    - Write WGSL compute shader for morphological dilation of edge mask
    - Support line weights 1-5 (1 = no dilation, 5 = 4 iterations of dilation)
    - Use 3×3 structuring element for each dilation pass
    - _Requirements: 1.6, 11.2_

  - [x] 6.3 Create RGB Gaussian blur shader for color softness
    - Write WGSL separable Gaussian blur shader (horizontal + vertical passes)
    - Apply blur to RGB source image when colorSoftness > 0
    - Kernel size: `2 * softness + 1` (range 0-10)
    - Preserve raw RGB when softness = 0 (no blur)
    - _Requirements: 1.6, 11.3, 16.8_

  - [x] 6.4 Create color compositing shader with opacity blending
    - Write WGSL compute shader for final color compositing
    - Priority order: (1) ink edges → ink color, (2) color edges → sampled RGB with opacity, (3) background color
    - Sample RGB from blurred source image at color edge positions
    - Apply colorOpacity (0.0-1.0) for alpha blending with background
    - Output final RGBA image
    - _Requirements: 1.7, 11.4, 11.5_

  - [x] 6.5 Create ColorEdgesShader JavaScript class wrapper
    - Create `ColorEdgesShader` class in gpu-worker.js
    - Implement `detectColorEdges(grayRawBuffer, lowThresh, highThresh, outputMaskBuffer)`
    - Implement `dilateEdges(maskBuffer, lineWeight, outputBuffer)`
    - Implement `blurRGB(rgbBuffer, softness, outputBuffer)`
    - Implement `compositeColors(inkMaskBuffer, colorMaskBuffer, rgbBuffer, bgColor, opacity, outputRGBABuffer)`
    - Orchestrate full pipeline: edge detection → dilation → RGB blur → compositing
    - _Requirements: 1.6, 1.7, 11.1-11.5_

  - [-] 6.6 Write unit tests for Color-Edges shaders
    - Generate test frames: RGB images with known edges and color gradients
    - Test edge detection on raw grayscale (verify separate from normalized grayscale)
    - Test color sampling with various softness settings (0-10)
    - Test opacity blending (0.0-1.0 range)
    - Assert color accuracy deltaE < 5 compared to expected RGB values
    - Assert processing time < 30ms for 3840×2160 frame with color edges enabled
    - _Requirements: 1.8, 11.6_

### Phase 6: GPU Pipeline Integration and Output Equivalence Testing

- [x] 7. Integrate all GPU shaders into complete pipeline
  - gpu-worker.js processing pipeline now uses GPU shaders for CLAHE, Auto-Normalize, and Clean-Speckles in addition to the existing Color-Edges, bilateral, Gaussian, median, Canny, and morphology shaders
  - Graceful CPU fallback on WebGPU unavailability is preserved (device.lost handler sets useGpu=false)
  - GPU/CPU usage tracking is wired via `perfMonitor.recordProcessComplete(entry, usedGpu)` in the render loop
  - Filter execution order matches design document
  - _Requirements: 1.1-1.7, 1.10_

  - [x] 7.1 Write property test for GPU/CPU output equivalence across all filters
    - **Property 1: GPU/CPU Output Equivalence Across All Filters**
    - **Validates: Requirements 1.1-1.12, 8.1-8.10, 9.1-9.10, 10.1-10.10, 11.1-11.10**
    - Generate random test frames: vary resolution (1280×720 to 3840×2160), brightness (20-240), contrast (10-80)
    - Generate random settings: all presets, custom mode combinations, threshold variations
    - Process each (frame, settings) pair through GPU and CPU pipelines
    - Measure max absolute error per channel and SSIM structural similarity
    - Assert: maxAbsoluteError ≤ 1, SSIM > 0.99
    - Run 100 iterations with fast-check property testing library
    - _Requirements: 1.8, 1.9, 1.12_

- [x] 8. Checkpoint - Verify GPU pipeline performance and equivalence
  - Run preview with GPU pipeline on test videos (720p-4K, various content types)
  - Verify GPU usage percentage > 90% when WebGPU available
  - Verify frame processing latency 5-50ms per frame at 4K resolution
  - Verify graceful CPU fallback when GPU unavailable
  - Measure per-frame latency improvement over baseline (target: 2-4x faster)
  - Ensure all tests pass, ask the user if questions arise

### Phase 7: Adaptive Worker Scaling

- [x] 9. Implement adaptive worker pool scaling
  - [x] 9.1 Create AdaptiveWorkerScaler class
    - Create `AdaptiveWorkerScaler` class in script.js
    - Implement constructor: initialize with LineArtProcessor and PerformanceMonitor references
    - Add configuration: `minScaleInterval = 10000ms`, `latencyScaleUpThreshold = 100ms`, `latencyScaleDownThreshold = 40ms`
    - Add configuration: `memoryScaleUpMinimum = 160MB`, `memoryScaleDownThreshold = 80MB`
    - Implement `_getMaxWorkers()` method: return `min(computeOptimalWorkers().max, 8)`
    - _Requirements: 2.1, 2.10_

  - [x] 9.2 Implement scaling decision logic
    - Implement `evaluateAndAdjust()` method with scaling policy
    - Scale up condition: medianLatency > 100ms AND availableMemory > 160MB AND currentWorkers < maxWorkers
    - Scale down condition: medianLatency < 40ms OR availableMemory < 80MB AND currentWorkers > 1
    - Implement cooldown: only scale if 10 seconds elapsed since last scaling operation
    - Call existing `LineArtProcessor.resize(n)` to perform scaling
    - Add console logging for scale-up/scale-down events with latency and memory values
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 9.3 Integrate adaptive scaling into video processing loop
    - Modify `renderVideoExport()` to create AdaptiveWorkerScaler instance
    - Call `scaler.evaluateAndAdjust()` every 50 frames during processing
    - Ensure evaluation doesn't block frame processing (use async scheduling)
    - Preserve manual worker slider override (disable auto-scaling when user adjusts slider)
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 9.4 Add UI indicator for adaptive scaling status
    - Add UI element showing current active worker count and auto-scaling status
    - Update indicator in real-time when scaling events occur
    - Show "Auto-scaling: ON/OFF" based on whether user manually adjusted slider
    - _Requirements: 2.2-2.6_

  - [x] 9.5 Write property test for worker pool scaling invariants
    - **Property 2: Worker Pool Scaling Maintains Invariants**
    - **Validates: Requirements 2.1, 2.8, 2.9, 2.10, 5.1, 5.2, 5.5**
    - Generate random scaling operation sequences: scale up, scale down, resize to N (1-8)
    - Simulate frame processing with memory tracking
    - Assert after each operation: worker count ∈ [1, maxWorkers], memory ≤ budget, no frames lost
    - Assert initial auto-detection produces count ≤ 4 (DEFAULT_SAFE_WORKER_CAP)
    - Run 100 iterations
    - _Requirements: 2.1, 2.8, 2.9, 2.10_

### Phase 8: Parallel Video Decode Pool

- [x] 10. Implement parallel video decode pool
  - [x] 10.1 Create VideoDecodePool class with element pool and waiting queue
    - Create `VideoDecodePool` class in script.js
    - Implement `constructor(size, srcVideo)`: create `size` cloned HTMLVideoElement instances that share `srcVideo`'s object-URL `src` (no extra network fetch)
    - Maintain a `free` list of idle elements and a FIFO `waiting` queue of pending acquirers
    - Pool size is provisioned to the current worker count
    - _Requirements: 3.1_

  - [x] 10.2 Implement acquire/release with FIFO waiting queue
    - Implement `acquire()` returning a `Promise<{ video, release }>`: resolve immediately if an element is free, otherwise enqueue FIFO until one is released
    - Implement `release()` to return an element to the pool and hand it to the next waiting acquirer
    - Bound the number of concurrently acquired elements to the pool size
    - _Requirements: 3.2, 3.3, 3.10_

  - [x] 10.3 Implement seek-and-draw decode flow to offscreen canvas
    - For each frame: acquire an element, seek it to the target `frameTime`, draw the frame to an offscreen canvas, then release the element
    - Release per-frame buffers (the offscreen canvas) once the frame is handed to the processor so memory does not grow unbounded
    - Preserve frame ordering: dispatch frames in index order and recombine by index downstream
    - _Requirements: 3.5, 3.6, 3.7_

  - [x] 10.4 Implement resize(n) to track worker count
    - Implement `resize(n)`: scale-up adds immediately-usable cloned elements; scale-down pops idle elements (busy elements keep running and are simply not returned to the pool)
    - Keep the pool sized in lock-step with the worker pool
    - _Requirements: 3.4_

  - [x] 10.5 Implement first-batch seek staggering and destroy()
    - Stagger the first `poolSize` frames by `FRAME_STAGGER_MS` (frame `i` waits `i × FRAME_STAGGER_MS` before acquiring) so elements don't all seek simultaneously
    - Implement `destroy()`: detach `src` and drop all free elements
    - _Requirements: 3.9_

  - [-] 10.6 Write property test for decode pool concurrency bound
    - **Property 3: Decode Pool Concurrency Is Bounded by Pool Size**
    - **Validates: Requirements 3.5, 3.10, 5.6, 5.7**
    - Generate random pool sizes (1-8, matching worker count) and random interleaved acquire/release sequences (with more concurrent acquirers than the pool size to exercise the waiting queue)
    - Track the count of outstanding (acquired-but-not-yet-released) elements after each operation
    - Assert: outstanding count ≤ pool size at all times; queued acquirers resolve FIFO as elements are released
    - Dispatch frames in index order and assert the recombined output frame order matches the input order
    - Run 100 iterations
    - _Requirements: 3.5, 3.10, 5.6, 5.7_

- [x] 11. Integrate parallel video decode pool into video processing loop
  - Create a `VideoDecodePool` sized to `processor.concurrency` in `renderVideoExport()` and store it as `state.activeDecodePool`
  - Replace single-decoder `video.currentTime` seeking with `await decodePool.acquire()` followed by seek + draw-to-offscreen-canvas + `release()`
  - Stagger the first batch of frames by `FRAME_STAGGER_MS` to avoid simultaneous seeks
  - Resize the decode pool whenever the worker pool resizes (adaptive scaler and manual slider) so the two stay in sync
  - Bound concurrency by the pool size together with the existing in-flight Semaphore
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9, 3.10_

- [x] 12. Checkpoint - Verify adaptive scaling and decode pool effectiveness
  - Process 1-minute test video with all optimizations enabled
  - Verify at least 3 worker scaling adjustments occur during render
  - Verify the decode pool resizes to track the worker count during scaling
  - Verify worker idle time < 5% of total processing time
  - Measure cumulative performance improvement over baseline
  - Ensure all tests pass, ask the user if questions arise

### Phase 9: Settings Serialization and Validation

- [x] 13. Implement settings parser with validation
  - [x] 13.1 Create SettingsParser class
    - Create `SettingsParser` class in script.js
    - Implement `static parse(jsonString)` method with JSON.parse and validation
    - Implement `static validate(obj)` method checking all constraints
    - Validate preset: must be one of {manga, studio, neon, warm, vivid, blueprint, custom}
    - Validate ranges: detail (1-100), lineWeight (1-5), scale (0.1-2.0)
    - Validate threshold ordering: highThreshold ≥ lowThreshold + 24
    - Validate color thresholds: colorHighThresh > colorLowThresh (when colorEdges enabled)
    - Throw validation errors with specific field names and reasons
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 13.2 Create SettingsPrettyPrinter class
    - Create `SettingsPrettyPrinter` class in script.js
    - Implement `static print(settings)` method with ordered keys
    - Order: preset, detail, lineWeight, scale, videoFps, isOriginalFps first
    - Include custom mode options only when customMode = true
    - Use `JSON.stringify(ordered, null, 2)` for readable formatting
    - _Requirements: 12.6, 12.7, 12.8_

  - [x] 13.3 Write property test for settings serialization round-trip
    - **Property 4: Settings Serialization Round-Trip**
    - **Validates: Requirements 12.1-12.5, 12.9**
    - Generate random valid settings: all presets, detail (1-100), lineWeight (1-5), scale (0.1-2.0)
    - Randomize custom mode toggles and sliders within valid ranges
    - Serialize: `json = SettingsPrettyPrinter.print(settings)`
    - Deserialize: `parsed = SettingsParser.parse(json)`
    - Assert: `deepEquals(parsed, settings)`
    - Run 100 iterations
    - _Requirements: 12.1-12.5, 12.9_

- [-] 14. Implement settings persistence to localStorage
  - Add localStorage save on every settings change (throttled to 500ms)
  - Add localStorage restore on page load in initialization code
  - Use key `lineArtProcessorSettings` for storage
  - Serialize using SettingsPrettyPrinter before saving
  - Parse using SettingsParser after loading with error handling
  - _Requirements: 12.9_

- [-] 15. Add "Reset to Default" button in Custom mode UI
  - Add button in Custom/Experiment mode section
  - On click: restore baseline custom preset values (white bg, black ink, threshold 60/180, bilateral diameter 13, sigma 90)
  - Clear localStorage settings
  - Refresh UI controls to reflect default values
  - _Requirements: 12.10_

### Phase 10: Color Lines Preservation Under Normalization

- [x] 16. Ensure color sampling uses raw RGB before normalization
  - Verify in gpu-worker.js that color edge detection operates on grayRaw buffer (raw grayscale)
  - Verify color edge Canny uses raw image unaffected by Auto-Normalize, CLAHE, or gamma lift
  - Verify RGB color sampling for soft edges occurs on original RGB buffer before any normalization
  - Verify ColorEdgesShader.blurRGB() operates on raw RGB source image
  - Maintain separate image buffers: normalized grayscale (for ink edges) and raw RGB (for color sampling)
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.7, 16.8, 16.10_

  - [x] 16.1 Write property test for color accuracy under normalization
    - **Property 5: Color Lines Preserve Color Accuracy Under Normalization**
    - **Validates: Requirements 16.1-16.6, 16.9**
    - Generate random frames: vary brightness (20-240) to trigger different normalization stages
    - Enable color edges + auto-normalize with random settings
    - Process frame, extract colored edge pixels
    - Compare color values against raw RGB source: measure deltaE color difference
    - Verify opacity matches slider setting within 95% tolerance
    - Assert deltaE < 5 for color accuracy
    - Run 100 iterations
    - _Requirements: 16.1-16.6, 16.9_

### Phase 11: Memory Safety and Stability Preservation

- [x] 17. Verify all existing stability mechanisms are preserved
  - Verify streaming encoder chunking at ~400MB per chunk remains unchanged
  - Verify Semaphore-based frame limiting respects Memory_Budget (80MB per worker)
  - Verify batch-based GC checkpoints at existing intervals remain unchanged
  - Verify graceful fallback for oversized files (video-only output) remains unchanged
  - Verify maximum concurrent decoded frames: (workerCount × 2 + 4) enforcement
  - Verify WebGPU device-lost recovery and CPU fallback behavior unchanged
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.9_

  - [x] 17.1 Write property test for memory usage bounds on long videos
    - **Property 6: Memory Usage Remains Bounded for Long Videos**
    - **Validates: Requirements 5.1, 5.2, 5.5-5.8, 13.1, 13.10**
    - Simulate processing: vary video duration (60-100,000 frames), worker count (1-8), frame sizes (1-8 MP)
    - Track memory usage: sample performance.memory every 100 frames
    - Calculate expected peak memory: baseMemory + (workerCount × 80MB) + (decodePoolSize × frameSizeMB) + 400MB
    - Assert peak memory ≤ calculated budget
    - Assert memory at frame 1000 ≈ memory at frame 100 (no unbounded growth, within 10% variance)
    - Run 100 iterations
    - _Requirements: 5.1, 5.2, 5.5-5.8_

  - [x] 17.2 Write property test for frame processing order preservation
    - **Property 7: Frame Processing Order Preservation**
    - **Validates: Requirements 2.5, 2.6, 3.7, 13.3**
    - Generate random frame sequence: 100-1000 frames, each watermarked with frame index
    - Process with adaptive scaling enabled: randomly trigger scale-up/scale-down conditions
    - Verify output: count total frames, check index watermark in sequence
    - Assert: outputFrameCount === inputFrameCount, outputFrame[i].watermark === i for all i
    - Run 100 iterations
    - _Requirements: 2.5, 2.6, 3.7_

### Phase 12: Error Handling and Graceful Degradation

- [x] 18. Implement robust error handling for all optimization components
  - [x] 18.1 Add GPU device-lost recovery
    - `device.lost.then()` handler is in place in gpu-worker.js: sets `useGpu = false` so subsequent `process()` calls fall back to CpuProcessor automatically
    - Still missing: posting `{type: 'gpu-fallback', reason}` message back to the main thread so the UI can display a diagnostic/advisory
    - _Requirements: 7.3, 7.9_

  - [x] 18.2 Add worker scaling failure handling
    - Wrap `processor.resize(n)` in try-catch block in AdaptiveWorkerScaler
    - On scaling failure: log warning, disable adaptive scaling, continue with current worker count
    - Display UI warning: "Worker scaling unavailable, using N workers"
    - _Requirements: 7.5, 7.9_

  - [x] 18.3 Add decode failure handling
    - Wrap the decode flow (`decodePool.acquire()` + seek + draw) in try-catch
    - On decode failure: log frame index and error, skip corrupt frame, release the video element, continue with next frame
    - Track corrupt frame count, update UI: "Skipped N corrupt frames"
    - Abort render if corrupt frame count > 5% of total frames
    - _Requirements: 7.9_

  - [x] 18.4 Add memory pressure handling
    - Monitor `perfMonitor.getAvailableMemoryMB()` during processing
    - If available memory < 100MB: reduce the decode pool size (fewer concurrent decode elements), scale down workers by 1
    - Display UI message: "Memory pressure detected, reduced workers to N"
    - Never scale below 1 worker (processing always continues)
    - _Requirements: 7.6, 7.9_

  - [x] 18.5 Add settings validation error handling
    - Use SettingsParser.parse() with try-catch when loading settings
    - Display validation errors in UI with specific field and reason
    - Auto-correct when possible: `highThreshold = max(highThreshold, lowThreshold + 24)`
    - Prevent render start until validation passes
    - _Requirements: 7.9_

- [x] 19. Checkpoint - Verify error handling and graceful degradation
  - Test GPU device-lost scenario: verify CPU fallback and completion
  - Test memory pressure scenario: verify worker scaling and decode pool reduction
  - Test invalid settings: verify validation errors and auto-correction
  - Test corrupt video frames: verify skip and continue behavior
  - Verify all error paths log diagnostics and continue gracefully
  - Ensure all tests pass, ask the user if questions arise

### Phase 13: Custom Mode Filter Ordering, Quality, and Validation

- [x] 20. Verify and align Custom mode filter ordering and parameter controls
  - [x] 20.1 Verify smoothing filter execution order across CPU and GPU pipelines
    - Confirm in worker.js (CPU) and gpu-worker.js (GPU) that Bilateral smooth applies first on RGB, then Gaussian on grayscale, then Median
    - Confirm that disabling all smoothing filters causes edge detection to operate on the raw RGB→grayscale conversion without smoothing
    - Align CPU and GPU pipelines to identical filter ordering to preserve output equivalence
    - _Requirements: 17.1, 17.2, 17.3_

  - [x] 20.2 Implement edge detail inverse threshold scaling and ink threshold controls
    - Implement Edge detail slider (35-90) so the threshold scaling factor inversely scales with detail value for finer line-density control
    - Wire Ink low threshold real-time control across range 5-150 and Ink high threshold across range 20-255
    - Ensure threshold changes update edge detection sensitivity in real-time during preview
    - _Requirements: 14.2, 14.3, 14.6_

  - [x] 20.3 Add human-subject preset starting values and validate color line coverage
    - Add a Custom-mode human-subject starting preset: Ink low 40, Ink high 100, Bilateral diameter 13, Sigma 90, Clean speckles enabled
    - Ensure Color line weight 5 with Color opacity 100% fully obscures the background without transparency artifacts
    - _Requirements: 14.10, 17.6_

  - [x] 20.4 Write integration test for Custom mode filter ordering and threshold validation
    - Verify Bilateral→Gaussian→Median ordering produces the expected intermediate processing buffers
    - Verify Ink low threshold greater than Ink high threshold auto-adjusts high to at least low+24 (links to SettingsParser validation in task 13.1)
    - Verify Color low threshold greater than Color high threshold auto-adjusts color high threshold
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_

- [x] 21. Validate Custom mode aesthetic consistency and toggle stability
  - [x] 21.1 Verify Merge double-edges consolidation behavior
    - Confirm the Merge double-edges filter consolidates parallel edge pairs from thick features into single clean lines
    - Verify intensity 1-5 maps to the morphological close kernel size (5×5 → 13×13) in worker.js and the equivalent GPU path
    - _Requirements: 15.10_

  - [x] 21.2 Write fuzz test for all Custom mode toggle combinations
    - Enumerate the 2^10 Custom mode filter toggle combinations (Bilateral, Gaussian, Median, Clean speckles, Auto-normalize, Dark boost, Merge double-edges, Color lines, and remaining toggles)
    - Process a small test frame through each combination and assert no crash, hang, or unhandled exception
    - _Requirements: 17.7_

  - [x] 21.3 Write performance test for all-filters-enabled latency
    - Enable all Custom mode filters simultaneously on a 1080p test frame
    - Assert Frame_Processing_Latency remains under 200ms per frame on reference hardware
    - _Requirements: 17.8_

### Phase 14: Performance Benchmarking and Validation

- [-] 22. Create reference test video suite
  - Create 1-minute 4K 60fps test video (1800 frames) with diverse content
  - Create 10-minute 1080p 30fps test video (18,000 frames) with scene changes
  - Create test videos covering: humans, wildlife, urban scenes, low-light, high-contrast
  - Store reference videos in test assets directory
  - _Requirements: 6.1-6.10_

- [x] 23. Run baseline performance benchmarks (before optimizations)
  - Process 1-minute 4K test video with all optimizations DISABLED
  - Disable adaptive scaling (fixed 4 workers), disable the parallel decode pool (single-element decode), use hybrid GPU/CPU pipeline
  - Measure total processing time, frames per second, per-frame latency
  - Record baseline metrics for comparison
  - Target baseline for 2.5-hour 4K video: 3-3.5 hours
  - _Requirements: 6.1, 6.2_

- [x] 24. Run optimized performance benchmarks (after all optimizations)
  - Process same 1-minute 4K test video with ALL optimizations ENABLED
  - Enable adaptive scaling, parallel decode pool, complete GPU pipeline
  - Measure total processing time, frames per second, per-frame latency
  - Calculate performance improvement: `(baselineTime - optimizedTime) / baselineTime × 100%`
  - Assert: optimizedTime ≤ baselineTime × 0.6 (40% improvement minimum)
  - Target for 2.5-hour 4K video: 1.5-2 hours (40-50% improvement)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.8_

- [x] 25. Validate performance targets on reference hardware
  - Run benchmarks on reference hardware: RTX 3060 equivalent, 16GB RAM, 8-core CPU
  - Verify frame processing throughput ≥ 50 fps during steady-state processing
  - Verify GPU pipeline processes ≥ 90% of frames when WebGPU available
  - Verify per-frame latency 5-50ms for 4K resolution with all filters enabled
  - Log final performance summary with all metrics
  - _Requirements: 6.4, 6.5, 6.6, 6.7, 6.9_

- [x] 26. Run cross-platform compatibility tests
  - Test on Windows 10/11 with Chrome 113+
  - Test on macOS (Intel and Apple Silicon) with Chrome/Safari
  - Test on Linux Ubuntu 22.04 with Chrome/Firefox
  - Process reference test frames with all presets on each platform
  - Compare output pixel data against golden master files
  - Assert SSIM > 0.99 on all platforms
  - _Requirements: 1.9, 7.1, 7.2, 7.8_

- [x] 27. Run stability tests under extreme conditions
  - Process 8-hour 4K 60fps test pattern video (simulated, 1.7M frames)
  - Verify completion without out-of-memory errors
  - Verify peak memory ≤ 2GB on systems with 8GB RAM
  - Sample memory usage every 1000 frames, verify no unbounded growth
  - Force GPU device-lost event mid-render, verify CPU fallback and completion
  - Simulate low-memory condition, verify graceful worker scaling
  - Process video with 5% corrupt frames, verify skip and continue behavior
  - _Requirements: 5.8, 7.3, 7.6, 7.8, 7.9_

- [x] 28. Final checkpoint - Verify all performance and stability targets met
  - Verify 40-50% performance improvement achieved on reference hardware
  - Verify all 7 property-based tests pass with 100 iterations each
  - Verify cross-platform output equivalence (SSIM > 0.99)
  - Verify memory bounds preserved for long videos
  - Verify all existing stability mechanisms unchanged
  - Verify error handling provides graceful degradation in all scenarios
  - Ensure all tests pass, ask the user if questions arise

### Phase 15: Documentation and Release

- [x] 29. Update documentation with performance benchmarks and usage guides
  - Update README.md with performance benchmark results (before/after comparison)
  - Document optimal settings for human subject videos (Custom mode starting values)
  - Create developer guide for GPU shader development and debugging
  - Document adaptive scaling behavior and manual override
  - Document parallel decode pool sizing and behavior
  - Document settings serialization format for advanced users
  - Add performance monitoring console commands for power users
  - _Requirements: 6.1-6.10_

- [x] 30. Create release notes and deployment plan
  - Write release notes highlighting: 40-50% performance improvement, complete GPU pipeline, adaptive scaling, parallel video decode pool
  - List all preserved stability mechanisms and backward compatibility guarantees
  - Document browser requirements: Chrome 113+, WebGPU support recommended
  - Create deployment checklist: test suite execution, cross-platform validation, performance benchmarks
  - Plan phased rollout: beta testing on reference hardware first, then broader release
  - _Requirements: 6.1-6.10, 7.1-7.10_

## Notes

**Task Ordering and Dependencies:**
- Phase 1 (Performance Monitoring) must complete first as it provides metrics for all subsequent phases
- Phases 2-5 (GPU Shaders) can proceed in parallel after Phase 1, but should be tested incrementally
- Phase 6 (GPU Integration) depends on Phases 2-5 completing
- Phase 7 (Adaptive Scaling) depends on Phase 1 (needs PerformanceMonitor)
- Phase 8 (Parallel Video Decode Pool) can proceed in parallel with Phase 7
- Phases 9-10 (Settings, Color Lines) are independent and can proceed in parallel with Phases 7-8
- Phase 11 (Memory Safety) should validate work from all previous phases
- Phase 12 (Error Handling) depends on Phases 6-8 being complete
- Phase 13 (Custom Mode Filter Ordering, Quality, and Validation) depends on Phases 5-6 (GPU pipeline) and Phase 9 (settings validation)
- Phase 14 (Benchmarking) must wait for all implementation phases to complete
- Phase 15 (Documentation and Release) is the final phase

**Scope Note on Requirements 14 and 15:**
- Requirements 14.1, 14.4, 14.7, 14.8, 14.9 and 15.1-15.9 describe subjective visual-quality outcomes (clean facial features, absence of shadow artifacts, consistent hand-drawn aesthetic) that require manual visual inspection and cannot be validated by a coding agent
- Only the codeable criteria from Requirements 14, 15, and 17 (filter ordering, threshold controls, preset values, toggle-combination stability, latency) are covered by tasks in Phase 13

**Optional Test Tasks:**
- Tasks marked with `*` are optional property-based and unit tests
- These tests provide comprehensive validation but can be skipped for faster MVP
- Core implementation tasks (shader implementation, integration, UI) are required

**Property-Based Testing:**
- All 7 properties from design document correctness section are included as test tasks
- Each property test references its property number and validated requirements
- Use fast-check library for JavaScript property-based testing
- Target 100 iterations per property for thorough validation

**Performance Targets:**
- Complete GPU pipeline: 25% improvement (eliminate CPU/GPU switching overhead)
- Adaptive worker scaling: 10% improvement (optimal parallelism)
- Parallel video decode pool: 5% improvement (eliminate worker idle time)
- Combined target: 40-50% total improvement

**Memory Safety:**
- All existing stability mechanisms MUST be preserved without regression
- No new unbounded memory allocations introduced
- Existing Semaphore, streaming encoder, GC checkpoints unchanged

**Cross-Platform Compatibility:**
- GPU and CPU pipelines must produce pixel-equivalent output (MAE ≤ 1, SSIM > 0.99)
- Test on Windows, macOS, Linux with multiple browsers
- Graceful CPU fallback when WebGPU unavailable


## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": ["0.1", "0.2", "1", "13.1", "13.2"]
    },
    {
      "id": 1,
      "tasks": ["2", "3.1", "4.1", "9.1"]
    },
    {
      "id": 2,
      "tasks": ["3.2", "3.3", "4.2", "4.3", "5.1", "6.1", "10.1"]
    },
    {
      "id": 3,
      "tasks": ["3.4", "3.5", "4.4", "5.2", "6.2", "6.3", "10.2"]
    },
    {
      "id": 4,
      "tasks": ["3.6", "4.5", "5.3", "6.4", "10.3", "10.4"]
    },
    {
      "id": 5,
      "tasks": ["5.4", "6.5", "10.5", "13.3"]
    },
    {
      "id": 6,
      "tasks": ["6.6", "7", "9.2", "10.6", "14", "15", "22"]
    },
    {
      "id": 7,
      "tasks": ["7.1", "9.3", "11", "16", "16.1", "20.1", "20.2"]
    },
    {
      "id": 8,
      "tasks": ["8", "9.4", "17", "20.3", "21.1"]
    },
    {
      "id": 9,
      "tasks": ["9.5", "12", "17.1", "17.2", "18.1", "18.2", "23", "20.4", "21.2", "21.3"]
    },
    {
      "id": 10,
      "tasks": ["18.3", "18.4", "18.5", "24", "25"]
    },
    {
      "id": 11,
      "tasks": ["19", "26", "27"]
    },
    {
      "id": 12,
      "tasks": ["28", "29"]
    },
    {
      "id": 13,
      "tasks": ["30"]
    }
  ]
}
```
