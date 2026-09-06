# Design Document: Video Processing Performance Boost

## Overview

This design specifies the technical implementation for optimizing video processing performance in a browser-based line-art animation converter. The application currently processes videos using WebGPU compute shaders for GPU acceleration with OpenCV.js WASM as a CPU fallback. The optimization goal is to reduce processing time for long 4K videos by 40-50% (from 3-3.5 hours to 1.5-2 hours for a 2.5-hour 4K 60fps video) while preserving all existing stability mechanisms.

### Performance Target

**Current Performance:** 2.5-hour 4K 60fps video (540,000 frames) processes in 3-3.5 hours  
**Target Performance:** Same video processes in 1.5-2 hours (40-50% improvement)  
**Reference Hardware:** NVIDIA RTX 3060 (or equivalent), 16GB system RAM, 8-core CPU

### Optimization Strategy

The performance improvements will be achieved through three complementary approaches:

1. **Complete GPU Pipeline Implementation** (25% improvement): Implement missing GPU compute shaders for CLAHE, Auto-Normalize, Clean-Speckles, and Color-Edges filters to eliminate CPU/GPU context switching overhead
2. **Adaptive Worker Scaling** (10% improvement): Dynamically adjust worker pool size based on real-time performance metrics and system resource availability
3. **Parallel Video Decode Pool** (5% improvement): Decode multiple frames concurrently using a pool of cloned video elements (sized to the worker count) so workers never idle waiting on a single decoder to seek and draw one frame at a time

### Critical Constraints

All optimization work must preserve these existing stability mechanisms without regression:

- **Streaming encode with chunking** (~400MB chunks) for bounded memory usage
- **Parallel FFmpeg workers** for encoding throughput
- **Graceful fallback** for oversized files (video-only output when audio muxing exceeds WASM heap)
- **Memory-bounded processing** (Semaphore-based frame limiting, 80MB per worker budget)
- **Batch-based GC checkpoints** to prevent memory accumulation
- **WebGPU device-lost recovery** with transparent CPU fallback

### Implementation Status

This design has been reconciled against the implemented code in `script.js`. Current status:

- **PerformanceMonitor** — IMPLEMENTED and wired into the render loop.
- **VideoDecodePool** (the parallel decode component) — IMPLEMENTED and wired into the render loop; `state.activeDecodePool` tracks the active pool and is resized in step with the worker pool.
- **SettingsParser / SettingsPrettyPrinter** — IMPLEMENTED and wired in.
- **AdaptiveWorkerScaler** — IMPLEMENTED (with `evaluateAndAdjust()`, manual-override disable via `disableAutoScaling()`, and evaluation every 50 frames from the render loop). It is auto-disabled if the user manually adjusts the worker control.
- **GPU compute shaders for CLAHE, Auto-Normalize, and Clean-Speckles** — NOT yet on the GPU path; these filters currently fall back to the OpenCV CPU pipeline. **Color-Edges IS on the GPU path.** The shader designs in Section 1 describe the target GPU implementation and remain the plan of record.

## Architecture

### Current Architecture

The application uses a Web Worker pool architecture with hybrid GPU/CPU processing:

```
Main Thread
├── LineArtProcessor (worker pool manager)
├── Video Decoder (HTML5 video element)
├── FFmpeg.js (encoding/muxing)
└── UI Controls

Worker Pool (4 workers by default)
├── Worker 1: gpu-worker.js
│   ├── WebGPU Device (if available)
│   └── OpenCV.js WASM (fallback)
├── Worker 2: gpu-worker.js
├── Worker 3: gpu-worker.js
└── Worker 4: gpu-worker.js
```

### Proposed Architecture

The optimized architecture adds three new components:

```
Main Thread
├── LineArtProcessor (worker pool manager) [ENHANCED]
│   ├── Adaptive Worker Scaling Logic [NEW]
│   └── Dynamic Resize API
├── Video Decoder (HTML5 video element)
├── Parallel Video Decode Pool [NEW]
│   ├── Pool of cloned <video> elements (sized to worker count)
│   ├── acquire() / release() with FIFO waiting queue
│   └── resize(n) — kept in sync with the worker pool
├── Performance Monitor [NEW]
│   ├── Frame Processing Latency Tracker
│   ├── Worker Utilization Tracker
│   └── GPU/CPU Pipeline Usage Tracker
├── FFmpeg.js (encoding/muxing)
└── UI Controls

Worker Pool (1-8 workers, dynamically adjusted)
├── Worker N: gpu-worker.js [ENHANCED]
│   ├── WebGPU Device (if available)
│   │   ├── Complete GPU Pipeline [ENHANCED]
│   │   │   ├── CLAHE Compute Shader [PLANNED — CPU fallback today]
│   │   │   ├── Auto-Normalize Compute Shaders [PLANNED — CPU fallback today]
│   │   │   ├── Clean-Speckles Compute Shader [PLANNED — CPU fallback today]
│   │   │   └── Color-Edges Compute Shaders [IMPLEMENTED on GPU path]
│   │   └── Existing GPU Shaders (bilateral, Canny, morphology)
│   └── OpenCV.js WASM (fallback)
```

The decode component is a **parallel pool of cloned `HTMLVideoElement` instances**, not a look-ahead prefetch queue. The pool is sized to the current worker count, and each element shares the same object-URL `src` as the single source video, so no extra network fetch occurs. Callers `acquire()` an element, seek and draw the requested frame to an offscreen canvas, then `release()` the element back to the pool. Concurrency (and therefore decode memory) is bounded structurally by the pool size together with the existing in-flight `Semaphore`, and the pool is resized whenever the worker pool resizes.


## Components and Interfaces

### 1. GPU Compute Shader Implementations

#### 1.1 CLAHE (Contrast Limited Adaptive Histogram Equalization) Shader

**Purpose:** Apply per-tile histogram equalization with clip limiting to boost local contrast on GPU

**Interface:**
```javascript
class CLAHEComputeShader {
  constructor(device, width, height, clipLimit, tileSize);
  apply(inputBuffer, outputBuffer);  // Both GPUBuffer (CV_8UC1 equivalent)
}
```

**Algorithm:**
1. Divide grayscale image into 8×8 pixel tiles
2. Compute histogram for each tile (256 bins, parallel reduction)
3. Apply clip limit redistribution (redistribute excess above clip threshold uniformly)
4. Compute cumulative distribution function (CDF) per tile
5. Apply bilinear interpolation between adjacent tile CDFs for interior pixels
6. Apply single-tile transformation for border pixels

**WGSL Compute Passes:**
- Pass 1: Tile histogram computation (parallel atomic histogram binning)
- Pass 2: Clip and redistribute (parallel per-tile)
- Pass 3: CDF computation (parallel scan per tile)
- Pass 4: Bilinear interpolation output (parallel per pixel)

**Performance Target:** 15ms for 3840×2160 frame on RTX 3060

#### 1.2 Auto-Normalize Filter Shaders

**Purpose:** Adaptive brightness correction using gamma lift, histogram stretch, and CLAHE cascade

**Interface:**
```javascript
class AutoNormalizeFilter {
  constructor(device, width, height);
  async analyze(grayBuffer);  // Returns { mean, stdDev }
  applyGammaLift(inputBuffer, outputBuffer, gamma);
  applyHistogramStretch(inputBuffer, outputBuffer);
  applyCLAHE(inputBuffer, outputBuffer, adaptiveClipLimit);
}
```

**Stages:**

**Stage 1 - Gamma Lift (mean < 80):**
- Build 256-entry LUT: `output = 255 * (input/255)^(1/gamma)`
- Apply via texture lookup (single compute pass)
- Gamma range: 1.5 (mild lift) to 3.0 (near-black frames)

**Stage 2 - Histogram Stretch (std dev < 45):**
- Parallel reduction to find min/max intensity
- Remap: `output = ((input - min) * 255) / (max - min)`
- Two compute passes: reduction + remapping

**Stage 3 - Adaptive CLAHE (all frames):**
- Clip limit: `clamp(150 / mean, 1.5, 4.5)`
- Use CLAHE shader from 1.1

**Performance Target:** 20ms total for 3840×2160 frame (including statistics computation)

#### 1.3 Clean-Speckles Shader (Connected Component Analysis)

**Purpose:** Remove isolated edge fragments while preserving continuous lines

**Interface:**
```javascript
class CleanSpecklesShader {
  constructor(device, width, height, minArea);
  apply(binaryMaskBuffer, outputBuffer);  // CV_8UC1 equivalent
}
```

**Algorithm (Parallel Connected Component Labeling):**
1. Initialize each white pixel with unique label
2. Iterate label propagation (8 fixed iterations):
   - Each pixel adopts minimum label from 8-connected neighbors
3. Compact labels (parallel compaction to dense label range)
4. Compute component areas (parallel histogram of label counts)
5. Filter components: zero pixels belonging to components with area < minArea

**Area Thresholds:**
- Intensity 1 (fine): 4 pixels
- Intensity 2 (medium): 12 pixels
- Intensity 3 (coarse): 30 pixels

**Performance Target:** 25ms for 3840×2160 binary mask


#### 1.4 Color-Edges Filter Shaders

**Purpose:** Render soft colored lines using original image colors on GPU

**Interface:**
```javascript
class ColorEdgesShader {
  constructor(device, width, height);
  detectColorEdges(grayRawBuffer, lowThresh, highThresh, outputMaskBuffer);
  dilateEdges(maskBuffer, lineWeight, outputBuffer);
  blurRGB(rgbBuffer, softness, outputBuffer);
  compositeColors(inkMaskBuffer, colorMaskBuffer, rgbBuffer, 
                  bgColor, opacity, outputRGBABuffer);
}
```

**Processing Steps:**

**Step 1 - Color Edge Detection:**
- Apply Canny on raw grayscale (before CLAHE normalization)
- Use separate thresholds from ink edges
- Single compute pass (reuse existing Sobel/NMS/hysteresis shaders)

**Step 2 - Optional Softness Blur:**
- If colorSoftness > 0: apply Gaussian blur to RGB source
- Kernel size: `2 * softness + 1`
- Two separable passes (horizontal + vertical)

**Step 3 - Color Sampling & Compositing:**
- For each pixel:
  - If ink edge: output ink color (takes priority)
  - Else if color edge: sample RGB from blurred source, apply opacity blending
  - Else: output background color
- Single compute pass with texture sampling

**Performance Target:** 30ms total for 3840×2160 frame with color edges enabled

### 2. Adaptive Worker Scaling

#### 2.1 Worker Pool Manager Enhancement

**Current Implementation:**
- Fixed worker count (default 4, user-adjustable via slider)
- `LineArtProcessor.resize(n)` supports manual scaling
- Scale-up: spawn new workers immediately
- Scale-down: drain excess workers after current task

**Enhancement - Automatic Scaling Logic:**

```javascript
class AdaptiveWorkerScaler {
  constructor(processor, performanceMonitor) {
    this._processor = processor;
    this._perfMon = performanceMonitor;
    this._scalingEnabled = true;
    this._lastScaleTime = 0;
    this._minScaleInterval = 10000;  // 10 seconds between adjustments
  }

  // Permanently disable auto-scaling for this render. Called when the user
  // manually adjusts the worker slider so their explicit choice is honoured.
  disableAutoScaling() {
    this._scalingEnabled = false;
  }

  // Resize the worker pool AND the active decode pool together so the two
  // stay in sync (mirrors the manual worker-slider handlers).
  _applyResize(n) {
    this._processor.resize(n);
    if (state.activeDecodePool) state.activeDecodePool.resize(n);
  }

  evaluateAndAdjust() {
    if (!this._scalingEnabled) return;
    if (Date.now() - this._lastScaleTime < this._minScaleInterval) return;
    
    const medianLatency = this._perfMon.getMedianLatency(50);
    if (medianLatency <= 0) return;  // no frames recorded yet
    const memoryAvailable = this._perfMon.getAvailableMemoryMB();
    const currentWorkers = this._processor.concurrency;
    
    // Scale up: latency too high AND memory available AND room to grow.
    if (medianLatency > 100 && memoryAvailable > 160 && currentWorkers < this._getMaxWorkers()) {
      try {
        this._applyResize(currentWorkers + 1);
        this._lastScaleTime = Date.now();
        console.log(`[Adaptive] Scaled up to ${currentWorkers + 1} workers (latency ${medianLatency}ms)`);
      } catch (err) {
        this._scalingEnabled = false;  // self-disable on resize error
        console.warn('[Adaptive] Scale-up failed, disabling auto-scaling:', err);
      }
    }
    
    // Scale down: latency low OR memory pressure, and more than one worker.
    else if ((medianLatency < 40 || memoryAvailable < 80) && currentWorkers > 1) {
      try {
        this._applyResize(currentWorkers - 1);
        this._lastScaleTime = Date.now();
        console.log(`[Adaptive] Scaled down to ${currentWorkers - 1} workers (latency ${medianLatency}ms, mem ${memoryAvailable}MB)`);
      } catch (err) {
        this._scalingEnabled = false;  // self-disable on resize error
        console.warn('[Adaptive] Scale-down failed, disabling auto-scaling:', err);
      }
    }
  }
  
  _getMaxWorkers() {
    const { max } = computeOptimalWorkers();
    return Math.min(max, 8);  // Hard cap at 8 to prevent GPU saturation
  }
}
```

**Scaling Policy:**
- **Scale Up Condition:** Median latency (last 50 frames) > 100ms AND available memory > 160MB AND current count < max
- **Scale Down Condition:** Median latency < 40ms OR available memory < 80MB AND current count > 1
- **Cooldown Period:** 10 seconds between adjustments to prevent thrashing
- **Memory Budget:** 80MB per worker + 160MB headroom for scale-up
- **Pool sync:** each adjustment resizes the worker pool and the active `VideoDecodePool` together (`_applyResize`)
- **Manual override:** `disableAutoScaling()` is invoked when the user adjusts the worker control, and a resize error self-disables scaling for the remainder of the render
- **Evaluation cadence:** invoked every 50 frames from the render loop


### 3. Parallel Video Decode Pool

#### 3.1 Decode Pool Architecture

**Purpose:** Decode multiple frames concurrently across a pool of cloned video elements so workers never idle waiting on a single decoder to seek and draw one frame at a time.

**Design:** `VideoDecodePool` manages a pool of cloned `HTMLVideoElement` instances sized to the current worker count. Each element is created from the single source video and shares the same object-URL `src`, so no extra network fetch occurs. Callers `acquire()` an element (resolving to `{ video, release }`), seek to the target frame time, draw the frame to an offscreen canvas, then call `release()` to return the element to the pool. When all elements are busy, `acquire()` returns a promise that is queued in a FIFO waiting list and resolved as soon as an element is released.

There is **no look-ahead prefetch**, **no `frameIndex → frame` map**, **no `maxPrefetchDepth`**, and **no byte-budget (`maxMemoryMB`) accounting**. Memory is bounded *structurally*: the number of concurrently acquired video elements can never exceed the pool size (= worker count), and the existing in-flight `Semaphore` further bounds the number of frames in flight. Decode memory is therefore bounded by `poolSize × frameSize` without a separate byte budget.

**Interface:**
```javascript
class VideoDecodePool {
  // size: number of video elements (= worker count)
  // srcVideo: the single source <video>; clones share its object-URL src
  constructor(size, srcVideo);

  // Resolves to { video, release }. If an element is free it resolves
  // immediately; otherwise the request is queued FIFO until one is released.
  acquire();   // → Promise<{ video: HTMLVideoElement, release: () => void }>

  // Add or remove elements so the pool tracks the worker pool size.
  // Scale-up adds immediately-usable clones; scale-down pops idle elements
  // (busy elements keep running and are simply not returned to the pool).
  resize(n);

  // Tear down: detach src and drop all free elements.
  destroy();
}
```

**Decode flow (per frame, in the render loop):**
```javascript
const { video: vEl, release: releaseVideo } = await decodePool.acquire();
try {
  const offCanvas = document.createElement('canvas');
  await seekVideo(vEl, frameTime);                 // seek this element
  drawMediaToCanvas(vEl, offCanvas, scale, custom); // draw frame to offscreen canvas
} finally {
  releaseVideo();                                   // return element to the pool
}
// offCanvas is then handed to processor.renderToData(...) for GPU/CPU processing
```

**Key Design Decisions:**

- **Pool size = worker count.** The pool is provisioned at `processor.concurrency` and resized in lock-step with the worker pool (see Integration Point). This keeps decode parallelism matched to processing parallelism.
- **First-batch staggering.** To avoid multiple video elements seeking at the exact same instant (which can stall decode), the first `decodePoolSize` frames are staggered by `FRAME_STAGGER_MS` (30ms) — frame `i` waits `i × FRAME_STAGGER_MS` before acquiring. This honours the existing `FRAME_STAGGER_MS` constraint.
- **Structural memory bound.** Concurrency is capped by the pool size plus the in-flight `Semaphore`; there is no per-frame byte accounting and no prefetch buffer that could grow unbounded. Per-frame buffers (the offscreen canvas) are released as soon as the frame is handed to the processor.
- **Frame ordering.** Frames are dispatched in index order and recombined by index downstream, so parallel decode does not reorder the output sequence.

**Integration Point:**
```javascript
// In the render loop:
const decodePoolSize = processor.concurrency;
const decodePool = new VideoDecodePool(decodePoolSize, sourceVideo);
state.activeDecodePool = decodePool;

for (let i = 0; i < totalFrames; i++) {
  // Stagger the first batch so elements don't all seek simultaneously.
  if (i > 0 && i < decodePoolSize) {
    await new Promise(r => setTimeout(r, i * FRAME_STAGGER_MS));
  }

  await inFlightSem.acquire();        // existing Semaphore bounds frames in flight

  const { video: vEl, release } = await decodePool.acquire();
  // ... seek + draw + release as shown above, then process the frame
}

// When the worker pool resizes (adaptive scaler or manual slider),
// the decode pool is resized to match:
//   state.activeDecodePool.resize(newWorkerCount);
```


### 4. Performance Monitoring

#### 4.1 Performance Monitor Component

**Purpose:** Track detailed real-time metrics for bottleneck analysis and adaptive scaling

**Interface:**
```javascript
class PerformanceMonitor {
  constructor() {
    this._frameLatencies = [];  // { frameIndex, decodeMs, processMs, encodeMs, timestamp }
    this._workerUtilization = [];  // { timestamp, busyWorkers, totalWorkers }
    this._gpuUsageCount = 0;
    this._cpuUsageCount = 0;
    this._maxHistorySize = 100;
  }
  
  recordFrameStart(frameIndex) {
    return {
      frameIndex,
      startTime: performance.now(),
      decodeStart: performance.now()
    };
  }
  
  recordDecodeComplete(entry) {
    entry.decodeMs = performance.now() - entry.decodeStart;
    entry.processStart = performance.now();
  }
  
  recordProcessComplete(entry, usedGPU) {
    entry.processMs = performance.now() - entry.processStart;
    entry.encodeStart = performance.now();
    if (usedGPU) this._gpuUsageCount++;
    else this._cpuUsageCount++;
  }
  
  recordEncodeComplete(entry) {
    entry.encodeMs = performance.now() - entry.encodeStart;
    entry.totalMs = performance.now() - entry.startTime;
    
    this._frameLatencies.push(entry);
    if (this._frameLatencies.length > this._maxHistorySize) {
      this._frameLatencies.shift();
    }
  }
  
  getMedianLatency(lastN) {
    const recent = this._frameLatencies.slice(-lastN);
    if (recent.length === 0) return 0;
    
    const sorted = recent.map(e => e.processMs).sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }
  
  getP95Latency(lastN) {
    const recent = this._frameLatencies.slice(-lastN);
    if (recent.length === 0) return 0;
    
    const sorted = recent.map(e => e.processMs).sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * 0.95);
    return sorted[idx];
  }
  
  getGPUUsagePercent() {
    const total = this._gpuUsageCount + this._cpuUsageCount;
    return total === 0 ? 0 : (this._gpuUsageCount / total) * 100;
  }
  
  getAvailableMemoryMB() {
    if (typeof performance.memory === 'undefined') return 1000;  // Fallback
    const usedMB = performance.memory.usedJSHeapSize / (1024 * 1024);
    const limitMB = performance.memory.jsHeapSizeLimit / (1024 * 1024);
    return limitMB - usedMB;
  }
  
  getSummary() {
    const latencies = this._frameLatencies.map(e => e.processMs);
    const median = this.getMedianLatency(latencies.length);
    const p95 = this.getP95Latency(latencies.length);
    const avgFPS = 1000 / median;
    
    return {
      totalFrames: this._frameLatencies.length,
      medianLatencyMs: median.toFixed(1),
      p95LatencyMs: p95.toFixed(1),
      avgFPS: avgFPS.toFixed(1),
      gpuUsagePercent: this.getGPUUsagePercent().toFixed(1),
      bottleneck: this._identifyBottleneck()
    };
  }
  
  _identifyBottleneck() {
    const recent = this._frameLatencies.slice(-50);
    if (recent.length === 0) return 'unknown';
    
    const avgDecode = recent.reduce((s, e) => s + e.decodeMs, 0) / recent.length;
    const avgProcess = recent.reduce((s, e) => s + e.processMs, 0) / recent.length;
    const avgEncode = recent.reduce((s, e) => s + e.encodeMs, 0) / recent.length;
    
    const max = Math.max(avgDecode, avgProcess, avgEncode);
    if (max === avgDecode) return 'decode';
    if (max === avgProcess) return 'processing';
    return 'encode';
  }
  
  reset() {
    this._frameLatencies = [];
    this._workerUtilization = [];
    this._gpuUsageCount = 0;
    this._cpuUsageCount = 0;
  }
}
```

**Metrics Exposed:**
- **Frame Processing Latency:** Decode, process, encode times per frame
- **Median/P95/P99 Latencies:** Statistical distribution of processing times
- **Worker Utilization:** Percentage of time workers are busy vs idle
- **GPU Pipeline Usage:** Percentage of frames processed on GPU vs CPU
- **Memory Availability:** Real-time available heap memory
- **Bottleneck Analysis:** Identifies slowest stage (decode, process, encode)

**Console Logging:**
```javascript
// Log summary every 100 frames
if (frameIndex % 100 === 0) {
  const summary = perfMonitor.getSummary();
  console.log(`[Perf] Frame ${frameIndex}: ${summary.avgFPS} fps, ${summary.medianLatencyMs}ms median, ${summary.gpuUsagePercent}% GPU, bottleneck: ${summary.bottleneck}`);
}
```


### 5. Settings Parser and Serializer

#### 5.1 Settings Schema

**Purpose:** Enable testing and configuration persistence through JSON serialization

**Settings Object Structure:**
```typescript
interface ProcessingSettings {
  preset: PresetKey;
  detail: number;           // 1-100
  lineWeight: number;       // 1-5
  scale: number;            // 0.1-2.0
  videoFps: number;
  isOriginalFps: boolean;
  
  // Custom mode options
  customMode: boolean;
  useBilateral: boolean;
  bilateralPasses: number;  // 1-5
  useGaussian: boolean;
  gaussianPasses: number;   // 1-5
  useMedian: boolean;
  medianPasses: number;     // 1-3
  
  // Filter options
  cleanSpeckles: boolean;
  cleanSpecklesIntensity: number;  // 1-3
  autoNormalize: boolean;
  darkBoost: boolean;
  darkBoostClip: number;    // 1.0-6.0
  mergeDoubleEdge: boolean;
  mergeDoubleEdgeIntensity: number;  // 1-5
  
  // Color edges
  colorEdges: boolean;
  colorLowThresh: number;   // 5-150
  colorHighThresh: number;  // 20-255
  colorLineWeight: number;  // 1-5
  colorSoftness: number;    // 0-10
  colorOpacity: number;     // 0.0-1.0
}

type PresetKey = 'manga' | 'studio' | 'neon' | 'warm' | 'vivid' | 'blueprint' | 'custom';
```

#### 5.2 Parser Implementation

```javascript
class SettingsParser {
  static parse(jsonString) {
    try {
      const obj = JSON.parse(jsonString);
      return this.validate(obj);
    } catch (error) {
      throw new Error(`Invalid JSON: ${error.message}`);
    }
  }
  
  static validate(obj) {
    const errors = [];
    
    // Required fields
    if (!['manga', 'studio', 'neon', 'warm', 'vivid', 'blueprint', 'custom'].includes(obj.preset)) {
      errors.push('Invalid preset value');
    }
    if (!this._inRange(obj.detail, 1, 100)) errors.push('detail must be 1-100');
    if (!this._inRange(obj.lineWeight, 1, 5)) errors.push('lineWeight must be 1-5');
    if (!this._inRange(obj.scale, 0.1, 2.0)) errors.push('scale must be 0.1-2.0');
    
    // Custom mode thresholds
    if (obj.customMode) {
      const preset = obj.preset === 'custom' ? getCustomPreset() : STYLE_PRESETS[obj.preset];
      if (preset.lowThreshold >= preset.highThreshold - 24) {
        errors.push('highThreshold must be at least 24 units above lowThreshold');
      }
      if (obj.colorEdges && obj.colorLowThresh >= obj.colorHighThresh) {
        errors.push('colorHighThresh must be greater than colorLowThresh');
      }
    }
    
    if (errors.length > 0) {
      throw new Error(`Validation errors: ${errors.join(', ')}`);
    }
    
    return obj;
  }
  
  static _inRange(value, min, max) {
    return typeof value === 'number' && value >= min && value <= max;
  }
}
```

#### 5.3 Pretty Printer Implementation

```javascript
class SettingsPrettyPrinter {
  static print(settings) {
    // Order keys for readability
    const ordered = {
      preset: settings.preset,
      detail: settings.detail,
      lineWeight: settings.lineWeight,
      scale: settings.scale,
      videoFps: settings.videoFps,
      isOriginalFps: settings.isOriginalFps
    };
    
    // Add custom mode options if enabled
    if (settings.customMode) {
      Object.assign(ordered, {
        customMode: true,
        useBilateral: settings.useBilateral,
        bilateralPasses: settings.bilateralPasses,
        useGaussian: settings.useGaussian,
        gaussianPasses: settings.gaussianPasses,
        useMedian: settings.useMedian,
        medianPasses: settings.medianPasses,
        cleanSpeckles: settings.cleanSpeckles,
        cleanSpecklesIntensity: settings.cleanSpecklesIntensity,
        autoNormalize: settings.autoNormalize,
        darkBoost: settings.darkBoost,
        darkBoostClip: settings.darkBoostClip,
        mergeDoubleEdge: settings.mergeDoubleEdge,
        mergeDoubleEdgeIntensity: settings.mergeDoubleEdgeIntensity,
        colorEdges: settings.colorEdges,
        colorLowThresh: settings.colorLowThresh,
        colorHighThresh: settings.colorHighThresh,
        colorLineWeight: settings.colorLineWeight,
        colorSoftness: settings.colorSoftness,
        colorOpacity: settings.colorOpacity
      });
    }
    
    return JSON.stringify(ordered, null, 2);
  }
}
```


## Data Models

### Frame Processing Pipeline State

```typescript
interface FrameState {
  frameIndex: number;
  timestamp: number;
  
  // Decode phase
  decodeStartTime: number;
  decodeCompleteTime: number;
  imageData: ImageData;
  
  // Process phase
  processStartTime: number;
  processCompleteTime: number;
  usedGPU: boolean;
  outputData: Uint8ClampedArray;
  
  // Encode phase
  encodeStartTime: number;
  encodeCompleteTime: number;
  
  // Derived metrics
  decodeLatencyMs: number;
  processLatencyMs: number;
  encodeLatencyMs: number;
  totalLatencyMs: number;
}
```

### Worker Pool State

```typescript
interface WorkerPoolState {
  workers: Worker[];
  concurrency: number;
  freePool: number[];          // indices of idle workers
  waitQueue: Function[];       // resolve callbacks waiting for workers
  drainSet: Set<number>;       // workers to terminate after current task
  pending: Map<number, PendingTask>;
  
  // Adaptive scaling state
  lastScaleTime: number;
  scalingEnabled: boolean;
}

interface PendingTask {
  id: number;
  resolve: Function;
  reject: Function;
  startTime: number;
}
```

### Decode Pool State

```typescript
interface DecodePoolState {
  size: number;                  // number of video elements (= worker count)
  srcVideo: HTMLVideoElement;    // source video; clones share its object-URL src
  free: HTMLVideoElement[];      // idle, ready-to-acquire elements
  waiting: Function[];           // FIFO resolve callbacks awaiting a free element
}
```

Memory is bounded structurally by `size` (the number of concurrently acquired
elements) rather than by tracked byte usage, so there is no per-frame memory
accounting or decoded-frame cache.

### Performance Metrics

```typescript
interface PerformanceMetrics {
  frameLatencies: FrameLatency[];
  workerUtilization: WorkerUtilizationSample[];
  gpuUsageCount: number;
  cpuUsageCount: number;
}

interface FrameLatency {
  frameIndex: number;
  decodeMs: number;
  processMs: number;
  encodeMs: number;
  totalMs: number;
  timestamp: number;
}

interface WorkerUtilizationSample {
  timestamp: number;
  busyWorkers: number;
  totalWorkers: number;
}

interface PerformanceSummary {
  totalFrames: number;
  medianLatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  avgFPS: number;
  gpuUsagePercent: number;
  workerUtilization: number;
  bottleneck: 'decode' | 'processing' | 'encode' | 'unknown';
}
```

### GPU Pipeline Buffers

```typescript
interface GPUPipelineBuffers {
  // Input/output
  rgbaIn: GPUBuffer;      // Original RGBA input
  rgbaOut: GPUBuffer;     // Final RGBA output
  
  // Grayscale processing
  gray: GPUBuffer;        // Normalized grayscale (for ink edges)
  grayRaw: GPUBuffer;     // Raw grayscale (for color edges)
  
  // Smoothing buffers (ping-pong)
  smooth1: GPUBuffer;
  smooth2: GPUBuffer;
  
  // Canny edge detection
  magnitude: GPUBuffer;   // Gradient magnitude
  direction: GPUBuffer;   // Gradient direction
  suppressed: GPUBuffer;  // After NMS
  edgesA: GPUBuffer;      // Hysteresis ping-pong
  edgesB: GPUBuffer;
  
  // Morphology
  maskA: GPUBuffer;       // Ink edge mask ping-pong
  maskB: GPUBuffer;
  
  // Color edges (separate pipeline)
  edgesColorA: GPUBuffer;
  edgesColorB: GPUBuffer;
  maskColorA: GPUBuffer;
  maskColorB: GPUBuffer;
  
  // Read-back staging
  readBuf: GPUBuffer;     // MAP_READ staging buffer
}
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Reflection on Correctness Properties

After analyzing all acceptance criteria for property-based testing applicability, the following redundancies were identified:

**Eliminated Redundancies:**
- **Requirements 1.2-1.7** (individual GPU filter implementations) are subsumed by **Requirement 1.8** (comprehensive GPU/CPU equivalence). Testing overall equivalence validates all individual filters.
- **Requirement 1.12** (SSIM > 0.99) is redundant with **Requirement 1.8** (pixel-level equivalence). Both measure the same equivalence property using different metrics—we will validate both in a single property.
- **Requirements 8.1-8.10, 9.1-9.10, 10.1-10.10, 11.1-11.10** (individual algorithm implementation details) are subsumed by the comprehensive GPU/CPU equivalence property. Testing end-to-end equivalence validates all underlying algorithms.

**Properties Retained:**
The following properties provide unique validation value and will be implemented as property-based tests:

### Property 1: GPU/CPU Output Equivalence Across All Filters

*For any* valid video frame (varying sizes 720p-4K, brightness 0-255, contrast low/high, content types) and *for any* valid processing settings (all presets, custom mode combinations), processing the frame through GPU_Pipeline and CPU_Pipeline SHALL produce pixel-equivalent output with maximum absolute error ≤ 1 intensity level per channel AND structural similarity (SSIM) > 0.99.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.12, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10**

**Test Strategy:**
- Generate random frames: vary resolution (1280×720 to 3840×2160), mean brightness (20-240), std dev (10-80), content (gradients, edges, noise, realistic images)
- Generate random settings: cycle through all presets, randomize custom mode toggles (bilateral/gaussian/median combinations), randomize thresholds within valid ranges
- Process each (frame, settings) pair on both GPU and CPU pipelines
- Measure: maxAbsoluteError per channel, SSIM structural similarity
- Assert: maxAbsoluteError ≤ 1, SSIM > 0.99

**Generator Strategy:**
```
arbitraryFrame = {
  width: randomChoice([1280, 1920, 2560, 3840]),
  height: randomChoice([720, 1080, 1440, 2160]),
  pixels: generateImageWithProperties(meanBrightness, stdDev, contentType)
}

arbitrarySettings = {
  preset: randomChoice(allPresets),
  detail: randomInt(1, 100),
  lineWeight: randomInt(1, 5),
  customMode: randomBool(),
  ... (if customMode: randomize all custom toggles and sliders)
}
```

### Property 2: Worker Pool Scaling Maintains Invariants

*For any* sequence of worker pool scaling operations (scale up, scale down, resize to N), the worker pool SHALL maintain these invariants: (1) worker count always in range [1, maxWorkers], (2) memory budget never exceeded (peakMemory ≤ baseMemory + workerCount × 80MB), (3) no in-flight frames are lost, (4) DEFAULT_SAFE_WORKER_CAP of 4 is respected for auto-detection.

**Validates: Requirements 2.1, 2.8, 2.9, 2.10, 5.1, 5.2, 5.5**

**Test Strategy:**
- Generate random sequence of scaling commands: scale up by 1, scale down by 1, resize to random N (1-8)
- Simulate frame processing: track in-flight frame count, measure memory allocation
- After each scaling operation: verify worker count ∈ [1, max], sum(workerMemory) ≤ budget, no frames dropped
- Verify initial auto-detection produces count ≤ 4 regardless of hardware

**Generator Strategy:**
```
arbitraryScalingSequence = array(1-20, randomChoice([
  { action: 'scaleUp' },
  { action: 'scaleDown' },
  { action: 'resizeTo', n: randomInt(1, 8) }
]))
```


### Property 3: Decode Pool Concurrency Is Bounded by Pool Size

*For any* pool size (1-8, matching the worker count) and *for any* sequence of interleaved `acquire()` / `release()` operations, the number of concurrently acquired video elements SHALL never exceed the pool size, and frames SHALL be dispatched in index order so the output sequence is preserved. Because concurrency is capped at the pool size, decode memory is bounded structurally by `poolSize × frameSize` without any separate byte budget.

**Validates: Requirements 3.5, 3.10, 5.6, 5.7**

**Test Strategy:**
- Generate random pool sizes / worker counts (1-8) and random sequences of acquire/release operations (including more concurrent acquire requests than the pool size, so the waiting queue is exercised)
- Track the count of outstanding (acquired-but-not-yet-released) elements after each operation
- Assert: outstanding count ≤ pool size at all times, and queued acquirers resolve only as elements are released (FIFO)
- Dispatch frames in index order and assert the recombined output frame order matches the input order

**Generator Strategy:**
```
arbitraryPoolConfig = {
  poolSize: randomInt(1, 8),            // = worker count
  frameWidth: randomChoice([1280, 1920, 2560, 3840]),
  frameHeight: randomChoice([720, 1080, 1440, 2160])
}

arbitraryAcquireSequence = array(1-40, randomChoice([
  { action: 'acquire' },
  { action: 'release' }   // released against a previously acquired element
]))

// invariant: concurrentlyAcquired ≤ poolConfig.poolSize
// derived bound: decodeMemoryMB ≤ poolSize × ((width × height × 4) / (1024 × 1024))
```

### Property 4: Settings Serialization Round-Trip

*For any* valid ProcessingSettings object (all presets, all custom mode combinations, all valid parameter ranges), serializing then deserializing SHALL produce a structurally equivalent settings object: `parse(print(settings)) deepEquals settings`.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.9**

**Test Strategy:**
- Generate random valid settings: vary preset, detail (1-100), lineWeight (1-5), scale (0.1-2.0), all custom mode toggles and sliders
- Serialize: `json = SettingsPrettyPrinter.print(settings)`
- Deserialize: `parsed = SettingsParser.parse(json)`
- Assert: `deepEquals(parsed, settings)`

**Generator Strategy:**
```
arbitrarySettings = {
  preset: randomChoice(['manga', 'studio', 'neon', 'warm', 'vivid', 'blueprint', 'custom']),
  detail: randomInt(1, 100),
  lineWeight: randomInt(1, 5),
  scale: randomFloat(0.1, 2.0),
  videoFps: randomChoice([12, 18, 24, 30, 60]),
  customMode: randomBool(),
  ...(if customMode: {
    useBilateral: randomBool(),
    bilateralPasses: randomInt(1, 5),
    useGaussian: randomBool(),
    gaussianPasses: randomInt(1, 5),
    useMedian: randomBool(),
    medianPasses: randomInt(1, 3),
    cleanSpeckles: randomBool(),
    cleanSpecklesIntensity: randomInt(1, 3),
    ... (all other custom options)
  })
}
```

### Property 5: Color Lines Preserve Color Accuracy Under Normalization

*For any* frame with auto-normalize enabled AND color edges enabled, the color sampling for soft edges SHALL occur on raw RGB before normalization, resulting in color accuracy deltaE < 5 and opacity retention ≥ 95% of slider setting, regardless of normalization strength (gamma lift, histogram stretch, CLAHE clip limit).

**Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.9**

**Test Strategy:**
- Generate random frames: vary mean brightness (20-240) to trigger different normalization stages
- Enable color edges + auto-normalize with random settings
- Process frame, extract colored edge pixels
- Compare color values against raw RGB source: measure deltaE color difference
- Verify opacity matches slider setting within 95% tolerance

**Generator Strategy:**
```
arbitraryNormalizationFrame = {
  image: generateImageWithBrightness(meanBrightness = randomInt(20, 240)),
  settings: {
    autoNormalize: true,
    colorEdges: true,
    colorOpacity: randomFloat(0.0, 1.0),
    colorSoftness: randomInt(0, 10),
    darkBoost: randomBool(),
    darkBoostClip: randomFloat(1.0, 6.0)
  }
}
```

### Property 6: Memory Usage Remains Bounded for Long Videos

*For any* video length (1 second to 8 hours) and *for any* worker count (1-8), peak memory usage SHALL never exceed `baseMemory + (workerCount × 80MB) + (decodePoolSize × frameSizeMB) + 400MB` (base + workers + decode pool + FFmpeg chunk), where `decodePoolSize = workerCount`, and memory SHALL NOT grow unbounded over time (no memory leak).

**Validates: Requirements 5.1, 5.2, 5.5, 5.6, 5.7, 5.8, 13.1, 13.10**

**Test Strategy:**
- Simulate processing: vary video duration (60 frames to 100,000 frames), vary worker count (1-8), vary frame size (1-8 MP)
- Track memory usage: sample performance.memory every 100 frames
- Verify peak memory ≤ calculated budget
- Verify memory at frame 1000 ≈ memory at frame 100 (no unbounded growth)

**Generator Strategy:**
```
arbitraryVideoWorkload = {
  frameCount: randomChoice([60, 600, 6000, 60000, 100000]),
  frameWidth: randomChoice([1280, 1920, 2560, 3840]),
  frameHeight: randomChoice([720, 1080, 1440, 2160]),
  workerCount: randomInt(1, 8),
  decodePoolSize: workerCount   // pool tracks the worker count
}

expectedPeakMemory = 300 + (workerCount × 80) + (decodePoolSize × frameSizeMB) + 400
```


### Property 7: Frame Processing Order Preservation

*For any* sequence of N frames processed with adaptive worker scaling active (workers may scale up/down during processing), the output video SHALL contain exactly N frames in the original input sequence order (outputFrame[i].index === i for all i).

**Validates: Requirements 2.5, 2.6, 3.7, 13.3**

**Test Strategy:**
- Generate random frame sequence: 100-1000 frames, each watermarked with frame index
- Process with adaptive scaling enabled: randomly trigger scale-up/scale-down conditions
- Verify output: count total frames, check index watermark in sequence
- Assert: outputFrameCount === inputFrameCount, outputFrame[i].watermark === i

**Generator Strategy:**
```
arbitraryFrameSequence = array(100-1000, frame => {
  pixels: embedWatermark(generateRandomFrame(), frameIndex)
})

simulateScalingEvents = array(5-15, randomChoice([
  { atFrame: randomInt(0, totalFrames), action: 'scaleUp' },
  { atFrame: randomInt(0, totalFrames), action: 'scaleDown' }
]))
```

## Error Handling

### GPU Device Lost Recovery

**Scenario:** WebGPU device becomes unavailable during processing (driver crash, system suspend, GPU reset)

**Detection:**
```javascript
device.lost.then((info) => {
  console.error(`[GPU] Device lost: ${info.message}, reason: ${info.reason}`);
  handleDeviceLost(info.reason);
});
```

**Recovery Strategy:**
1. Detect device-lost event in worker
2. Post `{ type: 'gpu-fallback', reason }` message to main thread
3. Worker automatically falls back to OpenCV CPU pipeline
4. Main thread logs diagnostic info but does NOT interrupt rendering
5. Processing continues from next frame using CPU path

**Constraints:**
- Current frame in GPU pipeline is lost (acceptable)
- Resume processing from `currentFrameIndex + 1`
- No retry attempts on GPU (permanent fallback for remainder of render)

### Worker Scaling Failure

**Scenario:** Attempt to scale up workers fails (spawn error, initialization timeout, insufficient memory)

**Detection:**
```javascript
try {
  processor.resize(newWorkerCount);
} catch (error) {
  console.warn(`[Scaling] Failed to scale to ${newWorkerCount}: ${error.message}`);
  // Continue with current worker count
}
```

**Recovery Strategy:**
1. Log scaling failure with diagnostic info
2. Disable adaptive scaling for this render session
3. Continue processing with current worker count
4. UI displays warning: "Worker scaling unavailable, using N workers"

### Decode Failure

**Scenario:** Seeking or drawing a frame on an acquired video element fails (corrupt frame, unsupported codec mid-stream, seek timeout)

**Detection:**
```javascript
const { video: vEl, release } = await decodePool.acquire();
try {
  await seekVideo(vEl, frameTime);
  drawMediaToCanvas(vEl, offCanvas, scale, custom);
} catch (error) {
  console.warn(`[Decode] Failed to decode frame ${frameIndex}: ${error.message}`);
  return handleDecodeFailure(frameIndex);
} finally {
  release();   // always return the element to the pool
}
```

**Recovery Strategy:**
1. Log frame index and error details
2. Skip the corrupt frame (do not add to output video)
3. Release the video element back to the pool so it can be reused
4. Continue with the next frame
5. UI updates: "Skipped 1 corrupt frame at timestamp X"
6. Final output frame count: `totalFrames - corruptFrameCount`

**Constraints:**
- Maximum 5% corrupt frames tolerated (if more, abort render with error)
- Audio sync may drift if many frames skipped

### Memory Pressure Handling

**Scenario:** System memory drops below safe threshold during processing (browser tab background, other apps consuming RAM)

**Detection:**
```javascript
const availableMB = perfMonitor.getAvailableMemoryMB();
if (availableMB < 100) {
  handleMemoryPressure();
}
```

**Recovery Strategy:**
1. Reduce the decode pool size (fewer concurrent video elements) so less decode memory is held at once
2. If adaptive scaling active: scale down workers by 1 (the decode pool resizes to match)
3. Wait for in-flight frames to complete and release memory
4. Resume processing at reduced capacity
5. UI displays: "Memory pressure detected, reduced workers to N"

**Constraints:**
- Never scale below 1 worker (processing always continues)
- The decode pool size is never reduced below 1 (at least one concurrent decode element)


### Settings Validation Errors

**Scenario:** User provides invalid settings (threshold low >= high, out-of-range values, incompatible filter combinations)

**Detection:**
```javascript
try {
  const settings = SettingsParser.parse(jsonString);
} catch (error) {
  console.error(`[Settings] Validation failed: ${error.message}`);
  return showValidationError(error);
}
```

**Validation Rules:**
1. **Threshold ordering:** `lowThreshold + 24 ≤ highThreshold`
2. **Range bounds:** `detail ∈ [1,100]`, `lineWeight ∈ [1,5]`, `scale ∈ [0.1,2.0]`
3. **Color edge thresholds:** `colorLowThresh < colorHighThresh`
4. **Preset existence:** `preset ∈ {manga, studio, neon, warm, vivid, blueprint, custom}`

**Recovery Strategy:**
1. Display validation error in UI with specific field and reason
2. Auto-correct if possible: `highThreshold = max(highThreshold, lowThreshold + 24)`
3. Prevent render start until validation passes
4. Preserve user's other valid settings

### FFmpeg Encoding Failure

**Scenario:** FFmpeg fails during chunk encode or final mux (out of memory, invalid format, WASM heap exceeded)

**Detection:**
```javascript
await ffmpeg.exec(...);  // May throw
```

**Recovery Strategy:**
1. If chunk encode fails: retry once with reduced chunk size (200MB)
2. If retry fails: abort render with error "Encoding failed: [reason]"
3. If final mux fails (audio too large): fall back to video-only output (existing graceful degradation)
4. Save partial output so user doesn't lose all work

**Constraints:**
- Existing WASM heap limit handling preserved
- Video-only fallback remains unchanged

## Testing Strategy

### Unit Testing Strategy

**Scope:** Individual components and algorithms

**Test Framework:** Jest (JavaScript) for CPU code, WebGPU shader testing via compute pass validation

**Coverage Targets:**
- **GPU Compute Shaders:** Each shader tested in isolation with known inputs/outputs
  - CLAHE: OpenCV reference outputs for various clip limits
  - Auto-Normalize: Known gamma/stretch/CLAHE transforms
  - Clean-Speckles: Binary masks with known component sizes
  - Color-Edges: RGB outputs with known color values
- **Worker Pool:** Spawn, resize, drain, termination logic
- **Decode Pool:** pool sizing to worker count, acquire/release, resize, frame ordering
- **Performance Monitor:** Metric recording, statistical calculations (median, p95)
- **Settings Parser:** Validation rules, JSON round-trip
- **Adaptive Scaler:** Scaling decision logic with mocked metrics

**Example Unit Tests:**
```javascript
describe('CLAHEComputeShader', () => {
  it('produces output within MAE < 3 of OpenCV reference', async () => {
    const input = generateTestImage(256, 256, 'lowContrast');
    const gpuOutput = await claheShader.apply(input, clipLimit=2.5);
    const cpuOutput = await opencvCLAHE(input, clipLimit=2.5);
    expect(meanAbsoluteError(gpuOutput, cpuOutput)).toBeLessThan(3);
  });
});

describe('AdaptiveWorkerScaler', () => {
  it('scales up when latency high and memory available', () => {
    perfMonitor.setMedianLatency(120);
    perfMonitor.setAvailableMemory(200);
    scaler.evaluateAndAdjust();
    expect(processor.concurrency).toBe(initialCount + 1);
  });
});
```

### Property-Based Testing Strategy

**Test Framework:** fast-check (JavaScript property-based testing library)

**Configuration:**
- **Iterations per property:** 100 minimum (due to randomization)
- **Shrinking:** Enabled to find minimal failing examples
- **Timeout:** 30 seconds per property test
- **Seed:** Logged for reproducibility

**Property Test Tagging:**
Each property test MUST include a comment tag referencing the design document:

```javascript
/**
 * Feature: video-processing-performance-boost, Property 1:
 * GPU/CPU Output Equivalence Across All Filters
 * 
 * For any valid video frame and processing settings, GPU and CPU pipelines
 * SHALL produce pixel-equivalent output with MAE ≤ 1 and SSIM > 0.99.
 */
it('property: GPU/CPU equivalence', () => {
  fc.assert(
    fc.property(
      fc.record({
        width: fc.constantFrom(1280, 1920, 2560, 3840),
        height: fc.constantFrom(720, 1080, 1440, 2160),
        meanBrightness: fc.integer(20, 240),
        contrast: fc.integer(10, 80)
      }),
      arbitrarySettings(),
      async (frameConfig, settings) => {
        const frame = generateFrame(frameConfig);
        const gpuOutput = await processGPU(frame, settings);
        const cpuOutput = await processCPU(frame, settings);
        
        const mae = meanAbsoluteError(gpuOutput, cpuOutput);
        const ssim = structuralSimilarity(gpuOutput, cpuOutput);
        
        expect(mae).toBeLessThanOrEqual(1);
        expect(ssim).toBeGreaterThan(0.99);
      }
    ),
    { numRuns: 100 }
  );
});
```

**All 7 properties from Correctness Properties section MUST be implemented as property-based tests.**


### Integration Testing Strategy

**Scope:** End-to-end workflows, cross-platform compatibility, performance validation

**Test Environment:**
- **Browsers:** Chrome 113+, Edge 113+, Firefox (experimental WebGPU)
- **Platforms:** Windows 10/11, macOS (Intel + Apple Silicon), Linux (Ubuntu 22.04)
- **Hardware:** Reference hardware (RTX 3060 equivalent) + low-end hardware (integrated GPU)

**Integration Test Suites:**

#### 1. Cross-Platform Output Equivalence
**Purpose:** Verify GPU/CPU pipelines produce identical output on all platforms

**Test Cases:**
- Process 10 reference test frames (humans, wildlife, urban, low-light, high-contrast) with each preset
- Compare output pixel data against golden master files
- Assert: SSIM > 0.99 on all platforms

**Execution:** Run on CI matrix (Windows/Chrome, macOS/Safari, Linux/Firefox)

#### 2. Performance Benchmarks
**Purpose:** Validate 40-50% performance improvement target

**Test Cases:**
- **Baseline:** Process 1-minute 4K 60fps video with all optimizations disabled (fixed 4 workers, single-element decode, hybrid GPU/CPU)
- **Optimized:** Same video with all optimizations enabled (adaptive scaling, parallel decode pool, complete GPU pipeline)
- Measure total processing time, frames per second
- Assert: `optimizedTime ≤ baselineTime * 0.6` (40% improvement minimum)

**Reference Videos:**
- 1 minute 4K 60fps (1800 frames) - target: baseline 3min → optimized 1.8min
- 10 minutes 1080p 30fps (18,000 frames) - target: baseline 15min → optimized 9min

#### 3. Stability Under Extreme Conditions
**Purpose:** Verify system remains stable with long videos and adverse conditions

**Test Cases:**
- **8-hour video:** Process 8-hour 4K 60fps test pattern, verify completion without OOM
- **Rapid scene changes:** Process video with new scene every 2 seconds, verify no visual artifacts
- **GPU device loss:** Force WebGPU device-lost event mid-render, verify CPU fallback and completion
- **Memory pressure:** Simulate low-memory condition (mock performance.memory), verify graceful scaling
- **Corrupt frames:** Process video with 5% intentionally corrupt frames, verify skip and continue

**Assertions:**
- Processing completes successfully
- Peak memory ≤ 2GB on 8GB RAM systems
- No unbounded memory growth (sample every 1000 frames)
- Output frame count correct (excluding skipped corrupt frames)

#### 4. Adaptive Scaling Effectiveness
**Purpose:** Verify adaptive worker scaling improves performance

**Test Cases:**
- Process video with varying system load simulation (artificially delay random frames)
- Monitor worker count changes over time
- Verify: At least 3 scaling adjustments occur
- Verify: Average latency with adaptive scaling < average latency with fixed 4 workers

#### 5. Parallel Decode Effectiveness
**Purpose:** Verify parallel decode (a pool of video elements) reduces worker idle time vs a smaller pool

**Test Cases:**
- Process the same video at a smaller decode-pool size (fewer concurrent video elements) vs a larger decode-pool size
- Measure worker idle time percentage (time waiting for decode)
- Assert: `idleTime_largerPool < idleTime_smallerPool * 0.5` (50% reduction)
- Assert: `idleTime_largerPool < 5%` of total processing time

#### 6. Settings Persistence and Restoration
**Purpose:** Verify custom mode settings save/restore correctly

**Test Cases:**
- Configure custom mode with random valid settings
- Serialize to localStorage
- Reload page
- Verify all settings restored correctly
- Process frame, verify output matches expected for those settings

### Manual Testing Strategy

**Scope:** Subjective quality validation for human subject videos and aesthetic consistency

**Test Protocol:**

#### Human Subject Line-Art Quality (Requirement 14)
1. Process 5 test videos of human subjects with varying lighting (bright outdoor, dim indoor, side-lit, backlit, spotlight)
2. Use Custom/Experiment mode with recommended starting values: `lowThresh=40, highThresh=100, bilateralDiameter=13, sigma=90, cleanSpeckles=enabled`
3. Manual inspection checklist:
   - [ ] Facial features (eyes, nose, mouth, eyebrows) rendered as continuous clean lines
   - [ ] No shadow-induced spurious lines on skin
   - [ ] Hair strands preserved without excessive fragmentation
   - [ ] Clothing folds do not create double-edge artifacts
4. Adjust thresholds iteratively to find optimal range for human subjects
5. Document optimal settings in requirements notes

#### Consistent Line-Art Aesthetic (Requirement 15)
1. Process 4 diverse video types: wildlife, people, plants, urban scenes
2. Use same preset (Studio) for all
3. Manual inspection checklist:
   - [ ] Consistent line weight across all videos
   - [ ] Balanced ink distribution (not too sparse or too dense)
   - [ ] Clean continuous strokes (not fragmented pixel noise)
   - [ ] Consistent background treatment
4. Compute automated metrics: edge density histogram, line continuity score
5. Assert: All videos fall within 10% of target metric ranges

### Performance Profiling

**Tools:**
- Chrome DevTools Performance Profiler
- WebGPU Timestamp Queries (for GPU pass timing)
- performance.memory API (for heap tracking)

**Profiling Sessions:**

1. **GPU Shader Performance:** Measure per-pass execution time
   - Target: CLAHE < 15ms, Auto-Normalize < 20ms, Clean-Speckles < 25ms, Color-Edges < 30ms (all at 4K resolution)

2. **Bottleneck Identification:** Profile full render of 1-minute video
   - Identify slowest stages: decode, process, encode
   - Verify processing is the slowest stage (decode/encode are fast)

3. **Memory Growth Analysis:** Profile 10,000 frame render
   - Sample heap usage every 100 frames
   - Plot over time, verify flat or bounded growth (no leak)

## Implementation Phases

### Phase 1: GPU Compute Shaders (Weeks 1-3)
**Goal:** Implement missing GPU shaders for complete GPU pipeline

**Tasks:**
1. Implement CLAHE compute shader (tile histogram, clip, CDF, interpolation)
2. Implement Auto-Normalize shaders (gamma LUT, histogram stretch, adaptive CLAHE)
3. Implement Clean-Speckles shader (parallel connected component labeling)
4. Implement Color-Edges shaders (Canny on raw gray, color sampling, compositing)
5. Write unit tests for each shader against OpenCV reference outputs
6. Integrate shaders into gpu-worker.js processing pipeline
7. Test end-to-end: verify preview and video export work on GPU path

**Deliverables:**
- 4 new WGSL compute shaders in gpu-worker.js
- Unit tests for each shader
- Integration test: full video render on GPU-only path

**Success Criteria:**
- All shaders produce output within MAE < 3 of CPU reference
- GPU pipeline handles all filters without CPU fallback
- Property 1 (GPU/CPU equivalence) passes


### Phase 2: Performance Monitoring (Week 4)
**Goal:** Implement comprehensive performance monitoring infrastructure

**Tasks:**
1. Create PerformanceMonitor class in script.js
2. Instrument frame processing loop: record decode, process, encode timestamps
3. Implement statistical calculations: median, p95, p99 latencies
4. Implement GPU/CPU usage tracking (increment counters in worker message handlers)
5. Implement memory availability monitoring (performance.memory API)
6. Implement bottleneck identification heuristic
7. Add console logging: summary every 100 frames
8. Add final render summary log with all metrics

**Deliverables:**
- PerformanceMonitor class with all metrics
- Instrumented renderVideo() loop
- Console logging of real-time metrics

**Success Criteria:**
- All metrics populate correctly during test render
- Bottleneck identification matches manual profiler analysis
- Memory tracking reflects actual heap usage

### Phase 3: Adaptive Worker Scaling (Week 5)
**Goal:** Implement dynamic worker pool resizing based on performance metrics

**Tasks:**
1. Create AdaptiveWorkerScaler class in script.js
2. Implement scaling decision logic (latency thresholds, memory checks)
3. Integrate with existing LineArtProcessor.resize() method
4. Add cooldown timer to prevent thrashing
5. Hook into renderVideo() loop: evaluate scaling every 50 frames
6. Add UI indicator showing current worker count and auto-scaling status
7. Write unit tests for scaling decisions with mocked metrics
8. Test manual override: user slider disables auto-scaling

**Deliverables:**
- AdaptiveWorkerScaler class
- Integration with renderVideo() loop
- UI indicator for active workers

**Success Criteria:**
- Property 2 (worker pool invariants) passes
- Test render shows at least 3 scaling adjustments
- Manual slider override works correctly

### Phase 4: Parallel Video Decode Pool (Week 6)
**Goal:** Implement a parallel video decode pool to eliminate the decode bottleneck

**Tasks:**
1. Create VideoDecodePool class in script.js
2. Implement acquire()/release() with a FIFO waiting queue
3. Size the pool to the worker count; clone video elements sharing the source object-URL src
4. Implement resize(n) so the pool tracks worker-pool changes (and destroy())
5. Integrate into the render loop: acquire → seek → draw → release per frame
6. Apply FRAME_STAGGER_MS staggering to the first batch of acquires
7. Test with varying pool sizes: measure worker idle time
8. Add console logging: pool size and acquire/release activity

**Deliverables:**
- VideoDecodePool class
- Integration with the render loop
- Worker idle time measurement

**Success Criteria:**
- Property 3 (decode pool concurrency bounded by pool size) passes
- Worker idle time < 5% at the larger decode-pool size
- Decode pool size tracks the worker count

### Phase 5: Settings Serialization (Week 7)
**Goal:** Implement settings parser and serializer for testing and persistence

**Tasks:**
1. Create SettingsParser class with validation rules
2. Create SettingsPrettyPrinter class with consistent formatting
3. Write unit tests for validation edge cases (threshold ordering, range bounds)
4. Implement localStorage persistence in script.js
5. Implement "Reset to Default" button in custom mode
6. Test round-trip: save → reload page → verify restoration

**Deliverables:**
- SettingsParser and SettingsPrettyPrinter classes
- localStorage integration
- Reset button

**Success Criteria:**
- Property 4 (settings round-trip) passes
- All validation rules enforce correctly
- Settings persist across page reloads

### Phase 6: Integration Testing & Optimization (Weeks 8-9)
**Goal:** Validate performance targets and cross-platform compatibility

**Tasks:**
1. Set up cross-platform CI matrix (Windows, macOS, Linux)
2. Create reference test video suite (1min 4K, 10min 1080p, diverse content)
3. Run performance benchmarks: measure baseline vs optimized times
4. Run stability tests: 8-hour video, GPU device loss, memory pressure
5. Run cross-platform equivalence tests: compare outputs with golden masters
6. Profile bottlenecks: identify any remaining performance issues
7. Optimize hot paths if targets not met
8. Run all 7 property-based tests with 100 iterations each

**Deliverables:**
- CI pipeline with cross-platform test matrix
- Performance benchmark results
- Property-based test suite

**Success Criteria:**
- **Performance target achieved:** 2.5-hour 4K video completes in ≤ 2 hours (40% improvement)
- **All 7 properties pass** with 100 iterations
- **Cross-platform equivalence:** SSIM > 0.99 on all platforms
- **Stability:** 8-hour video completes without OOM

### Phase 7: Documentation & Release (Week 10)
**Goal:** Finalize documentation and prepare release

**Tasks:**
1. Update README with performance benchmarks
2. Document optimal settings for human subjects
3. Write developer guide for GPU shader development
4. Document adaptive scaling behavior and tuning
5. Create user guide for custom mode parameters
6. Add performance monitoring console commands for power users
7. Final QA pass: all test suites green
8. Release notes with performance improvements and new features

**Deliverables:**
- Updated documentation
- Release notes
- Deployment to production

**Success Criteria:**
- All documentation complete and accurate
- All tests passing
- Performance targets validated on production build

## Risk Mitigation

### Risk 1: GPU Shader Complexity
**Probability:** Medium  
**Impact:** High (CLAHE and connected components are algorithmically complex)

**Mitigation:**
- Start with simple reference implementation in JavaScript
- Validate correctness before optimizing
- Use existing OpenCV outputs as test oracles
- Break algorithms into multiple compute passes if single-pass is too complex

### Risk 2: Cross-Platform WebGPU Inconsistencies
**Probability:** Medium  
**Impact:** High (output equivalence is critical requirement)

**Mitigation:**
- Test on all target platforms early (Chrome Windows, macOS, Linux)
- Use strict WGSL specification compliance (avoid vendor-specific extensions)
- Implement extensive integration tests with golden masters
- Maintain CPU fallback as safety net

### Risk 3: Performance Target Not Achieved
**Probability:** Low  
**Impact:** High (40% improvement is primary goal)

**Mitigation:**
- Measure baseline performance before starting
- Track incremental improvements after each phase
- Profile early and often to identify bottlenecks
- Have fallback optimizations identified (e.g., reduce the decode pool size, optimize shader dispatches)

### Risk 4: Memory Regression
**Probability:** Low  
**Impact:** Critical (existing stability must be preserved)

**Mitigation:**
- Never modify existing memory safety mechanisms
- Add integration tests for memory bounds before implementation
- Monitor memory usage continuously during development
- Test with longest video (8 hours) to catch leaks early

### Risk 5: Adaptive Scaling Instability
**Probability:** Medium  
**Impact:** Medium (scaling thrashing could harm performance)

**Mitigation:**
- Implement conservative cooldown period (10 seconds)
- Use hysteresis in thresholds (scale up at 100ms, scale down at 40ms)
- Allow manual override (user can disable auto-scaling)
- Test with synthetic load variations to validate stability

## Appendix A: WGSL Shader Pseudocode

### CLAHE Shader (4 Compute Passes)

**Pass 1: Tile Histogram Computation**
```wgsl
@compute @workgroup_size(8, 8)
fn compute_tile_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tileX = gid.x;  // Tile coordinate (0 to width/8 - 1)
  let tileY = gid.y;
  let tileIdx = tileY * tilesPerRow + tileX;
  
  // Each workgroup processes one 8×8 tile
  for (var py = 0u; py < 8u; py++) {
    for (var px = 0u; px < 8u; px++) {
      let pixelX = tileX * 8u + px;
      let pixelY = tileY * 8u + py;
      let intensity = grayImage[pixelY * width + pixelX];
      
      // Atomic increment histogram bin for this tile
      atomicAdd(&tileHistograms[tileIdx * 256 + u32(intensity * 255.0)], 1u);
    }
  }
}
```

**Pass 2: Clip and Redistribute**
```wgsl
@compute @workgroup_size(256)
fn clip_and_redistribute(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tileIdx = gid.x / 256u;
  let bin = gid.x % 256u;
  
  let histValue = tileHistograms[gid.x];
  let clipLimit = params.clipLimit * 64.0 / 256.0;  // 64 pixels per tile
  
  // Clip excess
  var clipped = 0u;
  var newValue = histValue;
  if (f32(histValue) > clipLimit) {
    clipped = u32(f32(histValue) - clipLimit);
    newValue = u32(clipLimit);
  }
  
  // Store clipped histogram
  clippedHistograms[gid.x] = newValue;
  
  // Accumulate total clipped pixels for this tile (reduction)
  atomicAdd(&totalClipped[tileIdx], clipped);
  
  workgroupBarrier();
  
  // Redistribute equally
  let redistributePerBin = totalClipped[tileIdx] / 256u;
  clippedHistograms[gid.x] += redistributePerBin;
}
```


**Pass 3: CDF Computation (Parallel Prefix Sum)**
```wgsl
@compute @workgroup_size(256)
fn compute_cdf(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tileIdx = gid.x / 256u;
  let bin = gid.x % 256u;
  
  // Work-efficient parallel scan (Blelloch)
  var temp = clippedHistograms[gid.x];
  var offset = 1u;
  
  // Up-sweep (reduce) phase
  for (var d = 128u; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (bin < d) {
      let ai = offset * (2u * bin + 1u) - 1u;
      let bi = offset * (2u * bin + 2u) - 1u;
      temp[bi] += temp[ai];
    }
    offset <<= 1u;
  }
  
  // Down-sweep phase
  if (bin == 0u) { temp[255] = 0u; }
  for (var d = 1u; d < 256u; d <<= 1u) {
    offset >>= 1u;
    workgroupBarrier();
    if (bin < d) {
      let ai = offset * (2u * bin + 1u) - 1u;
      let bi = offset * (2u * bin + 2u) - 1u;
      let t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  
  cdfs[gid.x] = temp;
}
```

**Pass 4: Bilinear Interpolation Output**
```wgsl
@compute @workgroup_size(16, 16)
fn apply_clahe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= width || y >= height) { return; }
  
  let intensity = grayImage[y * width + x];
  let bin = u32(intensity * 255.0);
  
  // Determine which tile(s) this pixel belongs to
  let tileX = f32(x) / 8.0;
  let tileY = f32(y) / 8.0;
  
  // Tile indices (clamped to image bounds)
  let tileX0 = u32(floor(tileX));
  let tileY0 = u32(floor(tileY));
  let tileX1 = min(tileX0 + 1u, tilesPerRow - 1u);
  let tileY1 = min(tileY0 + 1u, tilesPerCol - 1u);
  
  // Interpolation weights
  let wx = fract(tileX);
  let wy = fract(tileY);
  
  // Fetch CDF values from 4 neighboring tiles
  let cdf00 = cdfs[(tileY0 * tilesPerRow + tileX0) * 256u + bin];
  let cdf10 = cdfs[(tileY0 * tilesPerRow + tileX1) * 256u + bin];
  let cdf01 = cdfs[(tileY1 * tilesPerRow + tileX0) * 256u + bin];
  let cdf11 = cdfs[(tileY1 * tilesPerRow + tileX1) * 256u + bin];
  
  // Bilinear interpolation
  let cdf0 = mix(f32(cdf00), f32(cdf10), wx);
  let cdf1 = mix(f32(cdf01), f32(cdf11), wx);
  let finalCdf = mix(cdf0, cdf1, wy);
  
  // Normalize CDF to [0, 1]
  let maxCdf = 64.0;  // 8×8 = 64 pixels per tile
  outputImage[y * width + x] = finalCdf / maxCdf;
}
```

### Connected Component Labeling (Iterative Label Propagation)

```wgsl
// Initialization: Each white pixel gets unique label
@compute @workgroup_size(16, 16)
fn init_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= width || y >= height) { return; }
  
  let idx = y * width + x;
  if (binaryMask[idx] > 127u) {
    labels[idx] = idx;  // Unique label = pixel index
  } else {
    labels[idx] = 0xFFFFFFFFu;  // Background marker
  }
}

// Label propagation (run 8 iterations)
@compute @workgroup_size(16, 16)
fn propagate_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= width || y >= height) { return; }
  
  let idx = y * width + x;
  let currentLabel = labelsIn[idx];
  if (currentLabel == 0xFFFFFFFFu) { return; }  // Skip background
  
  var minLabel = currentLabel;
  
  // Check 8-connected neighbors
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nx = clamp(i32(x) + dx, 0, i32(width) - 1);
      let ny = clamp(i32(y) + dy, 0, i32(height) - 1);
      let neighborLabel = labelsIn[u32(ny) * width + u32(nx)];
      if (neighborLabel != 0xFFFFFFFFu) {
        minLabel = min(minLabel, neighborLabel);
      }
    }
  }
  
  labelsOut[idx] = minLabel;
}

// Component area computation (parallel histogram)
@compute @workgroup_size(16, 16)
fn compute_areas(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= width || y >= height) { return; }
  
  let idx = y * width + x;
  let label = labels[idx];
  if (label != 0xFFFFFFFFu) {
    atomicAdd(&componentAreas[label], 1u);
  }
}

// Filter by area threshold
@compute @workgroup_size(16, 16)
fn filter_components(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= width || y >= height) { return; }
  
  let idx = y * width + x;
  let label = labels[idx];
  if (label == 0xFFFFFFFFu) {
    outputMask[idx] = 0u;
    return;
  }
  
  let area = componentAreas[label];
  if (area >= minArea) {
    outputMask[idx] = 255u;  // Keep component
  } else {
    outputMask[idx] = 0u;    // Remove component
  }
}
```

## Appendix B: Performance Calculation Worksheet

### Baseline Performance (Current Implementation)

**Video:** 2.5 hours 4K 60fps = 540,000 frames at 3840×2160

**Current Processing:**
- **Frame Processing Time:** ~20-100ms per frame (GPU for some filters, CPU for CLAHE/Clean-Speckles/Color-Edges)
- **Worker Count:** Fixed 4 workers
- **Decode Overhead:** 5-10% worker idle time waiting for decode
- **Total Time:** 3-3.5 hours (180-210 minutes)

**Calculation:**
- Average frames per second: 540,000 / (195 * 60) = ~46 fps
- Average frame latency: 1000 / 46 ≈ 21.7ms per frame

### Optimized Performance (Target)

**Optimizations Applied:**

1. **Complete GPU Pipeline:** CLAHE (15ms → 5ms), Auto-Normalize (20ms → 8ms), Clean-Speckles (30ms → 10ms), Color-Edges (40ms → 15ms)
   - **Improvement:** ~25% reduction in frame processing time

2. **Adaptive Worker Scaling:** Average 5-6 workers instead of fixed 4 (on reference hardware)
   - **Improvement:** ~10% increase in throughput

3. **Parallel Video Decode Pool:** Eliminate 5-10% idle time
   - **Improvement:** ~5% increase in throughput

**Combined Improvement:** `1 - (0.75 * 0.90 * 0.95) = 1 - 0.64 = 36%` baseline improvement  
**With real-world compounding:** ~40-45% improvement expected

**Target Calculation:**
- Optimized time: 195 minutes * 0.55 = **107 minutes (1 hour 47 minutes)**
- Optimized fps: 540,000 / (107 * 60) = **84 fps**
- Average frame latency: 1000 / 84 ≈ **12ms per frame**

**Target Met:** 107 minutes < 120 minutes (2 hours) ✓

