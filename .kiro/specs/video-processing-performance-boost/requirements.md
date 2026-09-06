# Requirements Document

## Introduction

This document specifies requirements for optimizing the video processing performance of a browser-based line-art animation converter. The application currently processes videos entirely in-browser using WebGPU compute shaders (GPU-accelerated) with OpenCV.js WASM fallback (CPU), FFmpeg.js for encoding/decoding, and Web Workers for parallel processing.

**Performance Target**: Reduce processing time for a 2.5-hour 4K 60fps video (540,000 frames) from 3-3.5 hours to 1.5-2 hours (40-50% improvement).

**Critical Constraint**: All existing stability mechanisms (streaming encode with chunking, parallel FFmpeg workers, graceful fallback for oversized files, memory-bounded processing, batch-based GC checkpoints) MUST be preserved without regression.

## Glossary

- **GPU_Pipeline**: The WebGPU compute shader code path that processes video frames using GPU acceleration
- **CPU_Pipeline**: The OpenCV.js WASM fallback code path that processes frames on CPU when WebGPU is unavailable
- **Worker_Pool**: The collection of Web Workers that process video frames in parallel
- **Video_Decoder**: The browser component that decodes video frames from the source file
- **Frame_Processing_Latency**: The time taken to process a single video frame from decode through line-art conversion
- **Memory_Budget**: The calculated maximum RAM allocation for worker pool based on available system memory
- **Streaming_Encoder**: The FFmpeg-based component that encodes processed frames in chunks to prevent out-of-memory conditions
- **CLAHE_Filter**: Contrast Limited Adaptive Histogram Equalization image processing algorithm
- **Auto_Normalize_Filter**: Adaptive brightness and contrast correction algorithm (gamma lift, histogram stretch, adaptive CLAHE)
- **Clean_Speckles_Filter**: Connected component analysis algorithm that removes isolated edge fragments
- **Color_Edges_Filter**: Soft colored lines feature that renders edges with source image colors
- **Decode_Pool**: A pool of cloned HTMLVideoElement instances (sized to the worker count) that allow multiple frames to be sought and drawn in parallel, enabling concurrent frame decode so workers spend less time idle waiting on decode
- **Performance_Monitor**: The instrumentation component that tracks frame processing metrics

## Requirements

### Requirement 1: Complete GPU Pipeline Implementation with Output Equivalence

**User Story:** As a user processing long 4K videos on any platform, I want all image processing filters to use GPU acceleration when WebGPU is available AND produce identical visual output to CPU processing, so that frame processing completes 2-4x faster without any quality differences across platforms.

#### Acceptance Criteria

1. WHEN WebGPU is available AND CLAHE filter is requested, THE GPU_Pipeline SHALL execute CLAHE using a WebGPU compute shader
2. WHEN WebGPU is available AND Auto_Normalize_Filter is requested, THE GPU_Pipeline SHALL execute gamma lift using a WebGPU compute shader with LUT
3. WHEN WebGPU is available AND Auto_Normalize_Filter is requested, THE GPU_Pipeline SHALL execute histogram stretch using a WebGPU compute shader with NORM_MINMAX operation
4. WHEN WebGPU is available AND Auto_Normalize_Filter is requested, THE GPU_Pipeline SHALL execute adaptive CLAHE using a WebGPU compute shader with per-tile processing
5. WHEN WebGPU is available AND Clean_Speckles_Filter is requested, THE GPU_Pipeline SHALL execute connected component analysis using a WebGPU compute shader
6. WHEN WebGPU is available AND Color_Edges_Filter is requested, THE GPU_Pipeline SHALL execute color edge detection using a WebGPU compute shader
7. WHEN WebGPU is available AND Color_Edges_Filter is requested, THE GPU_Pipeline SHALL execute color blending with opacity control using a WebGPU compute shader
8. FOR ALL valid video frames processed through GPU and CPU pipelines with identical settings, THE output pixel data SHALL match exactly (pixel-perfect equivalence, allowing only for 1 intensity level difference due to floating-point rounding)
9. FOR ALL platforms (Windows, macOS, Linux, Android, iOS) WHERE the application runs, THE GPU_Pipeline and CPU_Pipeline SHALL produce visually identical output for the same input frame and settings
10. WHEN WebGPU initialization fails OR WebGPU becomes unavailable during processing, THE Worker_Pool SHALL transparently fall back to CPU_Pipeline for all affected filters
11. WHEN processing a video frame using GPU_Pipeline with all filters enabled, THE Frame_Processing_Latency SHALL be 5-50ms per frame on reference hardware (4K resolution, RTX 3060 or equivalent)
12. FOR ALL test cases executed on GPU_Pipeline and CPU_Pipeline with identical settings, THE visual difference measured by SSIM SHALL be greater than 0.99 (visually lossless)

### Requirement 2: Adaptive Worker Scaling

**User Story:** As a user with varying system resources, I want the worker pool to automatically adjust its size based on real-time performance metrics, so that processing remains efficient without manual intervention.

#### Acceptance Criteria

1. WHEN the application initializes, THE Worker_Pool SHALL compute an initial worker count based on available CPU cores and Memory_Budget
2. WHILE processing video frames, THE Performance_Monitor SHALL track Frame_Processing_Latency for each completed frame
3. WHEN the median Frame_Processing_Latency over the last 50 frames is below 40ms AND Memory_Budget allows additional workers AND current worker count is below maximum safe limit, THE Worker_Pool SHALL increase worker count by 1
4. WHEN the median Frame_Processing_Latency over the last 50 frames is above 100ms OR available system memory drops below Memory_Budget threshold, THE Worker_Pool SHALL decrease worker count by 1
5. WHEN scaling up worker count, THE Worker_Pool SHALL spawn new workers without interrupting in-flight frame processing
6. WHEN scaling down worker count, THE Worker_Pool SHALL terminate excess workers only after their current tasks complete
7. THE Worker_Pool SHALL maintain worker count between 1 and the hardware-based maximum at all times
8. WHEN Memory_Budget calculation indicates less than 80MB available per worker, THE Worker_Pool SHALL refuse to scale up regardless of latency metrics
9. WHILE processing a 2.5-hour 4K 60fps video on reference hardware, THE Worker_Pool SHALL complete at least 3 scaling adjustments (up or down) in response to changing system load
10. THE Worker_Pool SHALL respect the existing DEFAULT_SAFE_WORKER_CAP of 4 workers as the initial auto-detected value

### Requirement 3: Parallel Video Decode Pipeline

**User Story:** As a user processing high-resolution videos, I want frames to be decoded in parallel using a pool of video elements, so that workers never idle waiting for a single decoder to seek and draw one frame at a time.

#### Acceptance Criteria

1. WHEN video processing begins, THE Decode_Pool SHALL provision a pool of cloned HTMLVideoElement instances sized to match the current Worker_Pool worker count
2. WHILE processing video frames, THE Decode_Pool SHALL seek and draw multiple frames concurrently across its video elements so that decode work for distinct frames proceeds in parallel
3. WHEN a worker requests the next frame for processing, THE Decode_Pool SHALL assign an available video element to decode that frame without blocking on other in-progress decodes
4. WHEN the Worker_Pool changes its worker count, THE Decode_Pool SHALL resize the pool of video elements to match the new worker count
5. WHEN a video element completes decoding and the drawn frame is transferred to a worker, THE Decode_Pool SHALL release any per-frame buffers so that memory does not grow unbounded as processing continues
6. WHEN a video seek operation is required for a frame, THE Video_Decoder SHALL overlap seek latency for that frame with ongoing decode and processing of other frames
7. THE Decode_Pool SHALL preserve frame ordering so that decoded frames are delivered to the output pipeline in correct processing sequence despite parallel decode
8. WHEN processing a 2.5-hour 4K 60fps video, THE percentage of worker idle time waiting for decode SHALL be less than 5% of total processing time
9. THE Decode_Pool SHALL respect existing FRAME_STAGGER_MS timing constraints to prevent video element decode stalls
10. THE Decode_Pool SHALL bound the number of concurrently decoding video elements to the worker count so that parallel decode does not introduce unbounded memory growth

### Requirement 4: Performance Monitoring Instrumentation

**User Story:** As a developer analyzing performance bottlenecks, I want detailed real-time metrics on frame processing throughput and latency, so that I can identify optimization opportunities.

#### Acceptance Criteria

1. WHEN a video frame enters the processing pipeline, THE Performance_Monitor SHALL record the frame index and timestamp
2. WHEN a video frame completes processing, THE Performance_Monitor SHALL record the completion timestamp and calculate Frame_Processing_Latency
3. WHEN a video frame is decoded from Video_Decoder, THE Performance_Monitor SHALL record decode latency separately from processing latency
4. WHEN FFmpeg encodes a processed frame chunk, THE Performance_Monitor SHALL record encode latency per chunk
5. THE Performance_Monitor SHALL maintain a rolling window of the last 100 frame latencies for statistical analysis
6. THE Performance_Monitor SHALL calculate and expose median, p95, and p99 Frame_Processing_Latency metrics
7. THE Performance_Monitor SHALL track GPU pipeline usage percentage (frames processed on GPU vs CPU)
8. THE Performance_Monitor SHALL track worker pool utilization (percentage of time workers are busy vs idle)
9. WHEN processing completes, THE Performance_Monitor SHALL log summary statistics including total processing time, average frames per second, and bottleneck analysis
10. THE Performance_Monitor SHALL expose real-time metrics through browser DevTools console logging at configurable intervals

### Requirement 5: Memory Safety Preservation

**User Story:** As a user processing extremely long videos, I want memory usage to remain bounded regardless of video length, so that the browser never crashes with out-of-memory errors.

#### Acceptance Criteria

1. THE Streaming_Encoder SHALL continue processing video in chunks of approximately 400MB regardless of total video length
2. THE Worker_Pool SHALL maintain the existing Semaphore mechanism that limits concurrent in-flight frames based on Memory_Budget
3. THE application SHALL continue to apply batch-based garbage collection checkpoints at the same intervals as current implementation
4. WHEN source video size exceeds WASM heap capacity, THE application SHALL produce video-only output without audio muxing (existing graceful fallback)
5. THE Worker_Pool SHALL never allocate more than 80MB per worker as calculated in existing Memory_Budget formula
6. WHEN Decode_Pool decoded-frame memory usage exceeds 200MB, THE Decode_Pool SHALL pause starting additional concurrent decodes until memory is released
7. THE application SHALL never simultaneously hold more than (worker_count × 2 + 4) decoded frames in memory
8. FOR ALL processing of videos up to 8 hours in length at 4K 60fps, THE peak memory usage SHALL not exceed 2GB on systems with 8GB RAM
9. THE application SHALL preserve existing memory-bounded processing architecture without introducing new unbounded allocations
10. WHEN adaptive worker scaling increases worker count, THE Memory_Budget calculation SHALL prevent scaling if insufficient memory is available

### Requirement 6: Performance Target Achievement

**User Story:** As a user processing long 4K videos, I want processing time reduced by 40-50%, so that I can complete renders in half the time.

#### Acceptance Criteria

1. WHEN processing a 2.5-hour 4K 60fps video with WebGPU available AND all optimizations enabled, THE total processing time SHALL be 2 hours or less
2. WHEN processing a 2.5-hour 4K 60fps video with WebGPU available AND all optimizations enabled, THE processing time SHALL be at least 40% faster than baseline (3 hours)
3. WHEN processing a 1-hour 1080p 30fps video with WebGPU available, THE processing time SHALL be 30 minutes or less
4. THE frame processing throughput SHALL average at least 50 frames per second during steady-state processing on reference hardware
5. THE GPU_Pipeline SHALL process at least 90% of frames when WebGPU is available (remaining 10% accounts for initialization and fallback)
6. THE adaptive worker scaling SHALL contribute at least 10% performance improvement over fixed 4-worker baseline
7. THE parallel video decode SHALL contribute at least 5% performance improvement by reducing worker idle time
8. THE completed GPU pipeline SHALL contribute at least 25% performance improvement over hybrid GPU/CPU pipeline
9. FOR ALL processing runs on reference hardware, THE actual processing time SHALL be within 10% of predicted time based on frame count and measured per-frame latency
10. THE performance improvements SHALL apply to videos of all resolutions from 720p to 4K without regression

### Requirement 7: Backward Compatibility and Graceful Degradation

**User Story:** As a user on various browser platforms, I want the application to work reliably regardless of whether WebGPU is available, so that I can process videos on any device.

#### Acceptance Criteria

1. WHEN WebGPU is not available in the browser, THE application SHALL initialize the CPU_Pipeline using OpenCV.js WASM
2. WHEN WebGPU device initialization fails, THE application SHALL log the failure reason and fall back to CPU_Pipeline within 5 seconds
3. WHEN WebGPU device is lost during processing, THE application SHALL recover by falling back to CPU_Pipeline and resuming from the next frame
4. THE application SHALL produce pixel-accurate output on CPU_Pipeline matching baseline implementation (before optimizations)
5. WHEN adaptive worker scaling cannot adjust worker count due to memory constraints, THE Worker_Pool SHALL continue processing with the current worker count
6. WHEN parallel video decode cannot maintain the target number of concurrent video elements due to memory pressure, THE Decode_Pool SHALL continue with a reduced number of concurrent video elements
7. THE application SHALL display clear user feedback indicating which processing path is active (GPU accelerated vs CPU fallback)
8. WHEN processing a video on a system without WebGPU support, THE application SHALL still complete successfully using CPU_Pipeline with existing performance characteristics
9. THE application SHALL never crash or hang due to optimization failures (GPU device lost, worker scaling errors, prefetch failures)
10. FOR ALL error conditions in optimized code paths, THE application SHALL log diagnostic information and gracefully degrade to baseline behavior

### Requirement 8: CLAHE GPU Implementation

**User Story:** As a developer maintaining the GPU pipeline, I want CLAHE filter implemented as a WebGPU compute shader matching OpenCV behavior, so that GPU pipeline produces identical output to CPU pipeline.

#### Acceptance Criteria

1. THE CLAHE GPU compute shader SHALL divide the input image into 8×8 pixel tiles
2. THE CLAHE GPU compute shader SHALL compute histogram for each tile independently
3. THE CLAHE GPU compute shader SHALL apply clip limit to each tile histogram (clip limit range 1.5 to 6.0)
4. THE CLAHE GPU compute shader SHALL redistribute clipped histogram bins uniformly across all intensity values
5. THE CLAHE GPU compute shader SHALL compute cumulative distribution function for each clipped tile histogram
6. THE CLAHE GPU compute shader SHALL apply bilinear interpolation between adjacent tile transformations for interior pixels
7. THE CLAHE GPU compute shader SHALL apply single-tile transformation for border pixels
8. FOR ALL input images processed with identical clip limits, THE CLAHE GPU output SHALL match OpenCV CLAHE output with mean absolute error less than 3 intensity levels per pixel
9. THE CLAHE GPU compute shader SHALL process a 3840×2160 frame in 15ms or less on reference hardware
10. THE CLAHE GPU compute shader SHALL handle grayscale single-channel input images (CV_8UC1 equivalent)

### Requirement 9: Auto-Normalize GPU Implementation

**User Story:** As a developer maintaining the GPU pipeline, I want Auto_Normalize_Filter implemented as WebGPU compute shaders, so that adaptive brightness correction runs on GPU.

#### Acceptance Criteria

1. WHEN input frame mean brightness is below 80, THE Auto_Normalize_Filter GPU SHALL apply gamma lift with gamma value between 1.5 and 3.0
2. THE gamma lift GPU compute shader SHALL use a 256-entry lookup table for power-law brightness transformation
3. WHEN input frame standard deviation is below 45, THE Auto_Normalize_Filter GPU SHALL apply histogram stretch using NORM_MINMAX remapping
4. THE histogram stretch GPU compute shader SHALL remap darkest pixel to 0 and brightest pixel to 255
5. THE Auto_Normalize_Filter GPU SHALL apply adaptive CLAHE to all frames with clip limit inversely proportional to mean brightness
6. THE adaptive CLAHE clip limit SHALL be calculated as clamp(150 / mean, 1.5, 4.5)
7. THE Auto_Normalize_Filter GPU SHALL compute mean and standard deviation statistics using parallel reduction compute shader
8. FOR ALL input frames processed with Auto_Normalize_Filter, THE GPU output SHALL match CPU output with mean absolute error less than 5 intensity levels per pixel
9. THE Auto_Normalize_Filter GPU SHALL process a 3840×2160 frame including statistics computation in 20ms or less on reference hardware
10. THE Auto_Normalize_Filter GPU SHALL operate on grayscale single-channel intermediate image (after RGB to grayscale conversion)

### Requirement 10: Clean-Speckles GPU Implementation

**User Story:** As a developer maintaining the GPU pipeline, I want Clean_Speckles_Filter implemented as a WebGPU compute shader using connected component analysis, so that isolated edge fragments are removed on GPU.

#### Acceptance Criteria

1. THE Clean_Speckles_Filter GPU SHALL label connected components in the binary edge mask using parallel connected component labeling algorithm
2. THE Clean_Speckles_Filter GPU SHALL compute area statistics (pixel count) for each connected component
3. WHEN speckle intensity is 1 (fine), THE Clean_Speckles_Filter GPU SHALL remove components with area less than 4 pixels
4. WHEN speckle intensity is 2 (medium), THE Clean_Speckles_Filter GPU SHALL remove components with area less than 12 pixels
5. WHEN speckle intensity is 3 (coarse), THE Clean_Speckles_Filter GPU SHALL remove components with area less than 30 pixels
6. THE Clean_Speckles_Filter GPU SHALL preserve all pixels belonging to components meeting the minimum area threshold
7. THE Clean_Speckles_Filter GPU SHALL use 8-connectivity for component labeling (diagonal neighbors are connected)
8. FOR ALL binary edge masks processed with identical intensity settings, THE Clean_Speckles_Filter GPU output SHALL match OpenCV connectedComponentsWithStats output exactly
9. THE Clean_Speckles_Filter GPU SHALL process a 3840×2160 binary edge mask in 25ms or less on reference hardware
10. THE Clean_Speckles_Filter GPU SHALL handle edge cases (single-pixel components, components touching image borders) identically to CPU implementation

### Requirement 11: Color-Edges GPU Implementation

**User Story:** As a developer maintaining the GPU pipeline, I want Color_Edges_Filter implemented as a WebGPU compute shader, so that soft colored line rendering runs on GPU.

#### Acceptance Criteria

1. THE Color_Edges_Filter GPU SHALL apply Canny edge detection to the raw grayscale image (before CLAHE normalization) using configurable low and high thresholds
2. THE Color_Edges_Filter GPU SHALL dilate color edge mask using configurable line weight (1-5 pixels)
3. WHEN color softness is greater than 0, THE Color_Edges_Filter GPU SHALL apply Gaussian blur to source RGB image with kernel size (softness × 2 + 1)
4. WHEN color softness is greater than 0, THE Color_Edges_Filter GPU SHALL apply histogram normalization to blurred grayscale for Canny input
5. THE Color_Edges_Filter GPU SHALL sample RGB color from blurred source image at each color edge pixel location
6. WHEN color opacity is less than 1.0, THE Color_Edges_Filter GPU SHALL blend sampled color with background color using alpha blending
7. THE Color_Edges_Filter GPU SHALL output packed RGBA pixels with color edge pixels using sampled colors and background pixels using background color
8. FOR ALL frames processed with identical color edge settings, THE Color_Edges_Filter GPU output SHALL match CPU output with mean absolute error less than 3 intensity levels per channel per pixel
9. THE Color_Edges_Filter GPU SHALL process a 3840×2160 frame with color edges enabled in 30ms or less on reference hardware
10. THE Color_Edges_Filter GPU SHALL correctly composite color edges with ink edges (ink edges take priority over color edges)

### Requirement 12: Parser and Pretty Printer for Processing Settings

**User Story:** As a developer testing processing configurations, I want to serialize and deserialize processing settings to JSON, so that I can save and reload configurations for testing.

#### Acceptance Criteria

1. WHEN processing settings are provided as a JavaScript object, THE Settings_Parser SHALL parse all fields including preset, detail, lineWeight, scale, videoFps, and custom filter options
2. WHEN processing settings contain invalid values, THE Settings_Parser SHALL return descriptive validation errors indicating which fields are invalid
3. THE Settings_Pretty_Printer SHALL format processing settings JavaScript objects into valid JSON strings
4. THE Settings_Pretty_Printer SHALL include all required fields (preset, detail, lineWeight, scale) and optional fields (custom filters) in output
5. FOR ALL valid processing settings objects, parsing then printing then parsing SHALL produce an equivalent settings object (round-trip property)
6. THE Settings_Parser SHALL validate that numeric settings fall within allowed ranges (detail 1-100, lineWeight 1-5, scale 0.1-2.0)
7. THE Settings_Parser SHALL validate that preset names match known preset identifiers (manga, studio, neon, warm, vivid, blueprint, custom)
8. THE Settings_Pretty_Printer SHALL format output JSON with consistent indentation and property ordering for human readability
9. THE Settings_Parser SHALL parse settings JSON strings generated by Settings_Pretty_Printer without errors
10. THE Settings_Pretty_Printer SHALL escape special characters in string values (preset names, color hex codes) according to JSON specification

### Requirement 13: Stability Under Extreme Inputs

**User Story:** As a user processing extremely long or high-resolution videos, I want the application to remain stable without crashes or hangs, so that I can complete renders reliably.

#### Acceptance Criteria

1. WHEN processing an 8-hour 4K 60fps video, THE application SHALL complete without out-of-memory crashes
2. WHEN processing a video with rapid scene changes (new scene every 2 seconds), THE Auto_Normalize_Filter SHALL adapt per-frame without introducing visual artifacts
3. WHEN adaptive worker scaling rapidly adjusts worker count (10+ changes during render), THE Worker_Pool SHALL maintain frame processing order and never lose frames
4. WHEN WebGPU device is lost mid-render, THE application SHALL recover to CPU_Pipeline and resume processing from the next frame without data loss
5. WHEN system memory pressure causes browser to reclaim memory, THE application SHALL gracefully reduce the number of concurrent decode video elements and continue processing
6. WHEN user reduces worker count manually during processing, THE Worker_Pool SHALL immediately scale down without dropping in-flight frames
7. WHEN processing a video with variable frame rate or resolution changes, THE Video_Decoder SHALL adapt and continue without errors
8. THE application SHALL process videos of any length up to browser file size limits without hard-coded duration caps
9. WHEN processing a video with corrupt frames or decode errors, THE application SHALL log the error, skip the corrupt frame, and continue with the next valid frame
10. FOR ALL processing runs lasting longer than 2 hours, THE application SHALL maintain stable memory usage without unbounded growth

### Requirement 14: Human Subject Line-Art Quality in Custom Mode

**User Story:** As a user processing videos of human subjects in Custom/Experiment mode, I want precise control over edge detection thresholds and smoothing filters, so that facial features and body contours are captured cleanly without shadow-induced artifacts creating unwanted lines.

#### Acceptance Criteria

1. WHEN processing a video frame containing human subjects in Custom mode, THE edge detection algorithm SHALL distinguish between true edge contours (facial features, body outlines) and false edges created by shadows
2. WHEN the user adjusts Ink low threshold in Custom mode, THE application SHALL update edge detection sensitivity in real-time with values ranging from 5 to 150
3. WHEN the user adjusts Ink high threshold in Custom mode, THE application SHALL update edge retention strength in real-time with values ranging from 20 to 255
4. WHEN the user enables Bilateral smooth in Custom mode, THE filter SHALL preserve sharp edges at facial feature boundaries while smoothing uniform skin tone regions
5. WHEN the user adjusts Bilateral passes in Custom mode (1-5 passes), THE application SHALL apply multiple smoothing iterations without destroying thin subject lines
6. WHEN the user adjusts Edge detail slider (35-90), THE threshold scaling factor SHALL inversely scale with detail value to provide finer control over line density
7. FOR ALL human subject videos processed in Custom mode with optimal settings, THE output line-art SHALL capture facial features (eyes, nose, mouth, eyebrows) as continuous clean lines without fragmentation
8. FOR ALL human subject videos processed in Custom mode, THE output line-art SHALL NOT include spurious lines caused by soft shadows on skin, clothing folds, or background gradients
9. WHEN the user enables Clean speckles in Custom mode, THE filter SHALL remove isolated noise dots while preserving fine hair strands and eyelashes
10. THE application SHALL provide preset starting values for human subject processing in Custom mode: Ink low threshold 40, Ink high threshold 100, Bilateral diameter 13, Sigma 90, Clean speckles enabled

### Requirement 15: Consistent Line-Art Aesthetic Across All Video Types

**User Story:** As a user processing various video types (wildlife, people, plants, urban scenes), I want the output to maintain a consistent clean line-art aesthetic regardless of input content, so that all renders have a unified artistic style.

#### Acceptance Criteria

1. FOR ALL video types processed with the same preset (Manga, Studio, Neon, Warm, Vivid, Blueprint), THE output SHALL exhibit consistent line weight, edge density, and background treatment
2. WHEN processing videos with varying lighting conditions (bright outdoor, dim indoor, mixed lighting), THE Auto_Normalize_Filter SHALL adapt per-frame to maintain consistent line visibility across scenes
3. WHEN processing videos with high-contrast subjects (dark clothing on light background or vice versa), THE edge detection SHALL produce line-art with balanced ink distribution
4. WHEN processing videos with low-contrast subjects (gray clothing on gray background), THE histogram stretch SHALL enhance edge visibility without introducing noise artifacts
5. FOR ALL processed videos, THE output SHALL resemble hand-drawn line-art with clean continuous strokes rather than photographic edge maps
6. WHEN processing videos with textured surfaces (fabric patterns, tree bark, brick walls), THE Clean speckles filter SHALL remove micro-texture noise while retaining macro-structure outlines
7. WHEN processing videos with motion blur, THE edge detection SHALL extract subject outlines without creating double-edge ghost artifacts
8. FOR ALL standard presets, THE bilateral smoothing parameters SHALL be tuned to produce cartoon-style flat regions bounded by clean ink lines
9. WHEN the user switches between presets during preview, THE line-art aesthetic SHALL transition smoothly without introducing discontinuities or visual inconsistencies
10. THE Merge double-edges filter SHALL consolidate parallel edge pairs from thick features into single clean lines, maintaining the line-art aesthetic

### Requirement 16: Color Lines Preservation During Normalization

**User Story:** As a user enabling Color lines (soft edges) in Custom mode with auto-normalize or CLAHE boost, I want colored lines to remain visible and properly colored, so that facial features and textures retain their original color information after brightness correction.

#### Acceptance Criteria

1. WHEN Auto-normalize frames is enabled AND Color lines checkbox is enabled, THE color sampling for soft edges SHALL occur on the raw RGB image before any normalization filters are applied
2. WHEN CLAHE boost is enabled AND Color lines checkbox is enabled, THE color sampling SHALL use the original RGB pixel values unaffected by CLAHE intensity adjustments
3. WHEN Gamma lift is applied to grayscale for edge detection AND Color lines is enabled, THE RGB source image for color sampling SHALL remain at original brightness levels
4. WHEN Histogram stretch is applied to grayscale for edge detection AND Color lines is enabled, THE paint colors SHALL retain original saturation and hue
5. FOR ALL frames processed with Auto-normalize and Color lines both enabled, THE colored soft edges SHALL remain fully visible with color opacity matching the Color opacity slider setting (0-100%)
6. WHEN the user adjusts CLAHE clip limit (1-6) with Color lines enabled, THE colored edges SHALL maintain consistent visibility regardless of clip limit value
7. THE Color edges Canny detection SHALL operate on the raw grayscale image (grayRaw) that has NOT been processed by Auto_Normalize_Filter
8. WHEN Color softness is greater than 0, THE Gaussian blur for color sampling SHALL be applied independently to the RGB source image without affecting the normalized grayscale used for ink edges
9. FOR ALL test videos of human subjects with Color lines and Auto-normalize enabled, THE facial features (lips, eyebrows, eye color) SHALL render in their original colors without brightness or saturation shifts
10. THE application SHALL maintain separate image buffers for normalized grayscale (for ink edges) and raw RGB (for color sampling) throughout the processing pipeline

### Requirement 17: Custom Mode Parameter Validation and Testing

**User Story:** As a developer testing Custom mode configurations, I want comprehensive validation of all filter parameter combinations, so that the application prevents invalid settings and provides meaningful feedback.

#### Acceptance Criteria

1. WHEN the user enables both Bilateral smooth and Gaussian smooth in Custom mode, THE application SHALL apply Bilateral filter first on RGB, then Gaussian filter on grayscale
2. WHEN the user enables Median smooth in Custom mode, THE filter SHALL execute after Bilateral and Gaussian filters in the processing chain
3. WHEN the user disables all smoothing filters (Bilateral, Gaussian, Median) in Custom mode, THE edge detection SHALL operate on the raw RGB to grayscale conversion without smoothing
4. WHEN the user sets Ink low threshold greater than Ink high threshold, THE application SHALL display a validation warning and auto-adjust high threshold to be at least 24 units above low threshold
5. WHEN the user enables Color lines with Color low threshold greater than Color high threshold, THE application SHALL display a validation warning and auto-adjust color high threshold accordingly
6. WHEN the user sets Color line weight to maximum (5) with Color opacity at 100%, THE colored edges SHALL completely obscure the background without transparency artifacts
7. FOR ALL Custom mode parameter combinations (2^10 possible toggle combinations), THE processing pipeline SHALL complete without crashes or hangs
8. WHEN the user enables all filters simultaneously (Bilateral, Gaussian, Median, Clean speckles, Auto-normalize, Dark boost, Merge double-edges, Color lines), THE Frame_Processing_Latency SHALL remain under 200ms per 1080p frame on reference hardware
9. THE application SHALL save Custom mode parameters to browser localStorage and restore them on page reload
10. THE application SHALL provide a "Reset to Default" button in Custom mode that restores the baseline custom preset values (white background, black ink, threshold 60/180, bilateral diameter 13, sigma 90)

## Property-Based Testing Guidance

The following properties should be validated through property-based testing:

### Output Quality Equivalence (CRITICAL - Cross-Platform)

**Property**: GPU and CPU pipelines produce pixel-perfect equivalent output across all platforms
- **Invariant**: For any valid input frame and settings, `maxPixelDifference(gpuOutput, cpuOutput) <= 1` intensity level
- **Test Strategy**: Generate random frames (varying sizes, brightness, contrast, content) and process through both pipelines with identical settings on Windows, macOS, Linux
- **Oracle**: Maximum absolute pixel error per channel, SSIM structural similarity metric (must be > 0.99)
- **Platform Coverage**: Execute on Chrome/Windows, Chrome/macOS, Chrome/Linux, Edge/Windows, Firefox/Linux to verify cross-platform consistency

### Human Subject Line Quality

**Property**: Shadow gradients on human skin do not generate edge artifacts
- **Invariant**: For frames with human subjects, `spuriousEdgeCount(shadowRegions) === 0`
- **Test Strategy**: Process test videos of human faces with varying lighting (side-lit, backlit, spotlight) at different Custom mode threshold settings
- **Oracle**: Manual visual inspection + automated edge density analysis in known shadow regions
- **Metamorphic**: Increasing Bilateral smoothing passes SHALL monotonically decrease edge density in shadow gradient regions

### Line-Art Aesthetic Consistency

**Property**: Output maintains clean line-art style regardless of input content type
- **Invariant**: For any video type, `lineArtQualityScore(output) > 0.85` (continuous strokes, minimal noise, balanced ink distribution)
- **Test Strategy**: Process diverse videos (wildlife, people, plants, urban) with the same preset, measure line continuity and noise ratio
- **Oracle**: Custom metric combining: edge continuity (fewer breaks), speckle ratio (lower is better), line weight uniformity

### Color Lines Preservation Under Normalization

**Property**: Color lines remain visible and properly colored when normalization is applied
- **Invariant**: When `autoNormalize === true AND colorEdges === true`, then `colorLineVisibility > 0.9 AND colorAccuracy > 0.95`
- **Test Strategy**: Process frames with color lines enabled, compare color sampling before and after normalization
- **Oracle**: Color difference deltaE < 5 for colored edge pixels, opacity retention >= 95% of slider setting
- **Metamorphic**: Applying CLAHE boost SHALL NOT reduce color line opacity or shift hue

### Memory Bounds Preservation

**Property**: Peak memory usage never exceeds calculated budget
- **Invariant**: `peakMemory <= baseMemory + (workerCount × 80MB) + (prefetchDepth × frameSize)`
- **Test Strategy**: Monitor memory usage during processing of videos of varying lengths with different worker counts
- **Oracle**: Browser memory profiling API (performance.memory)

### Processing Order Correctness

**Property**: Output frames maintain input sequence order
- **Invariant**: `outputFrame[i].index === i` for all frames in output video
- **Test Strategy**: Process video with frame index watermarking and verify output sequence
- **Oracle**: Frame index embedded in processed output

### Adaptive Scaling Stability

**Property**: Worker pool scaling never causes deadlock or frame loss
- **Invariant**: `framesProcessed === totalFrames` regardless of scaling events
- **Test Strategy**: Simulate varying system load and memory pressure during processing
- **Oracle**: Frame count verification and processing completion

### Round-Trip Settings Preservation

**Property**: Settings serialization and deserialization are inverses
- **Invariant**: `parse(print(settings)) === settings` (deep equality)
- **Test Strategy**: Generate random valid settings objects, serialize, deserialize, compare
- **Oracle**: Deep object equality comparison

### Parallel Decode Efficiency

**Property**: Worker idle time decreases as the decode-pool size (number of concurrent video elements) increases
- **Metamorphic**: `workerIdleTime(decodePoolSize=S) > workerIdleTime(decodePoolSize=L)` for L > S
- **Test Strategy**: Process same video with varying decode-pool sizes (number of concurrent video elements), measure worker idle percentage
- **Oracle**: Performance monitor worker utilization metrics

### GPU Fallback Correctness

**Property**: CPU fallback produces identical functional behavior
- **Invariant**: When GPU fails, processing continues and completes successfully
- **Test Strategy**: Force GPU device loss at random points during processing
- **Oracle**: Processing completion status and output frame count

## Notes

1. **Parser/Serializer Requirements**: Settings parsing and pretty-printing are essential for testing and configuration persistence. Round-trip property testing ensures serialization correctness.

2. **Reference Hardware**: Performance targets assume NVIDIA RTX 3060 (or AMD/Intel equivalent) with 16GB system RAM and 8-core CPU. Actual performance will scale with hardware capabilities.

3. **Existing Stability**: All memory-safety and streaming architecture requirements preserve the current implementation's proven stability with multi-hour 4K videos. These requirements serve as regression tests.

4. **WebGPU Availability**: As of 2024, WebGPU is available in Chrome 113+, Edge 113+, and experimental in Firefox. Safari support is in development. CPU fallback ensures compatibility.

5. **Property-Based Testing Priority**: Focus PBT on:
   - **GPU/CPU output equivalence** (Requirement 1) - highest priority for correctness
   - **Cross-platform consistency** - verify identical output on Windows, macOS, Linux
   - **Human subject quality** (Requirement 14) - ensure shadows don't create unwanted lines
   - **Color lines preservation** (Requirement 16) - verify normalization doesn't affect color sampling
   - **Memory bounds preservation** (Requirement 5) - prevent regressions to stability

6. **Human Subject Processing**: The Custom/Experiment mode is the primary testing ground for human subject videos. Optimal settings for clean facial features: Ink low threshold 35-45, Ink high threshold 90-110, Bilateral smooth enabled with 2-3 passes, Clean speckles enabled at intensity 2-3, Edge detail 55-65.

7. **Color Lines Architecture**: The requirement for separate image buffers (normalized grayscale for ink edges, raw RGB for color sampling) is critical to prevent color line disappearance when normalization is applied. This architectural constraint must be preserved in both GPU and CPU pipelines.

8. **Line-Art Aesthetic**: The application's core value proposition is producing clean line-art that resembles hand-drawn illustration, not photographic edge detection. All filters (bilateral smooth, merge double-edges, clean speckles) work together to achieve this aesthetic. Requirement 15 ensures this aesthetic is maintained across diverse input content.

9. **Cross-Platform Testing**: Given the importance of consistent output across platforms (Requirement 1, criterion 9), integration tests must verify pixel-perfect equivalence on Windows, macOS, and Linux for both GPU and CPU paths. Use reference test videos with known-good output frames as golden masters.

10. **Custom Mode Complexity**: With 10+ toggleable filters and 15+ adjustable parameters, Custom mode has over 1000 possible configurations. Property-based testing should focus on: filter ordering correctness, parameter boundary conditions, and the most common user workflows (human subjects with normalization, high-contrast scenes with bilateral smooth, low-light videos with CLAHE boost).
