# Requirements Document

## Introduction

The Cartoon Quality Pipeline adds an optional, quality-first line-art mode to the existing
browser-based Line Art Animator. The feature is gated behind a single "Cartoon Quality" toggle that
is off by default. When the toggle is off, the application behaves exactly as it does today: existing
presets, the GPU/CPU worker pool, preview, and FFmpeg-WASM video export are completely unchanged.
When the toggle is on, frames are routed through a new cartoon pipeline tuned for crisp,
hand-drawn-style human line art.

The cartoon pipeline runs entirely client-side (no backend, no uploads) and is composed of four
core stages: bilateral filtering, adaptive thresholding, human-segmentation guidance via a soft alpha
mask, and optional vectorization. Layered on top of segmentation is an optional, additive fifth stage
— pose and hand-landmark guidance — that performs region-aware detail boosting. A pose model and a
hand-landmark model run client-side to produce sparse anatomical keypoints (not edges and not a
silhouette). Those keypoints are splatted along an anatomically correct skeleton topology into a
low-resolution detail-boost map that marks where small, detail-dense anatomy lives (hands, fingers,
joints, limbs), so that finer line work is applied locally in those regions. The keypoint stage is
complementary to segmentation and never a replacement for it: segmentation only fades background
edges, while keypoint guidance only boosts local detail inside the subject. The pipeline selects an
execution tier based on detected browser capabilities (WebGPU, WebGL2 + WASM SIMD, or an OpenCV.js
CPU floor) and degrades gracefully so that rendering never crashes. All new parameters are additive
to the existing settings schema and flow through the existing serialization, validation, and
persistence mechanisms.

These requirements are derived from the approved design document and are intended to capture the
design intent for the additive cartoon mode while guaranteeing zero regression of existing behavior.

## Glossary

- **Line_Art_Animator**: The existing browser-based application that converts images and video frames into line art.
- **Standard_Pipeline**: The existing WebGPU compute pipeline plus the OpenCV.js CPU fallback used when Cartoon Mode is off.
- **Cartoon_Pipeline**: The new opt-in processing pipeline activated when Cartoon Mode is on.
- **Cartoon_Mode**: The boolean state of the "Cartoon Quality" toggle, carried on the settings object as `cartoonMode`.
- **Cartoon_Pipeline_Controller**: The main-thread orchestrator that coordinates capability detection, resolution selection, segmentation, edge extraction, compositing, and vectorization for one frame.
- **Capability_Detector**: The component that probes WebGPU, WebGL2, and WASM SIMD availability to select an execution tier.
- **Edge_Shader_Stage**: The component that performs bilateral filtering followed by adaptive thresholding to produce an edge-intensity layer.
- **Segmentation_Worker**: The Web Worker that runs the background-removal model to produce a soft human-probability mask.
- **Compositor**: The component that upscales and blurs the segmentation mask and multiplies it over the edge layer.
- **Vectorization_Worker**: The Web Worker that traces contours, simplifies them, smooths them into Bézier curves, and rasterizes the result to RGBA.
- **Resolution_Policy**: The pure module that decides processing resolution, segmentation resolution, and FPS cap.
- **Settings_Parser**: The existing component that parses and validates serialized settings strings.
- **Settings_Pretty_Printer**: The existing component that serializes settings objects to strings.
- **Soft_Alpha_Mask**: An upscaled and blurred segmentation probability map in the range `[0,1]`, multiplied over edges.
- **Execution_Tier**: One of Tier A (WebGPU), Tier B (WASM + SIMD with WebGL2 edges), or Tier C (OpenCV.js CPU floor).
- **Degraded_Mode**: Operation in which segmentation is skipped and full-frame edges are produced without segmentation guidance.
- **RGBA_Frame**: A `Uint8ClampedArray` of pixel data with the same width and height as the input frame.
- **Segmentation_Model**: A quantized background-removal model (RMBG-1.5 or BiRefNet-mini) under 50 MB, lazy-loaded on first cartoon activation.
- **Keypoint_Guidance_Stage**: The optional, additive stage that runs pose and hand-landmark inference and builds the Detail_Boost_Map for region-aware detail boosting. It is layered on top of segmentation and is complementary to it, never a replacement for it.
- **Keypoint_Guidance_Worker**: The Web Worker that runs the Pose_Model and Hand_Model to produce pose and hand landmarks.
- **Pose_Model**: A quantized full-body pose model (e.g. BlazePose 33-point or MoveNet 17-point) lazy-loaded on first keypoint-guidance activation.
- **Hand_Model**: A quantized hand-landmark model that produces 21 landmarks per detected hand, lazy-loaded on first keypoint-guidance activation.
- **Pose_Landmark**: A full-body joint keypoint with normalized coordinates `(x, y)` in `[0,1]` and a confidence `score`.
- **Hand_Landmark**: One of the 21 keypoints of a detected hand, with normalized coordinates `(x, y)` in `[0,1]` and a confidence `score`.
- **Hand**: A detected hand consisting of a `handedness` label of `Left` or `Right`, a confidence `score`, and exactly 21 Hand_Landmark values.
- **Keypoint**: A Pose_Landmark or a Hand_Landmark; a sparse anatomical point, never an edge or a silhouette.
- **Skeleton_Topology**: The fixed, anatomically correct connectivity (bone list) linking pose joints (`POSE_BONES`) and hand landmarks (`HAND_BONES`).
- **Bone**: A segment between two connected landmarks defined in the Skeleton_Topology; detail is splatted along bones, never between unconnected landmarks.
- **Detail_Boost_Map**: A low-resolution scalar field with every value in `[0,1]` that marks where finer line work is applied, built by splatting confident keypoints and bones; an all-zero map denotes the segmentation-only fallback.
- **Keypoint_Confidence**: The minimum landmark score, carried as `keypointConfidence`, at or above which a landmark counts as present and may contribute boost.
- **Segmentation_Only_Result**: The Cartoon_Pipeline output for a frame when no Detail_Boost_Map is applied (the Detail_Boost_Map is all-zero), used as the well-defined fallback for the Keypoint_Guidance_Stage.

## Requirements

### Requirement 1: Optional Cartoon Mode Toggle

**User Story:** As a user, I want a single Cartoon Quality toggle that is off by default, so that the existing application behavior is preserved unless I explicitly opt in.

#### Acceptance Criteria

1. WHEN the Line_Art_Animator initializes without a persisted Cartoon_Mode value, THE Line_Art_Animator SHALL set Cartoon_Mode to off.
2. WHILE Cartoon_Mode is off, THE Line_Art_Animator SHALL produce output byte-identical to the pre-feature Standard_Pipeline for the same input frame and settings.
3. WHEN a user enables Cartoon_Mode, THE Line_Art_Animator SHALL route every frame submitted after the toggle change through the Cartoon_Pipeline.
4. WHEN a user disables Cartoon_Mode, THE Line_Art_Animator SHALL restore Standard_Pipeline processing before the next frame is submitted.
5. WHEN a user disables Cartoon_Mode, THE Line_Art_Animator SHALL release Cartoon_Pipeline resources before the next frame is submitted.
6. WHILE Cartoon_Mode is off, THE Line_Art_Animator SHALL NOT request the Segmentation_Model.
7. WHILE Cartoon_Mode is off, THE Line_Art_Animator SHALL NOT load cartoon-specific workers.

### Requirement 2: Pipeline Routing

**User Story:** As a developer, I want frame rendering to branch cleanly on the Cartoon Mode state, so that both pipelines remain independent and the existing render entry point is reused.

#### Acceptance Criteria

1. WHEN a frame is submitted for rendering AND Cartoon_Mode is true, THE Line_Art_Animator SHALL delegate the frame to the Cartoon_Pipeline_Controller and SHALL NOT dispatch the same frame through the Standard_Pipeline worker pool.
2. WHEN a frame is submitted for rendering AND Cartoon_Mode is false, THE Line_Art_Animator SHALL dispatch the frame through the existing Standard_Pipeline worker pool and SHALL NOT invoke the Cartoon_Pipeline_Controller.
3. WHEN the Cartoon_Pipeline_Controller completes processing of a submitted frame, THE Cartoon_Pipeline_Controller SHALL return an RGBA_Frame whose width and height each equal the width and height of the submitted input frame.
4. WHEN the Cartoon_Pipeline_Controller completes processing of a submitted frame, THE Cartoon_Pipeline_Controller SHALL return a result containing a boolean field indicating whether GPU processing was used and a boolean field indicating whether processing ran in Degraded_Mode.
5. WHEN frames are submitted for rendering, THE Line_Art_Animator SHALL emit the corresponding rendered frames in the same order in which they were submitted, regardless of whether the Cartoon_Pipeline_Controller or the Standard_Pipeline worker pool processed each frame.

### Requirement 3: Bilateral Filtering Stage

**User Story:** As a user, I want micro-textures removed while structural boundaries stay sharp, so that the resulting line art is clean and hand-drawn in style.

#### Acceptance Criteria

1. WHEN the Cartoon_Pipeline processes a frame, THE Edge_Shader_Stage SHALL apply a bilateral filter before adaptive thresholding.
2. THE Edge_Shader_Stage SHALL map `bilateralStrength` in the range `[0,100]` to the bilateral filter smoothing amount as a monotonically non-decreasing relationship, and `bilateralRadius` in the range `[1,7]` to the bilateral filter window size as a monotonically non-decreasing relationship.
3. WHERE WebGPU is the selected Execution_Tier, THE Edge_Shader_Stage SHALL execute the bilateral filter as a WGSL compute pass.
4. WHERE WebGL2 is the selected Execution_Tier, THE Edge_Shader_Stage SHALL execute the bilateral filter as a GLSL ES 3.0 fragment shader.
5. WHERE the CPU floor is the selected Execution_Tier, THE Edge_Shader_Stage SHALL execute the bilateral filter using OpenCV.js.
6. THE Edge_Shader_Stage SHALL produce bilateral output values in the normalized greyscale range `[0,1]`, bounded within the minimum and maximum greyscale values of the filter window centered on the pixel.
7. WHERE `bilateralStrength` is 0, THE Edge_Shader_Stage SHALL produce bilateral output equal to the input greyscale value at every pixel.
8. IF the bilateral filter stage fails for a frame, THEN THE Edge_Shader_Stage SHALL signal the failure to the Cartoon_Pipeline_Controller and SHALL NOT emit partial output.

### Requirement 4: Adaptive Thresholding Stage

**User Story:** As a user, I want local lighting normalized and temporal line flicker reduced, so that video line art stays stable across frames.

#### Acceptance Criteria

1. WHEN the bilateral output is available, THE Edge_Shader_Stage SHALL apply adaptive thresholding to produce an edge-intensity layer in the range `[0,1]`.
2. THE Edge_Shader_Stage SHALL use `adaptiveBlockSize` as the side length of a centered square local window when computing the local threshold.
3. THE Edge_Shader_Stage SHALL apply `adaptiveC` in the range `[-20,20]` as the threshold bias subtracted from the local mean.
4. THE Edge_Shader_Stage SHALL produce edge intensity as a continuous, monotonically non-decreasing function of the difference between the bilateral value and the local threshold over a nonzero band centered on the threshold, such that as the per-pixel input difference between two frames approaches zero, the edge-intensity difference approaches zero.
5. IF `adaptiveBlockSize` is not an odd integer in the range `[9,151]`, THEN THE Settings_Parser SHALL reject the value with an error identifying the invalid parameter and its required range, and SHALL leave prior settings unchanged.

### Requirement 5: Human Segmentation Guidance

**User Story:** As a user, I want line art emphasis guided toward human subjects, so that background clutter is faded without hard cropping.

#### Acceptance Criteria

1. WHILE segmentation is enabled, WHEN the Segmentation_Worker completes inference for a frame, THE Segmentation_Worker SHALL produce a single-channel human-probability mask with every value bounded in the range `[0,1]`.
2. THE Segmentation_Worker SHALL run inference at the Segmentation_Model native resolution rather than the full processing resolution.
3. WHEN the human-probability mask is available, THE Compositor SHALL upscale the mask to the processing resolution and blur it using `segSoftness` in the range `[0,40]` as the blur radius to form the Soft_Alpha_Mask, with every resulting value bounded in the range `[0,1]` and with a `segSoftness` of 0 applying no blur to the upscaled mask.
4. THE Cartoon_Pipeline SHALL apply the Soft_Alpha_Mask as a soft spatial weight multiplied over the full-frame edge layer rather than as a hard crop.
5. WHEN segmentation guidance is applied with `segFadeStrength` in the range `[0,1]`, THE Compositor SHALL scale edge intensity at each pixel by a fade factor that equals `(1 - segFadeStrength)` where the Soft_Alpha_Mask value is 0, equals 1 where the Soft_Alpha_Mask value is 1, and varies monotonically with the mask value between those bounds.
6. WHILE segmentation is disabled, THE Cartoon_Pipeline SHALL apply the full-frame edge layer without a Soft_Alpha_Mask and SHALL NOT fade background edge intensity.

### Requirement 6: Soft-Alpha Compositing Invariant

**User Story:** As a user, I want segmentation to only suppress background detail and never amplify it, so that subject line work is preserved while background noise is reduced.

#### Acceptance Criteria

1. WHEN the Compositor combines the edge layer, whose values lie in the range `[0,1]`, with the Soft_Alpha_Mask, THE Compositor SHALL produce at every pixel a composite edge value that does not exceed the input edge value by more than an absolute tolerance of `1/255`.
2. WHERE the Soft_Alpha_Mask value equals `1` at a pixel, THE Compositor SHALL produce a composite edge value equal to the input edge value at that pixel within an absolute tolerance of `1/255`.
3. IF two pixels have equal input edge values but different Soft_Alpha_Mask values, THEN THE Compositor SHALL produce, for the pixel with the larger mask value, a composite edge value greater than or equal to the composite edge value of the pixel with the smaller mask value, within an absolute tolerance of `1/255`.
4. THE Compositor SHALL produce composite edge values in the range `[0,1]`.

### Requirement 7: Vectorization

**User Story:** As a user, I want edges converted into smooth vector-style strokes, so that the output looks like crisp illustrator curves.

#### Acceptance Criteria

1. WHILE vectorization is enabled, THE Vectorization_Worker SHALL binarize the edge-intensity layer by treating every pixel with intensity greater than or equal to 0.5 as an edge pixel, trace contours of the resulting binary edge mask using Marching Squares, simplify each traced contour using Ramer-Douglas-Peucker, and smooth the simplified contours into cubic Bézier curves.
2. THE Vectorization_Worker SHALL produce a simplified path that is a subsequence of the traced path, preserves both endpoints, and where every removed point lies within `rdpEpsilon` perpendicular distance of the retained polyline.
3. THE Vectorization_Worker SHALL produce cubic Bézier segments whose endpoints equal the retained input points so that the smoothed curve passes through every retained vertex, and SHALL scale each segment's control-point tangent length by `bezierSmoothing` in the range `[0,1]`, where 0 yields straight segments between retained vertices and 1 yields maximum curvature.
4. WHEN the smoothed curves are produced, THE Vectorization_Worker SHALL rasterize them into an RGBA_Frame whose width and height equal the input frame, using `cartoonLineWeight` in the range `[1,5]` as the stroke width in pixels.
5. THE Vectorization_Worker SHALL execute contour tracing, simplification, smoothing, and rasterization on a Web Worker thread separate from the user interface thread.
6. WHEN the binary edge mask contains no traceable contours, THE Vectorization_Worker SHALL produce an RGBA_Frame matching the input frame dimensions with no rendered strokes.
7. IF vectorization throws an error or does not complete within 2000 milliseconds for a frame, THEN THE Cartoon_Pipeline_Controller SHALL fall back to the rasterized edge layer for that frame, preserve the rasterized edge output unchanged, and report the fallback in the frame result.

### Requirement 8: Resolution and FPS Policy

**User Story:** As a user on any device, I want processing resolution capped to a sustainable budget, so that rendering stays smooth and memory stays bounded.

#### Acceptance Criteria

1. THE Resolution_Policy SHALL return a processing resolution with width less than or equal to 1280 pixels and height less than or equal to 720 pixels for any input frame and platform.
2. WHERE the platform is mobile, THE Resolution_Policy SHALL return a processing resolution whose longest edge is less than or equal to 854 pixels whether `qualityProfile` is `maxQuality` or `balanced`.
3. THE Resolution_Policy SHALL never return a processing resolution whose width exceeds the source width or whose height exceeds the source height.
4. THE Resolution_Policy SHALL return a segmentation input resolution equal to the Segmentation_Model native size.
5. WHEN the processing resolution is reduced below the source dimensions, THE Resolution_Policy SHALL preserve the source aspect ratio within a rounding tolerance of 1 pixel on each edge.
6. WHERE the platform is desktop, THE Resolution_Policy SHALL return an FPS cap of 30 frames per second when `qualityProfile` is `balanced` and an FPS cap of 24 frames per second when `qualityProfile` is `maxQuality`.
7. WHERE the platform is mobile, THE Resolution_Policy SHALL return an FPS cap of 24 frames per second when `qualityProfile` is `balanced` and an FPS cap of 15 frames per second when `qualityProfile` is `maxQuality`.

### Requirement 9: Capability Detection and Tiered Fallback

**User Story:** As a user, I want the cartoon pipeline to work across desktop and mobile browsers, so that the feature degrades gracefully instead of failing.

#### Acceptance Criteria

1. WHEN Cartoon_Mode is enabled, THE Capability_Detector SHALL probe WebGPU, WebGL2, and WASM SIMD availability and SHALL complete all probes within 3 seconds.
2. IF a WebGPU adapter and device are successfully obtained, THEN THE Cartoon_Pipeline SHALL select Tier A using a WebGPU inference provider and WGSL compute edge shaders.
3. IF WebGPU is unavailable AND a WebGL2 rendering context can be created AND WASM SIMD feature detection returns true, THEN THE Cartoon_Pipeline SHALL select Tier B using a WASM inference provider and WebGL2 fragment edge shaders.
4. IF neither WebGPU nor the combination of WebGL2 and WASM SIMD is available, THEN THE Cartoon_Pipeline SHALL select Tier C using OpenCV.js for bilateral filtering and adaptive thresholding and SHALL skip segmentation.
5. WHERE `crossOriginIsolated` is false, THE Segmentation_Worker SHALL use single-threaded SIMD WASM rather than multi-threaded WASM.
6. IF a capability probe throws an error or does not complete within the 3-second probe budget, THEN THE Capability_Detector SHALL record that capability as unavailable.
7. WHEN capability probing completes, THE Cartoon_Pipeline SHALL select exactly one Execution_Tier among Tier A, Tier B, and Tier C.

### Requirement 10: GPU-Resident Data Path

**User Story:** As a user, I want the cartoon pipeline to avoid visible stalls and flicker, so that playback stays smooth.

#### Acceptance Criteria

1. WHERE Tier A is selected AND the segmentation output tensor is a GPU buffer, THE Cartoon_Pipeline SHALL pass the segmentation mask to the Compositor as a GPU resource with zero CPU readbacks for that frame.
2. IF GPU output binding is unavailable, THEN THE Segmentation_Worker SHALL read the mask back to the CPU exactly once per frame at the Segmentation_Model native resolution.
3. WHERE Tier A is selected, THE Cartoon_Pipeline SHALL upload the input frame to a GPU texture exactly once per frame and share it between the segmentation pre-process and the Edge_Shader_Stage.
4. IF a GPU texture upload or buffer binding fails on Tier A, THEN THE Cartoon_Pipeline_Controller SHALL fall back to a CPU data path, produce a valid RGBA_Frame, and report the fallback in its frame result.

### Requirement 11: Settings Schema Extension and Serialization

**User Story:** As a user, I want my cartoon settings saved and restored reliably, so that my configuration persists across sessions and exports.

#### Acceptance Criteria

1. WHERE Cartoon_Mode is true, THE Settings_Pretty_Printer SHALL emit the cartoon parameter block in the serialized settings.
2. WHERE Cartoon_Mode is false, THE Settings_Pretty_Printer SHALL produce serialized settings byte-identical to the pre-feature schema.
3. WHEN a valid settings object is serialized and then parsed, THE Settings_Parser SHALL produce an object that deep-equals the original settings object after `adaptiveBlockSize` is normalized to an odd integer, with every other parameter value preserved exactly.
4. WHEN a pre-feature settings string is parsed, THE Settings_Parser SHALL succeed and produce a settings object whose `cartoonMode` resolves to false.
5. WHILE Cartoon_Mode is true, THE Settings_Parser SHALL validate each cartoon parameter against the range defined in Requirement 12, and IF any cartoon parameter is out of range or of the wrong type, THEN THE Settings_Parser SHALL reject the parse with an error indicating the offending parameter name and SHALL leave previously persisted settings unchanged.
6. WHILE Cartoon_Mode is false, THE Settings_Parser SHALL ignore cartoon parameter values.
7. IF a settings string is malformed or cannot be parsed, THEN THE Settings_Parser SHALL reject with an error indicating the parse failure and SHALL retain the previously persisted settings.

### Requirement 12: Parameter Validation Ranges

**User Story:** As a user, I want invalid cartoon parameters rejected before processing, so that the shaders and algorithms receive only safe values.

#### Acceptance Criteria

1. THE Settings_Parser SHALL accept `bilateralStrength` only within the inclusive range `[0,100]`, `bilateralRadius` only within the inclusive range `[1,7]`, and `adaptiveC` only within the inclusive range `[-20,20]`.
2. THE Settings_Parser SHALL accept `segSoftness` only within the inclusive range `[0,40]` and `segFadeStrength` only within the inclusive range `[0,1]`.
3. THE Settings_Parser SHALL accept `rdpEpsilon` only within the inclusive range `[0.5,8]`, `bezierSmoothing` only within the inclusive range `[0,1]`, and `cartoonLineWeight` only within the inclusive range `[1,5]`.
4. THE Settings_Parser SHALL accept `qualityProfile` only when it exactly matches the string `maxQuality` or the string `balanced`.
5. WHILE Cartoon_Mode is true, THE Settings_Parser SHALL accept `poseHandEnabled` only when its value is a boolean, `handDetailStrength` only within the inclusive range `[0,1]`, `jointDetailStrength` only within the inclusive range `[0,1]`, and `keypointConfidence` only within the inclusive range `[0,1]`.
6. THE Line_Art_Animator SHALL normalize `adaptiveBlockSize` to the nearest odd integer within the range `[9,151]`, rounding ties upward, and repeated normalization SHALL yield the same value as a single normalization.
7. IF a cartoon parameter is out of its defined range, non-numeric, or NaN, or `qualityProfile` is an unrecognized value, or `poseHandEnabled` is not a boolean, THEN THE Settings_Parser SHALL reject the value with a descriptive error identifying the parameter and SHALL leave the last valid persisted value unchanged.

### Requirement 13: Graceful Degradation

**User Story:** As a user, I want rendering to keep working even when capabilities or model loading fail, so that preview and export never crash.

#### Acceptance Criteria

1. IF a capability probe, model download, or inference session creation fails, THEN THE Cartoon_Pipeline_Controller SHALL enter Degraded_Mode and produce full-frame edges without segmentation guidance.
2. THE Cartoon_Pipeline_Controller SHALL resolve every frame to a valid RGBA_Frame and SHALL NOT reject to the render loop.
3. WHEN processing runs in Degraded_Mode, THE Cartoon_Pipeline_Controller SHALL report the degraded state in its frame result.
4. IF the GPU device is lost mid-render, THEN THE Cartoon_Pipeline_Controller SHALL dispose the GPU stage and re-initialize at a lower Execution_Tier on the next frame.
5. IF a user disables Cartoon_Mode while an asynchronous load is in progress, THEN THE Cartoon_Pipeline_Controller SHALL abort pending work and release workers without leaking resources.

### Requirement 14: Lazy Model Loading and Caching

**User Story:** As a user, I want the segmentation model downloaded only when needed and reused afterward, so that cold app load stays fast and repeat use is efficient.

#### Acceptance Criteria

1. WHEN Cartoon_Mode is enabled AND the Segmentation_Model is present in neither memory nor the Cache Storage API, THE Cartoon_Pipeline_Controller SHALL issue exactly one fetch for the Segmentation_Model from the application static origin.
2. WHEN the Segmentation_Model is fetched successfully AND passes the 50 MB size verification, THE Cartoon_Pipeline_Controller SHALL store it using the Cache Storage API for reuse.
3. WHEN the Segmentation_Model is already present in the Cache Storage API, THE Cartoon_Pipeline_Controller SHALL load it from cache without issuing a network fetch.
4. IF the Segmentation_Model exceeds the 50 MB budget, the download fails, or the download does not complete within 30 seconds, THEN THE Cartoon_Pipeline_Controller SHALL enter Degraded_Mode and report the degraded state in its frame result.
5. WHEN multiple Segmentation_Model fetches are requested concurrently, THE Cartoon_Pipeline_Controller SHALL deduplicate them into a single in-flight fetch.
6. IF writing the Segmentation_Model to the Cache Storage API fails, THEN THE Cartoon_Pipeline_Controller SHALL retain the model in memory and continue inference without entering Degraded_Mode.
7. IF a cached Segmentation_Model fails integrity or size verification, THEN THE Cartoon_Pipeline_Controller SHALL re-fetch the Segmentation_Model from the application static origin.
8. WHEN poseHandEnabled is true AND the Pose_Model and Hand_Model are present in neither memory nor the Cache Storage API, THE Cartoon_Pipeline_Controller SHALL issue exactly one fetch for each of the Pose_Model and the Hand_Model from the application static origin, and SHALL deduplicate concurrent fetch requests for the same model into a single in-flight fetch.
9. WHEN the Pose_Model or Hand_Model is fetched successfully AND passes its integrity verification, THE Cartoon_Pipeline_Controller SHALL store that model using the Cache Storage API for reuse.
10. WHILE poseHandEnabled is false, THE Cartoon_Pipeline_Controller SHALL NOT request, fetch, or load the Pose_Model or the Hand_Model.
11. WHEN the Pose_Model or Hand_Model download completes, THE Cartoon_Pipeline_Controller SHALL verify, before loading either keypoint model for inference, that the combined downloaded size of the Segmentation_Model, Pose_Model, and Hand_Model does not exceed the 50 MB model size budget.
12. WHEN poseHandEnabled is true AND the requested Pose_Model or Hand_Model is already present in the Cache Storage API, THE Cartoon_Pipeline_Controller SHALL load that model from the Cache Storage API without issuing a network fetch.
13. IF the combined downloaded size of the Segmentation_Model, Pose_Model, and Hand_Model exceeds the 50 MB model size budget, THEN THE Cartoon_Pipeline_Controller SHALL NOT load the Pose_Model or Hand_Model for inference and SHALL fall back to segmentation-only guidance without entering Degraded_Mode.
14. IF the Pose_Model or Hand_Model download fails, fails its integrity verification, or does not complete within 30 seconds, THEN THE Cartoon_Pipeline_Controller SHALL fall back to segmentation-only guidance without entering Degraded_Mode.

### Requirement 15: Cartoon Controls and Expanded Custom Section

**User Story:** As a user, I want controls to adjust cartoon parameters and experiment with the pipeline, so that I can tune output to my preference.

#### Acceptance Criteria

1. WHEN Cartoon_Mode is enabled, THE Line_Art_Animator SHALL display the cartoon control set as visible and interactive, and SHALL constrain each control to the inclusive range of its corresponding Requirement 12 parameter: bilateral strength to `bilateralStrength` `[0,100]`, bilateral radius to `bilateralRadius` `[1,7]`, adaptive block size to `adaptiveBlockSize` (odd integer in `[9,151]`), adaptive C to `adaptiveC` `[-20,20]`, segmentation softness to `segSoftness` `[0,40]`, background-fade strength to `segFadeStrength` `[0,1]`, line weight to `cartoonLineWeight` `[1,5]`, RDP epsilon to `rdpEpsilon` `[0.5,8]`, Bézier smoothing to `bezierSmoothing` `[0,1]`, hand-detail-strength to `handDetailStrength` `[0,1]`, joint-detail-strength to `jointDetailStrength` `[0,1]`, and keypoint-confidence to `keypointConfidence` `[0,1]`; SHALL constrain the segmentation toggle, the vectorization toggle, and the enable-pose-and-hand-detail toggle (`poseHandEnabled`) to boolean values; and SHALL offer the quality profile selector exactly the values `maxQuality` and `balanced`.
2. WHEN Cartoon_Mode is disabled, THE Line_Art_Animator SHALL hide the cartoon control set and SHALL keep the existing custom controls visible and unchanged.
3. WHEN a cartoon control value changes, THE Line_Art_Animator SHALL coalesce successive changes that occur within a 200 ms debounce window and, once no further change occurs within that window, SHALL re-render the preview using the current cartoon control values.
4. WHEN a cartoon control value changes, THE Line_Art_Animator SHALL persist the updated cartoon control values through the Settings_Pretty_Printer and Settings_Parser to localStorage using the existing persistence mechanism.
5. WHEN Cartoon_Mode is enabled AND previously persisted cartoon control values exist in localStorage, THE Line_Art_Animator SHALL restore those values to the corresponding cartoon controls.
6. IF persisting cartoon control values to localStorage fails because storage is unavailable or the quota is exceeded, THEN THE Line_Art_Animator SHALL retain the current cartoon control values in memory, continue preview rendering, and surface an indication that persistence failed.
7. WHEN a pose-or-hand guidance control value (`poseHandEnabled`, `handDetailStrength`, `jointDetailStrength`, or `keypointConfidence`) changes, THE Line_Art_Animator SHALL coalesce the change within the same 200 ms debounce window defined in criterion 3 and SHALL persist the updated value through the Settings_Pretty_Printer and Settings_Parser to localStorage using the same serialization and persistence mechanism as the other cartoon control values.
8. WHILE the enable-pose-and-hand-detail toggle is off, THE Line_Art_Animator SHALL NOT load the Pose_Model or the Hand_Model.
9. WHEN the enable-pose-and-hand-detail toggle transitions from off to on AND the first frame is subsequently submitted for rendering while the toggle is on, THE Line_Art_Animator SHALL lazy-load the Pose_Model and Hand_Model before producing that frame's keypoint guidance.

### Requirement 16: Backend Equivalence

**User Story:** As a user, I want consistent output regardless of which device tier runs the pipeline, so that the look is stable across browsers.

#### Acceptance Criteria

1. WHEN identical input and parameters are processed pairwise across the available Tier A, Tier B, and Tier C edge stages, THE Edge_Shader_Stage SHALL produce outputs that agree within a maximum absolute per-pixel error of 0.05 on the `[0,1]` edge-intensity scale and a structural similarity of at least 0.99.
2. WHEN edge-stage outputs are compared across tiers, THE Edge_Shader_Stage SHALL use the identical input frame and an identical processing resolution for every tier in the comparison.

### Requirement 17: Client-Side Privacy

**User Story:** As a user, I want all processing to stay on my device, so that my media is never uploaded.

#### Acceptance Criteria

1. THE Cartoon_Pipeline SHALL perform all inference and image processing within the browser on the local device.
2. THE Cartoon_Pipeline SHALL NOT transmit any input frame, intermediate buffer, or output frame to any network endpoint.
3. THE Cartoon_Pipeline_Controller SHALL load the inference runtime, Segmentation_Model, Pose_Model, and Hand_Model only from the same origin as the application, and SHALL NOT request them from any third-party network endpoint.
4. WHEN the Segmentation_Model download completes, THE Cartoon_Pipeline_Controller SHALL verify that the downloaded model size does not exceed the 50 MB budget before loading it for inference.
5. IF the downloaded Segmentation_Model size exceeds the 50 MB budget, THEN THE Cartoon_Pipeline_Controller SHALL discard the model without using it for inference and SHALL report the failure in its frame result.
6. THE Cartoon_Pipeline_Controller SHALL NOT transmit any Pose_Landmark, Hand_Landmark, or Detail_Boost_Map to any network endpoint.
7. THE Cartoon_Pipeline_Controller SHALL count the Pose_Model and Hand_Model within the same 50 MB model size budget as the Segmentation_Model and SHALL NOT upload either model.

### Requirement 18: Anatomical Validity of Keypoint Guidance

**User Story:** As a user, I want the optional pose and hand-landmark detail boosting to follow real human anatomy and only ever add detail, so that finer line work appears on hands, fingers, and joints without erasing existing lines or inventing structure where there is none.

#### Acceptance Criteria

1. WHEN the Keypoint_Guidance_Stage builds the Detail_Boost_Map for a frame, THE Keypoint_Guidance_Stage SHALL produce a Detail_Boost_Map with every value bounded in the range `[0,1]`, and THE Cartoon_Pipeline SHALL produce a composite edge intensity at every pixel that is bounded in the range `[0,1]` and is greater than or equal to the Segmentation_Only_Result at that pixel within an absolute tolerance of `1/255`.
2. WHEN the Keypoint_Guidance_Stage builds the Detail_Boost_Map, THE Keypoint_Guidance_Stage SHALL splat boost only along a Bone defined in the Skeleton_Topology (`POSE_BONES` or `HAND_BONES`) and only when both endpoints of that Bone are present, where a landmark is present only when its score is greater than or equal to `keypointConfidence` and its normalized coordinates lie within `[0,1]×[0,1]`, and SHALL contribute no boost between unconnected landmarks or between a present landmark and an absent or below-confidence landmark.
3. WHILE keypoint guidance is enabled, THE Keypoint_Guidance_Stage SHALL exclude from boost every landmark whose score is below `keypointConfidence`, SHALL reject any Hand that does not have exactly 21 Hand_Landmark values with a handedness of `Left` or `Right`, SHALL produce an all-zero Detail_Boost_Map when no Bone in the Skeleton_Topology has both endpoints present, and SHALL produce a total boost that is monotonically non-increasing as `keypointConfidence` increases while all other inputs are held constant.
4. IF the Pose_Model or Hand_Model is absent, fails to load, fails inference, does not complete loading within 30 seconds, does not complete inference within 2000 milliseconds for a frame, or the selected Execution_Tier does not support keypoints, THEN THE Cartoon_Pipeline_Controller SHALL fall back to segmentation-only guidance, SHALL resolve every frame to a valid RGBA_Frame without crashing the render loop, and SHALL produce output equal to the Segmentation_Only_Result for that frame.
5. THE Keypoint_Guidance_Stage SHALL map `handDetailStrength` to the boost magnitude splatted along `HAND_BONES` and `jointDetailStrength` to the boost magnitude splatted along `POSE_BONES`, each as a monotonically non-decreasing relationship, where a strength of 0 contributes zero boost along the corresponding bones.
6. WHERE both `handDetailStrength` and `jointDetailStrength` are 0, THE Cartoon_Pipeline SHALL produce composite output equal to the Segmentation_Only_Result within an absolute tolerance of `1/255`.
