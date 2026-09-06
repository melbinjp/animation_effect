# Line Art Animator (Native Server & CLI)

High-performance Python implementation of the line-art and inking pipeline. Optimized for multi-core CPUs and NVIDIA GPUs without browser memory or runtime constraints.

---

## Architecture & Hardware Acceleration

```
┌────────────────────────────────────────────────────────┐
│                      hw_detect.py                      │
│        (Probes for CUDA / CuPy / Hardware Encoder)     │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    [GPU Detected]                  [CPU Fallback]
    - CuPy array math               - NumPy array math
    - FFmpeg NVDEC/NVENC            - FFmpeg libx264
    - Worker pool capped at 4       - Worker pool = CPU core count
    - scipy/cupyx ndimage filters   - scipy ndimage filters
```

- **Dynamic Hardware Dispatch (`hw_detect.py`):** Automatically routes filter math (`XDoG`, Gaussian filters, morphology) through CuPy on CUDA GPUs or NumPy on CPUs.
- **Video I/O (`cli.py`):** Uses FFmpeg via pipe streaming. Dynamically enables `-hwaccel cuda` and `h264_nvenc` if an NVIDIA GPU is available; falls back to `libx264`.
- **Segmentation (`human.py`):** MediaPipe Selfie Multiclass pipeline (6 classes: background, hair, body skin, face skin, clothes, other).
- **Process Parallelism:** Video is divided into contiguous frame segments distributed across worker processes (`multiprocessing.Pool`). Worker counts are capped at 4 on GPU to prevent VRAM exhaustion.

---

## Requirements

- **OS:** Linux, macOS, or Windows
- **Python:** 3.9+
- **System Dependencies:** `ffmpeg` and `ffprobe` installed and accessible in `PATH`.
  - Ubuntu/Debian: `apt install -y ffmpeg libegl1 libgl1 libgles2`
- **Hardware (Optional):** NVIDIA GPU with CUDA 11.x or 12.x for hardware acceleration.

---

## Installation

### 1. Base Environment (CPU)

```bash
cd native
pip install -r requirements.txt -r requirements-webui.txt
```

> **Note:** `mediapipe` includes its own `opencv-contrib-python`. Do not separately install `opencv-python`.

### 2. Enable GPU Acceleration (NVIDIA)

RunPod GPU templates and systems with NVIDIA drivers already contain CUDA. Install the matching precompiled CuPy wheel:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

If CuPy is not installed, the pipeline executes on the CPU without throwing errors.

---

## CLI Usage

```bash
python cli.py input.mp4 -o output.mp4 [OPTIONS]
```

### Options Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-o, --output` | string | `output.mp4` | Destination video path. |
| `--preset` | string | `ultimate` | Style preset (`ultimate`, `portrait`, `subject`, `studio`, `manga`, `pencil`, `neon`, `vivid`, `warm`, `blueprint`, `classic`, `body`, `custom`). |
| `--detail` | integer | `62` | Detail extraction slider (1–100). |
| `--line-weight` | integer | `1` | Dilation pass count for line thickness. |
| `--human-aware` | flag | Auto | Enables MediaPipe segmentation (default `True` except for `classic`). |
| `--no-human-aware` | flag | - | Disables subject-aware edge suppression. |
| `--body-map-overlay`| flag | `False` | Bakes colorized body map overlay into output. |
| `--temporal-denoise`| flag | `False` | Motion-adaptive temporal smoothing on pre-Canny luminance. |
| `--white-balance` | flag | `False` | Gray-world white balance normalization. |
| `--max-dimension` | integer | `None` | Caps output resolution along the longest edge (e.g. `1080`). |
| `--fps` | float | `None` | Overrides output frame rate. Defaults to input source FPS. |
| `--workers` | integer | Auto | Concurrent worker processes. Defaults to all CPU cores, capped at 4 if GPU is present. |
| `--encoder` | string | `auto` | Video codec: `auto`, `nvenc`, `vaapi`, `qsv`, or `libx264`. |
| `--quality` | string | `balanced` | Quality tier (`indistinguishable`, `optimized`, `balanced`, `small`, `aggressive`, `maximum`). |
| `--crf` | integer | `None` | Explicit CRF override (e.g. `20`). |
| `--settings-json` | string | `None` | JSON string or file path containing pipeline settings overrides. |

---

## Web UI Studio

FastAPI-based studio interface providing frame preview, split-view comparison, timeline scrubbing, parameter tuning, and render queue execution.

```bash
python webui.py --host 127.0.0.1 --port 8765
```

Navigate to `http://127.0.0.1:8765/`.

### API Endpoints

- `POST /preview`: Renders a single-frame styled JPEG preview with timing metrics.
- `POST /process-image`: Processes and saves a static image with custom settings.
- `POST /jobs`: Launches an asynchronous background video render job via `cli.py`.
- `GET /jobs/{job_id}/status`: Polls status, frame progress, percentage, and ETA of an active render job.
- `GET /jobs/{job_id}/log`: Streams live stdout/stderr log of a running render job.
- `GET /jobs/{job_id}/download`: Serves the completed output video file.
- `POST /jobs/{job_id}/cancel`: Aborts active render job and its child processes.

---

## RunPod Cloud Deployment

### Method A: Automated Deployment (PowerShell)

Automates pod provisioning, environment setup, Web UI launch, and SSH tunnel creation:

```powershell
cd native
.\deploy-runpod-webui.ps1
```

*Prerequisites:* RunPod API key configured in credential store and SSH key generated at `~/.ssh/runpod_animation_effect_ed25519`.

### Method B: Manual Deployment (Linux / Any OS)

1. Provision an instance on [RunPod](https://runpod.io):
   - **Template:** `runpod/base:1.0.2-ubuntu2204` (CPU) or PyTorch CUDA template (GPU).
   - **Disk:** 20 GB container disk.
   - **Ports:** Expose port 22 (SSH).

2. Bootstrap pod environment:
   ```bash
   ssh -i <key> -p <port> root@<pod_ip> \
     "curl -fsSL https://raw.githubusercontent.com/melbinjp/animation_effect/main/native/setup_pod.sh | bash"
   ```

3. Launch Web UI in background:
   ```bash
   ssh -i <key> -p <port> root@<pod_ip> \
     "cd /workspace/animation_effect/native && nohup python3 webui.py --host 127.0.0.1 --port 8765 > webui.log 2>&1 &"
   ```

4. Establish local SSH tunnel:
   ```bash
   ssh -i <key> -p <port> -N -L 127.0.0.1:8765:127.0.0.1:8765 root@<pod_ip>
   ```

5. Access `http://127.0.0.1:8765/` in your local browser.

---

## Testing

Execute test suite verifying edge mathematics, confidence calculations, and CPU fallback:

```bash
pytest tests/
```
