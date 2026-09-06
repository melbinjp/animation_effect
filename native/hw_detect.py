import numpy as np
import scipy.ndimage as scipy_ndimage

try:
    import cupy
    import cupyx.scipy.ndimage as cupy_ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cupy = None
    cupy_ndimage = None

_force_cpu = False


def set_force_cpu(val: bool):
    """Overrides GPU usage temporarily (e.g. for fallback on OOM frames)."""
    global _force_cpu
    _force_cpu = val


def is_gpu_active():
    """Returns True if GPU is available and not currently forced to CPU."""
    return HAS_GPU and not _force_cpu


def get_xp():
    """Returns cupy if GPU is active, otherwise numpy."""
    return cupy if is_gpu_active() else np


def get_ndimage():
    """Returns cupyx.scipy.ndimage if GPU is active, otherwise scipy.ndimage."""
    return cupy_ndimage if is_gpu_active() else scipy_ndimage


def to_device(array):
    """Moves array to GPU VRAM if GPU is active. Otherwise returns NumPy array."""
    if is_gpu_active():
        return cupy.asarray(array)
    return np.asarray(array)


def to_host(array):
    """Moves GPU array back to CPU as NumPy array. Otherwise returns array."""
    if HAS_GPU and cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return np.asarray(array)


def clear_gpu_memory():
    """Flushes unused cached blocks in CuPy's default memory pools."""
    if HAS_GPU and cupy is not None:
        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def is_oom_error(exc):
    """Checks if an exception is a CUDA OutOfMemoryError."""
    if HAS_GPU and cupy is not None:
        try:
            return isinstance(exc, cupy.cuda.memory.OutOfMemoryError)
        except Exception:
            pass
    return False


def get_vram_info():
    """Returns (free_gb, total_gb) if GPU is available, else (0.0, 0.0)."""
    if HAS_GPU and cupy is not None:
        try:
            free_b, total_b = cupy.cuda.Device(0).mem_info
            return free_b / (1024 ** 3), total_b / (1024 ** 3)
        except Exception:
            pass
    return 0.0, 0.0


_logged = False


def log_hardware_status():
    global _logged
    if not _logged:
        if HAS_GPU:
            free_gb, total_gb = get_vram_info()
            if total_gb > 0:
                print(f"[INFO] Hardware Detection: NVIDIA GPU detected ({total_gb:.1f} GB VRAM). Using CuPy.")
            else:
                print("[INFO] Hardware Detection: NVIDIA GPU detected. Using CuPy for acceleration.")
        else:
            print("[INFO] Hardware Detection: No GPU/CuPy detected. Using CPU NumPy.")
        _logged = True
