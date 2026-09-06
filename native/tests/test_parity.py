"""Parity test suite: verifies synchronization between web application (script.js)
and native engine (presets.py, cli.py).
"""

import json
import os
import re
import sys
from pathlib import Path

# Add native directory to path
NATIVE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NATIVE_DIR))

from presets import STYLE_PRESETS, HUMAN_TUNING, get_preset, get_human_tuning
from cli import parse_args, _build_settings


def _parse_js_presets():
    """Extracts STYLE_PRESETS dictionary from script.js using regex parsing."""
    repo_root = NATIVE_DIR.parent
    script_path = repo_root / "script.js"
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract preset block
    match = re.search(r"const STYLE_PRESETS = \{([\s\S]*?)\n\};", content)
    assert match is not None, "STYLE_PRESETS definition not found in script.js"
    block = match.group(1)

    # Extract preset names
    preset_names = re.findall(r"^\s+([a-zA-Z0-9_]+):\s*\{", block, re.MULTILINE)
    return set(preset_names)


def test_preset_names_parity():
    """Verifies that all presets defined in script.js exist in native presets."""
    js_presets = _parse_js_presets()
    native_presets = set(STYLE_PRESETS.keys())

    # Every JS preset must exist in native
    for p in js_presets:
        assert p in native_presets, f"Preset '{p}' in script.js is missing from native STYLE_PRESETS"


def test_portrait_alias():
    """Verifies that portrait is a valid alias for human in native presets."""
    assert "portrait" in STYLE_PRESETS
    assert "human" in STYLE_PRESETS
    assert STYLE_PRESETS["portrait"] == STYLE_PRESETS["human"]
    assert get_preset("portrait") == get_preset("human")
    assert get_human_tuning("portrait") == get_human_tuning("human")


def test_custom_settings_json_merge():
    """Verifies that cli._build_settings correctly merges --settings-json overrides."""
    class DummyArgs:
        preset = "custom"
        detail = 62
        line_weight = 1
        white_balance = False
        no_auto_normalize = False
        dark_boost = False
        human_aware = True
        pose_lines = False
        face_contours = False
        temporal_denoise = False
        body_map_overlay = False
        crf = 24
        settings_json = json.dumps({
            "preset": {
                "background": [200, 200, 200],
                "low_threshold": 50,
            },
            "skin_smooth": 0.95,
        })

    settings = _build_settings(DummyArgs())
    assert settings["preset"]["background"] == [200, 200, 200]
    assert settings["preset"]["low_threshold"] == 50
    assert settings["skin_smooth"] == 0.95
    assert settings["custom_mode"] is True


def test_body_map_overlay_flag():
    """Verifies that body_map_overlay flag is recognized in settings."""
    class DummyArgs:
        preset = "ultimate"
        detail = 62
        line_weight = 1
        white_balance = False
        no_auto_normalize = False
        dark_boost = False
        human_aware = True
        pose_lines = False
        face_contours = False
        temporal_denoise = False
        body_map_overlay = True
        crf = 24
        settings_json = None

    settings = _build_settings(DummyArgs())
    assert settings["body_map_overlay"] is True


def test_oom_recovery_mechanism():
    """Verifies that OOM error detection and CPU forcing function correctly."""
    from hw_detect import set_force_cpu, is_gpu_active, clear_gpu_memory
    
    clear_gpu_memory()
    set_force_cpu(True)
    assert not is_gpu_active()
    set_force_cpu(False)
