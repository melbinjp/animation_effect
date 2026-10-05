"""Anime preset: scene classes, flat source colour, and preset parity."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anime_cel import paint_anime
from presets import STYLE_PRESETS, get_human_tuning, get_preset
from scene_seg import (
    GROUND, ISLAND_MIN_AT_256, SKY, STRUCTURE, VEGETATION,
    classify_hsv, next_scene, segment_rgb,
)


def _sat(rgb):
    mx = rgb.max(axis=-1).astype(np.float32)
    mn = rgb.min(axis=-1).astype(np.float32)
    return np.where(mx == 0, 0, (mx - mn) / np.maximum(mx, 1))


def _luma(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def test_classify_hsv_names():
    assert classify_hsv(220, 0.5, 0.8, 0.2) == SKY
    assert classify_hsv(120, 0.45, 0.5, 0.4) == VEGETATION
    assert classify_hsv(0, 0.05, 0.5, 0.8) == GROUND
    assert classify_hsv(20, 0.6, 0.7, 0.4) == STRUCTURE
    assert classify_hsv(220, 0.5, 0.8, 0.9) == STRUCTURE


def test_segment_sky_ground_and_speck():
    img = np.zeros((80, 80, 3), np.uint8)
    img[:24] = (40, 110, 230)
    img[24:57] = (150, 140, 130)
    img[57:] = (128, 128, 128)
    img[30:32, 40:42] = (20, 180, 40)
    seg = segment_rgb(img)
    assert np.mean(seg["klass"][:16] == SKY) > 0.7
    assert np.mean(seg["klass"][64:] == GROUND) > 0.7
    assert np.count_nonzero(seg["klass"] == VEGETATION) < 8


def test_two_vegetation_objects():
    img = np.full((96, 96, 3), (170, 150, 130), np.uint8)
    img[20:52, 8:36] = (30, 150, 40)
    img[20:52, 58:88] = (40, 170, 50)
    seg = segment_rgb(img)
    veg = [obj for obj in seg["objects"] if obj["klass"] == VEGETATION]
    assert len(veg) >= 2


def test_next_scene_hold_is_sequential_only():
    a = np.full((32, 32, 3), (40, 110, 230), np.uint8)
    b = np.full((32, 32, 3), (30, 160, 40), np.uint8)
    first = next_scene(a, None, True)
    assert first["reused"] is False
    assert first["hold"]["left"] == 2
    second = next_scene(b, first["hold"], True)
    assert second["reused"] is True
    assert second["mask"] is first["mask"]
    third = next_scene(b, second["hold"], True)
    assert third["reused"] is True
    fourth = next_scene(b, third["hold"], True)
    assert fourth["reused"] is False
    batch = next_scene(a, first["hold"], False)
    assert batch["reused"] is False
    assert batch["hold"] is None


def test_paint_saturates_and_lifts():
    rgb = np.full((32, 32, 3), (50, 70, 50), np.uint8)
    mask = np.ones((32, 32), np.uint8)
    klass = np.full((32, 32), VEGETATION, np.uint8)
    out = paint_anime(rgb, edges=np.zeros((32, 32), np.uint8), scene_mask=mask, scene_class=klass)
    pix = out[16, 8].astype(np.float32)
    src = rgb[16, 8].astype(np.float32)
    assert pix[1] > pix[0] and pix[1] > pix[2]
    assert _sat(pix.reshape(1, 1, 3))[0, 0] > _sat(src.reshape(1, 1, 3))[0, 0]
    assert _luma(pix.reshape(1, 1, 3))[0, 0] > _luma(src.reshape(1, 1, 3))[0, 0] + 30


def test_paint_boundary_is_darker_than_the_flat():
    rgb = np.full((24, 48, 3), (40, 140, 50), np.uint8)
    mask = np.ones((24, 48), np.uint8)
    mask[:, 24:] = 2
    klass = np.full((24, 48), VEGETATION, np.uint8)
    out = paint_anime(
        rgb, edges=np.zeros((24, 48), np.uint8),
        scene_mask=mask, scene_class=klass, ink=(28, 18, 42),
    )
    flat = _luma(out[12:13, 6:7])[0, 0]
    edge = _luma(out[12:13, 24:25])[0, 0]
    assert edge < flat - 40
    assert out[12, 26, 0] > out[12, 6, 0]


def test_island_minimum_matches_body_map():
    assert ISLAND_MIN_AT_256 == 40
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    human_py = open(os.path.join(here, "human.py"), encoding="utf-8").read()
    human_js = open(os.path.join(here, "..", "human.js"), encoding="utf-8").read()
    assert "CC_STAT_AREA] < 40" in human_py
    assert "filterSmallComponents(nativeMask, sw, sh, 40)" in human_js


def test_anime_preset_is_registered_without_removing_ultimate():
    assert "anime" in STYLE_PRESETS
    assert STYLE_PRESETS["ultimate"]["engine"] == "ultimate"
    assert STYLE_PRESETS["classic"]["engine"] == "classic"
    assert get_preset("anime")["engine"] == "anime"
    assert get_preset("anime")["skip_auto_normalize"] is True
    tuning = get_human_tuning("anime")
    assert tuning["subject_isolation"] < get_human_tuning("subject")["subject_isolation"]


def test_pipeline_anime_returns_colour_and_ultimate_stays_ink():
    import cv2
    from pipeline import process_frame

    rgb = np.full((48, 48, 3), (50, 90, 50), np.uint8)
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    anime_settings = {
        "preset": dict(get_preset("anime")),
        "engine": "anime",
        "detail": 62,
        "custom_mode": False,
        "white_balance": False,
        "auto_normalize": False,
        "dark_boost": False,
        "clean_speckles": True,
        "clean_speckles_intensity": 1,
        "merge_double_edge": True,
        "merge_double_edge_intensity": 2,
        "line_weight": 1,
        "human_aware": False,
    }
    anime_out, _ = process_frame(rgba, anime_settings)
    assert anime_out.shape == (48, 48, 3)
    center = anime_out[24, 8].astype(np.int16)
    assert center[1] > center[0] and center[1] > center[2]
    assert _luma(anime_out[24:25, 8:9])[0, 0] > 80

    ultimate_settings = dict(anime_settings)
    ultimate_settings["preset"] = dict(get_preset("ultimate"))
    ultimate_settings["engine"] = "ultimate"
    ultimate_settings["auto_normalize"] = True
    ink_out, _ = process_frame(rgba, ultimate_settings)
    assert ink_out.shape == anime_out.shape
    assert _sat(anime_out).mean() > _sat(ink_out).mean() + 0.1
    assert np.mean(np.abs(anime_out.astype(np.int16) - ink_out.astype(np.int16))) > 30
