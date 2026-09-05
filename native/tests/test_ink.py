"""Pure-numeric tests for ink.py, mirroring the equivalent cases in
tests/encode-resilience.test.mjs's JS counterpart. No video, no MediaPipe
model, no ffmpeg -- small synthetic arrays only, so this runs anywhere
numpy/opencv are installed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ink import compute_class_confidence, compute_xdog_map, apply_human_ink, silhouette_mask

HUMAN_BG, HUMAN_HAIR, HUMAN_BODY, HUMAN_FACE, HUMAN_CLOTHES, HUMAN_OTHER = range(6)


def test_xdog_map_flat_region_stays_near_background():
    # A perfectly flat gray field has zero local contrast everywhere, so the
    # DoG response is ~0 and every pixel should land on the "background"
    # branch (dog < eps -> 1 + tanh(negative) < 1).
    gray = np.full((32, 32), 128, dtype=np.uint8)
    xdog = compute_xdog_map(gray, sigma=0.82, tau=0.983, phi=210)
    assert xdog.shape == (32, 32)
    assert np.all(xdog <= 1.0)
    assert np.all(xdog >= 0.0)
    # Flat input: response should be uniform (no edges to detect).
    assert np.allclose(xdog, xdog[0, 0], atol=1e-4)


def test_xdog_map_detects_a_hard_edge():
    gray = np.zeros((32, 32), dtype=np.uint8)
    gray[:, 16:] = 255  # a hard vertical edge down the middle
    xdog = compute_xdog_map(gray, sigma=0.82, tau=0.983, phi=210)
    # Right at the edge, the DoG response should dip well below the flat
    # background level found a few pixels away from the transition.
    edge_val = xdog[16, 15]
    flat_val = xdog[16, 2]
    assert edge_val < flat_val


def test_class_confidence_is_one_deep_inside_a_solid_region():
    # A large solid block of a single class: every interior pixel's full
    # 3x3 neighborhood shares its class, so confidence should be exactly 1.
    mask = np.full((20, 20), HUMAN_BODY, dtype=np.uint8)
    conf = compute_class_confidence(mask)
    interior = conf[5:15, 5:15]
    assert np.allclose(interior, 1.0)


def test_class_confidence_drops_at_a_class_boundary():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 10:] = HUMAN_BODY  # a hard boundary at column 10
    conf = compute_class_confidence(mask)
    # Pixels right at the boundary have mixed-class neighbors -> confidence < 1.
    assert conf[10, 9] < 1.0
    assert conf[10, 10] < 1.0
    # Pixels far from the boundary are still fully confident.
    assert conf[10, 2] == 1.0
    assert conf[10, 17] == 1.0


def test_class_confidence_is_bounded_zero_to_one():
    rng = np.random.default_rng(42)
    mask = rng.integers(0, 6, (24, 24)).astype(np.uint8)
    conf = compute_class_confidence(mask)
    assert conf.min() >= 0.0
    assert conf.max() <= 1.0 + 1e-6


def test_apply_human_ink_bypasses_when_not_human_aware():
    ink = np.full((10, 10), 0.5, dtype=np.float32)
    mask = np.full((10, 10), HUMAN_BODY, dtype=np.uint8)
    out = apply_human_ink(ink, mask, None, {"human_aware": False})
    assert out is ink  # explicit no-op, matches worker.js's early return


def test_apply_human_ink_bypasses_when_no_class_mask():
    ink = np.full((10, 10), 0.5, dtype=np.float32)
    out = apply_human_ink(ink, None, None, {"human_aware": True})
    assert out is ink


def test_apply_human_ink_quiets_background_by_isolation_amount():
    ink = np.full((10, 10), 1.0, dtype=np.float32)
    mask = np.full((10, 10), HUMAN_BG, dtype=np.uint8)  # uniform background, confidence 1.0 everywhere
    settings = {"human_aware": True, "subject_isolation": 0.4, "skin_smooth": 0.8, "hair_boost": 1.32, "silhouette_boost": 0.72}
    out = apply_human_ink(ink, mask, None, settings)
    # Uniform mask -> confidence is 1.0 everywhere except the frame border
    # (silhouette_mask's dilate/erode both touch the border under
    # BORDER_CONSTANT zero-padding) -- check the interior, matching
    # compute_class_confidence's own border-behavior test above.
    assert np.allclose(out[2:-2, 2:-2], 1.0 * (1 - 0.4), atol=1e-5)


def test_apply_human_ink_boosts_hair_but_caps_at_one():
    ink = np.full((10, 10), 0.9, dtype=np.float32)
    mask = np.full((10, 10), HUMAN_HAIR, dtype=np.uint8)
    settings = {"human_aware": True, "subject_isolation": 0.38, "skin_smooth": 0.8, "hair_boost": 1.32, "silhouette_boost": 0.72}
    out = apply_human_ink(ink, mask, None, settings)
    assert np.all(out <= 1.0)  # 0.9 * 1.32 = 1.188, must clamp to 1.0
    assert np.allclose(out[2:-2, 2:-2], 1.0, atol=1e-5)


def test_silhouette_mask_marks_only_the_transition_band():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 10:] = HUMAN_BODY
    sil = silhouette_mask(mask)
    # Far from the boundary: no silhouette band.
    assert sil[10, 2] == 0.0
    assert sil[10, 17] == 0.0
    # Right at the boundary: silhouette band should be active somewhere nearby.
    assert sil[10, 8:12].sum() > 0
