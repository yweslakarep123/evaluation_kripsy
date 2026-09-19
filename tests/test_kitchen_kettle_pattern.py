#!/usr/bin/env python3
"""Unit tests for observational kettle-pattern helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_kitchen_kettle_pattern import (  # noqa: E402
    kl_divergence,
    laplace_normalize,
    shannon_entropy,
)
from kitchen_eval_stats import clip_episode_to_p4  # noqa: E402

ALL7 = [
    "microwave",
    "kettle",
    "bottom burner",
    "light switch",
    "top burner",
    "slide cabinet",
    "hinge cabinet",
]


def test_entropy_delta_and_uniform():
    assert math.isclose(shannon_entropy([1.0, 0.0, 0.0]), 0.0, abs_tol=1e-12)
    u = [0.25, 0.25, 0.25, 0.25]
    assert math.isclose(shannon_entropy(u), math.log(4.0), rel_tol=1e-9)


def test_kl_to_self_is_zero():
    p = laplace_normalize([3.0, 1.0, 0.0, 2.0])
    assert math.isclose(kl_divergence(p, p), 0.0, abs_tol=1e-12)


def test_clip_kettle_none_when_k4_without_kettle():
    order = ["microwave", "bottom burner", "light switch", "slide cabinet", "kettle"]
    scored, k, flags = clip_episode_to_p4(order, ALL7)
    assert k == 4
    assert "kettle" not in scored
    assert flags["kettle"] is None


if __name__ == "__main__":
    test_entropy_delta_and_uniform()
    test_kl_to_self_is_zero()
    test_clip_kettle_none_when_k4_without_kettle()
    print("ok")
