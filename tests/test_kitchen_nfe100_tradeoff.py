#!/usr/bin/env python3
"""Tests for NFE100 trade-off: p4 vs latency per executed action."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_kitchen_nfe100 import compute_utopia_distances  # noqa: E402


def test_utopia_uses_per_action_latency_and_p4():
    points = [
        {
            "model": "flowpolicy",
            "train_seed": "42",
            "nfe": 1,
            "sseed": 0,
            "latency_per_action_ms": 2.0,
            "p4": 1.0,
            "latency_ms": 8.0,
            "success_rate_p14": 0.25,
        },
        {
            "model": "diffusion_policy_cnn",
            "train_seed": "train0",
            "nfe": 100,
            "sseed": 0,
            "latency_per_action_ms": 4.0,
            "p4": 0.5,
            "latency_ms": 32.0,
            "success_rate_p14": 0.9,
        },
    ]
    l_max, rows = compute_utopia_distances(points)
    assert math.isclose(l_max, 4.0)
    by_model = {r["model"]: r for r in rows}
    fast = by_model["flowpolicy"]
    assert math.isclose(fast["x_tilde"], 0.5)
    assert math.isclose(fast["p4"], 1.0)
    assert math.isclose(fast["distance"], 0.5)
    slow = by_model["diffusion_policy_cnn"]
    assert math.isclose(slow["x_tilde"], 1.0)
    assert math.isclose(slow["p4"], 0.5)
    assert math.isclose(slow["distance"], math.sqrt(1.0 + 0.25))


def test_utopia_skips_missing_per_action():
    points = [
        {
            "model": "flowpolicy",
            "train_seed": "42",
            "nfe": 1,
            "latency_per_action_ms": None,
            "p4": 0.8,
        },
        {
            "model": "flowpolicy",
            "train_seed": "43",
            "nfe": 1,
            "latency_per_action_ms": 3.0,
            "p4": 0.6,
        },
    ]
    l_max, rows = compute_utopia_distances(points)
    assert math.isclose(l_max, 3.0)
    assert len(rows) == 1
    assert rows[0]["train_seed"] == "43"
    assert math.isclose(rows[0]["p4"], 0.6)


if __name__ == "__main__":
    test_utopia_uses_per_action_latency_and_p4()
    test_utopia_skips_missing_per_action()
    print("ok")
