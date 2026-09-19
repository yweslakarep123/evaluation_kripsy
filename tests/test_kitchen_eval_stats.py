#!/usr/bin/env python3
"""Tests for canonical Wilson 95% proportion helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kitchen_eval_stats import (  # noqa: E402
    K_MAX,
    TIMING_METHODOLOGY_KEYS,
    build_timing_methodology,
    classify_p4_outcome,
    clip_episode_to_p4,
    compute_wilson_ci,
    episode_k_capped,
    proportion_stats,
    restore_rng_state,
    save_rng_state,
    timing_sample_stats,
)


def test_wilson_60_of_100():
    lo, hi = compute_wilson_ci(60, 100, confidence=0.95)
    assert lo is not None and hi is not None
    assert 0.0 <= lo < 0.6 < hi <= 1.0
    assert math.isclose(lo, 0.502, abs_tol=0.003)
    assert math.isclose(hi, 0.691, abs_tol=0.003)
    stats = proportion_stats([1] * 60 + [0] * 40)
    assert stats["mean"] == 0.6
    assert stats["n_success"] == 60
    assert stats["n_samples"] == 100
    assert stats["ci_method"] == "wilson"
    assert stats["confidence"] == 0.95
    assert math.isclose(stats["ci_low"], lo)
    assert math.isclose(stats["ci_high"], hi)
    assert math.isclose(stats["std"], (hi - lo) / 2.0)
    # Display mean ± half-width stays near the pedagogical 0.60 ± 0.094
    assert math.isclose(stats["std"], 0.094, abs_tol=0.002)


def test_wilson_0_of_100():
    lo, hi = compute_wilson_ci(0, 100, confidence=0.95)
    assert lo == 0.0
    assert hi is not None and 0.0 < hi < 0.05
    stats = proportion_stats([0] * 100)
    assert stats["mean"] == 0.0
    assert stats["n_success"] == 0
    assert stats["ci_low"] == 0.0
    assert stats["ci_high"] == hi
    assert stats["std"] > 0.0


def test_wilson_100_of_100():
    lo, hi = compute_wilson_ci(100, 100, confidence=0.95)
    assert hi == 1.0
    assert lo is not None and 0.95 < lo < 1.0
    stats = proportion_stats([1] * 100)
    assert stats["mean"] == 1.0
    assert stats["n_success"] == 100
    assert stats["ci_high"] == 1.0
    assert stats["ci_low"] == lo
    assert stats["std"] > 0.0


def test_wilson_zero_trials():
    lo, hi = compute_wilson_ci(0, 0, confidence=0.95)
    assert lo is None and hi is None
    stats = proportion_stats([])
    assert stats["mean"] is None
    assert stats["std"] is None
    assert stats["n_samples"] == 0
    assert stats["n_success"] == 0
    assert stats["ci_low"] is None
    assert stats["ci_high"] is None


def test_wilson_stays_in_unit_interval():
    for k, n in [(0, 100), (1, 100), (97, 100), (99, 100), (100, 100), (60, 100)]:
        lo, hi = compute_wilson_ci(k, n)
        assert lo is not None and hi is not None
        assert 0.0 <= lo <= hi <= 1.0


def test_timing_sample_stats_p95():
    stats = timing_sample_stats(list(range(1, 101)))
    assert stats["n_samples"] == 100
    assert math.isclose(stats["mean"], 50.5)
    assert "std" not in stats
    assert math.isclose(stats["p50"], 50.5)
    assert 94.0 <= stats["p95"] <= 96.0
    assert stats["p99"] is not None and stats["p99"] >= stats["p95"]
    empty = timing_sample_stats([])
    assert empty["mean"] is None
    assert empty["n_samples"] == 0


def test_warmup_calls_discarded_from_timed_stats():
    warmup = [1.0] * 10
    timed = [10.0] * 93
    stats = timing_sample_stats(timed)
    method = build_timing_methodology(
        warmup_calls_discarded=len(warmup),
        n_timed_calls=len(timed),
    )
    assert method["warmup_calls_discarded"] == 10
    assert method["n_timed_calls"] == stats["n_samples"] == 93
    assert math.isclose(stats["mean"], 10.0)
    biased = timing_sample_stats(warmup + timed)
    assert biased["n_samples"] == 103
    assert biased["mean"] < stats["mean"]


def test_timing_methodology_keys():
    method = build_timing_methodology(
        warmup_calls_discarded=10,
        n_timed_calls=9300,
    )
    assert tuple(method.keys()) == TIMING_METHODOLOGY_KEYS
    assert method["warmup_calls_discarded"] == 10
    assert method["n_timed_calls"] == 9300
    assert method["batch_size"] == 1
    assert method["cuda_synchronize_before"] is True
    assert method["cuda_synchronize_after"] is True
    assert method["h2d_transfer_in_timed_region"] is False
    assert method["env_step_in_timed_region"] is False
    assert method["percentiles"] == [50, 95, 99]


def test_rng_restore_after_warmup_noise():
    import numpy as np
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    expected_np = np.random.rand(4)
    expected_th = torch.randn(4)

    np.random.seed(0)
    torch.manual_seed(0)
    state = save_rng_state()
    _ = np.random.rand(8)
    _ = torch.randn(8)  # discarded warmup draws
    restore_rng_state(state)
    got_np = np.random.rand(4)
    got_th = torch.randn(4)
    assert np.allclose(expected_np, got_np)
    assert torch.equal(expected_th, got_th)


ALL7 = [
    "microwave",
    "kettle",
    "bottom burner",
    "light switch",
    "top burner",
    "slide cabinet",
    "hinge cabinet",
]


def test_clip_episode_k7_excludes_leftovers():
    order = ALL7
    scored, k, flags = clip_episode_to_p4(order, ALL7)
    assert k == K_MAX == 4
    assert scored == ALL7[:4]
    for t in ALL7[:4]:
        assert flags[t] == 1
    for t in ALL7[4:]:
        assert flags[t] is None


def test_clip_episode_k4_leftovers_excluded():
    order = ALL7[:4]
    scored, k, flags = clip_episode_to_p4(order, ALL7)
    assert k == 4
    assert scored == order
    for t in order:
        assert flags[t] == 1
    for t in ALL7[4:]:
        assert flags[t] is None


def test_clip_episode_k3_counts_failures():
    order = ALL7[:3]
    scored, k, flags = clip_episode_to_p4(order, ALL7)
    assert k == 3
    assert scored == order
    for t in order:
        assert flags[t] == 1
    for t in ALL7[3:]:
        assert flags[t] == 0
    assert None not in flags.values()


def test_p4_same_as_uncapped_for_k_le_4():
    for n in range(0, 5):
        order = ALL7[:n]
        _, k, _ = clip_episode_to_p4(order, ALL7)
        assert k == n
        assert (k >= 4) == (len(order) >= 4)


def test_classify_p4_outcome():
    for k in (0, 1, 2, 3):
        assert classify_p4_outcome(k) == "miss_subtask"
    for k in (4, 5, 6, 7):
        assert classify_p4_outcome(k) == "success_p4"


def test_episode_k_capped_from_order():
    ep = {
        "completion_order": ALL7,
        "num_tasks_completed": 7,
    }
    assert episode_k_capped(ep) == 4


if __name__ == "__main__":
    test_wilson_60_of_100()
    test_wilson_0_of_100()
    test_wilson_100_of_100()
    test_wilson_zero_trials()
    test_wilson_stays_in_unit_interval()
    test_timing_sample_stats_p95()
    test_warmup_calls_discarded_from_timed_stats()
    test_timing_methodology_keys()
    test_rng_restore_after_warmup_noise()
    test_clip_episode_k7_excludes_leftovers()
    test_clip_episode_k4_leftovers_excluded()
    test_clip_episode_k3_counts_failures()
    test_p4_same_as_uncapped_for_k_le_4()
    test_classify_p4_outcome()
    test_episode_k_capped_from_order()
    print("ok")
