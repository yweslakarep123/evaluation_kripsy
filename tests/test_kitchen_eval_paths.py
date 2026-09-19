#!/usr/bin/env python3
"""Tests for kitchen_eval_paths helpers (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kitchen_eval_paths import (  # noqa: E402
    _fp_dir_names,
    normalize_fp_seed,
)
from analyze_kitchen_nfe100 import DIR_RE, DIR_RE_COMPACT, DIR_RE_NATIVE  # noqa: E402


def test_normalize_fp_seed():
    assert normalize_fp_seed("seed42") == "42"
    assert normalize_fp_seed("baseline_43") == "43"
    assert normalize_fp_seed("44") == "44"


def test_fp_dir_names_include_best_val_layout():
    names = _fp_dir_names("seed42", 7, 0)
    assert "seed_seed42_nfe7_sseed0" in names
    assert "seed42_NFE7" in names


def test_dir_regex_nfe_and_native():
    m = DIR_RE.match("seed_seed42_nfe7_sseed0")
    assert m is not None
    assert m.group("seed") == "seed42"
    assert int(m.group("nfe")) == 7

    m2 = DIR_RE_COMPACT.match("seed42_NFE7")
    assert m2 is not None
    assert m2.group("seed") == "42"

    m3 = DIR_RE_NATIVE.match("seed_train0_sseed0")
    assert m3 is not None
    assert m3.group("seed") == "train0"
    assert DIR_RE.match("seed_train0_sseed0") is None


if __name__ == "__main__":
    test_normalize_fp_seed()
    test_fp_dir_names_include_best_val_layout()
    test_dir_regex_nfe_and_native()
    print("ok")
