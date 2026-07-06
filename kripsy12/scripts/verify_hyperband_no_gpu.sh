#!/usr/bin/env bash
# Verifikasi logika Hyperband TANPA GPU (mock training).
#
#   conda activate flowpolicy-kitchen
#   ./scripts/verify_hyperband_no_gpu.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate flowpolicy-kitchen 2>/dev/null || true
fi

python3 << 'PY'
import json
import pathlib
import tempfile
from unittest.mock import patch

import hyperband_search as hb
from experiment_constants import sample_configs_hyperband
from hyperband_search import (
    _build_train_overrides_hb,
    compute_brackets,
    run_hyperband,
)
import numpy as np

rng = np.random.RandomState(42)
cfgs = sample_configs_hyperband(rng, 5, base_cfg_idx=1000)
assert cfgs[0]["cfg_idx"] == 1000
print("OK: sample_configs_hyperband")

cfg = {
    "cfg_idx": 1000,
    "training.num_epochs": 0,
    "optimizer.lr": 1e-4,
    "dataloader.batch_size": 128,
    "policy.Conditional_ConsistencyFM.num_segments": 2,
    "policy.Conditional_ConsistencyFM.eps": 1e-2,
    "policy.Conditional_ConsistencyFM.delta": 1e-2,
    "n_action_steps": 4,
    "n_obs_steps": 2,
    "policy.diffusion_step_embed_dim": 128,
    "_state_mlp_hidden": 256,
}
odl = _build_train_overrides_hb(
    cfg,
    seed=0,
    profile="standard",
    train_eps=[0],
    val_eps=[1],
    run_dir=pathlib.Path("/tmp/hb"),
    dataset_dir="FlowPolicy/data/kitchen/kitchen_demos_multitask",
    resume_training=False,
    delta_num_epochs=3,
    checkpoint_every=1,
    dataloader_num_workers=0,
)
assert any("flowpolicy_kitchen_lowdim" in x for x in odl)
assert any("task.dataset.dataset_dir=" in x for x in odl)
assert any("policy.obs_mlp_hidden=" in x for x in odl)
assert not any("point_cloud" in x for x in odl)
print("OK: override Hydra lowdim (FlowPolicyLowdim + MJL dataset)")

brs = compute_brackets(81, 3)
assert brs[0].s == 4 and brs[0].n == 81 and brs[0].rungs[-1].r_i == 81
print("OK: compute_brackets cocok Tabel 1 paper (R=81, eta=3)")


def fake_apply(cfg, mb):
    return dict(cfg)


def fake_eval(*, cfg, target_epoch, already_trained, run_dir, **kw):
    v = float(cfg["optimizer.lr"]) + 0.01 * int(cfg["cfg_idx"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "latest.ckpt").write_text("x")
    (run_dir / "training_final.json").write_text(json.dumps({"val_loss_final": v}))
    return v, 0, int(target_epoch)


with tempfile.TemporaryDirectory() as d:
    out, runs = pathlib.Path(d), pathlib.Path(d) / "runs"
    with patch.object(hb, "_evaluate_config_at_rung", side_effect=fake_eval):
        best = run_hyperband(
            out_root=out,
            runs_root=runs,
            R=12,
            eta=2,
            s_min=1,
            s_max=1,
            sampling_seed=7,
            search_train_seed=0,
            search_profile="standard",
            train_eps=[0, 1],
            val_eps=[2],
            dataset_dir="FlowPolicy/data/kitchen/kitchen_demos_multitask",
            checkpoint_every=1,
            dataloader_num_workers=0,
            py="python3",
            train_py=pathlib.Path("/tmp/t.py"),
            cwd_train="/tmp",
            apply_vram_limits_fn=fake_apply,
            max_batch_size=128,
        )
    st = json.loads((out / "hyperband_state.json").read_text())
    assert best and (out / "hyperband_state.json").is_file()
    print(f"OK: run_hyperband end-to-end, pemenang cfg_idx={best['cfg_idx']}")

print("\nSemua verifikasi logika Hyperband LULUS (tanpa GPU).")
PY
