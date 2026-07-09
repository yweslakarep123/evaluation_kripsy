# Kitchen Multitask Evaluation Experiment

Repositori ini berisi implementasi evaluasi **Franka Kitchen** untuk membandingkan **Diffusion Policy** dan **FlowPolicy** dengan protokol flat 100 episode per checkpoint.

## Struktur Proyek

```
experiment/
├── diffusion_policy/          # Diffusion Policy (Chi et al., 2023)
│   ├── eval_kitchen.py
│   ├── data/kitchen/          # Dataset + all_init_qpos.npy (566 inits)
│   └── data/kitchen_eval/     # Hasil evaluasi
├── kripsy12/
│   └── FlowPolicy/            # FlowPolicy baseline
│       ├── eval_kitchen.py
│       └── data/kitchen/      # Dataset (mirror diffusion_policy)
├── scripts/
│   └── run_kitchen_eval_1000.sh   # Orkestrator full run
└── logs/                      # Log evaluasi (dibuat saat run)
```

## Protokol Evaluasi

Evaluasi flat pada environment Kitchen standar (`KitchenAllV0` — 7 task, any-order):

| Task | Nama |
|------|------|
| 1 | bottom burner |
| 2 | top burner |
| 3 | light switch |
| 4 | slide cabinet |
| 5 | hinge cabinet |
| 6 | microwave |
| 7 | kettle |

**Desain eksperimen:**

- **100 episode per checkpoint**
- **3 checkpoint per model** — train0/1/2 (DP) atau baseline_42/43/44 (FlowPolicy)
- **3 model** — DP Transformer, DP CNN, FlowPolicy
- **900 episode total**

Setiap episode:

- Init state deterministik dari `all_init_qpos.npy` / `all_init_qvel.npy` (`init_idx = episode_idx % 566`)
- Env `KitchenAllV0` — 7 subtask, urutan penyelesaian bebas
- Max 280 steps, video MP4 semua episode
- Log numerik detail per episode di `trajectory_logs/` (NPZ + TXT vs demo GT)

### Metrik

| Metrik | Deskripsi |
|--------|-----------|
| **p1–p7** | Cumulative multistage success (Diffusion Policy style) |
| **Per-task success** | Success rate tiap subtask across 100 episode |
| **Per-task duration** | Wall-clock ms dari task sebelumnya selesai sampai task selesai |
| **Inference latency** | Waktu `predict_action` per policy call (ms) |

### Trajectory logs

Setiap episode menulis ke `trajectory_logs/`:

| File | Isi |
|------|-----|
| `ep_XXXX.npz` | Arrays numerik + metadata window/horizon, obs terpecah, label joint/action, alignment indices |
| `ep_XXXX_detail.txt` | Versi human-readable: legend joint/action, tabel urutan demo & rollout, detail per env step |
| `SCHEMA.txt` | Dokumentasi key/shape NPZ |

NPZ mencakup: `action_pred`, `action_executed`, `policy_obs`, `qp`/`qv`/`obj_qp`/`obj_qv`, `demo_obs`/`demo_action`, dan error vs demo GT per env step.

### Analisis trajectory logs

Setelah eval selesai, plot dan ringkasan numerik:

```bash
conda activate robodiff
cd /home/daffa/Documents/experiment

# Satu episode
python scripts/analyze_kitchen_trajectory.py \
  --npz diffusion_policy/data/kitchen_eval/diffusion_policy_transformer/seed_train0/trajectory_logs/ep_0000.npz \
  --output_dir /tmp/kitchen_analysis/ep_0000

# Semua episode satu seed
python scripts/analyze_kitchen_trajectory.py \
  --seed_dir diffusion_policy/data/kitchen_eval/diffusion_policy_transformer/seed_train0 \
  --output_dir /tmp/kitchen_analysis/seed_train0
```

Output analisis: `joints_vs_demo.png`, `actions_vs_demo.png`, `errors_vs_demo.png`, `action_pred_heatmap.png`, `analysis_summary.json`.

## Setup

### Conda environments

| Model | Env | Install |
|-------|-----|---------|
| Diffusion Policy | `robodiff` | `conda env create -f diffusion_policy/conda_environment.yaml` |
| FlowPolicy | `flowpolicy-kitchen` | Lihat `kripsy12/README.md` |

### Dataset Kitchen

Pastikan data ada di kedua lokasi:

```
diffusion_policy/data/kitchen/
  all_init_qpos.npy
  all_init_qvel.npy
  observations_seq.npy
  actions_seq.npy
  existence_mask.npy
  kitchen_demos_multitask/

kripsy12/FlowPolicy/data/kitchen/   # sama
```

Unduh dari [Diffusion Policy training data](https://diffusion-policy.cs.columbia.edu/data/training/) atau gunakan `kripsy12/scripts/download_kitchen_demos.sh`.

### Checkpoints

| Model | Path default |
|-------|--------------|
| DP Transformer | `diffusion_policy/data/diffusion_policy_transformer/train{0,1,2}/epoch=*.ckpt` |
| DP CNN | `diffusion_policy/data/diffusion_policy_cnn/train{0,1,2}/epoch=*.ckpt` |
| FlowPolicy | `kripsy12/FlowPolicy/data/outputs/baseline_{42,43,44}/*.ckpt` |

Dataset Kitchen untuk DP dapat di-symlink dari FlowPolicy:

```bash
mkdir -p diffusion_policy/data
ln -sfn ../kripsy12/FlowPolicy/data/kitchen diffusion_policy/data/kitchen
```

## Menjalankan Evaluasi

Set `MUJOCO_GL=egl` untuk headless Linux (GPU rendering).

### Smoke test (10 episode, 1 checkpoint per model)

```bash
bash scripts/run_kitchen_eval_1000.sh --smoke
```

### Diffusion Policy — per model

```bash
conda activate robodiff
cd diffusion_policy

MUJOCO_GL=egl python eval_kitchen.py \
  --model diffusion_policy_transformer \
  --output_root data/kitchen_eval \
  --device cuda:0

MUJOCO_GL=egl python eval_kitchen.py \
  --model diffusion_policy_cnn \
  --output_root data/kitchen_eval \
  --device cuda:0
```

### FlowPolicy — full run

```bash
conda activate flowpolicy-kitchen
cd kripsy12/FlowPolicy
MUJOCO_GL=egl python eval_kitchen.py \
  --output_root data/kitchen_eval/flowpolicy \
  --device cuda:0
```

### Semua model sekaligus

```bash
bash scripts/run_kitchen_eval_1000.sh          # full run (900 episode)
bash scripts/run_kitchen_eval_1000.sh --smoke  # smoke test
```

### CLI flags

| Flag | Default | Deskripsi |
|------|---------|-----------|
| `--smoke` | off | 10 episode, 1 checkpoint |
| `--n_episodes` | 100 | Episode per checkpoint |
| `--save-trajectory-logs` | on | Simpan NPZ+TXT per episode |
| `--no-save-trajectory-logs` | — | Matikan trajectory logs |
| `--dataset_dir` | data/kitchen | Folder all_init_qpos.npy |
| `--overwrite/--no-overwrite` | no-overwrite | Paksa re-run jika output ada |
| `-c / --checkpoints` | auto-scan | Path checkpoint (bisa glob) |
| `--device` | cuda:0 | GPU device |

## Struktur Output

```
data/kitchen_eval/
  diffusion_policy_transformer/
    seed_train0/
      eval_metrics.json
      eval_report.txt
      media/ep_0000.mp4 ... ep_0099.mp4
      trajectory_logs/
        SCHEMA.txt
        ep_0000.npz
        ep_0000_detail.txt
        ...
    seed_train1/ ...
    seed_train2/ ...
    summary.json
  diffusion_policy_cnn/ ...
  flowpolicy/
    seed_baseline_42/ ...
```

## Skala & Estimasi

| Model | Checkpoints | Total episodes |
|-------|-------------|----------------|
| DP Transformer | 3 | 300 |
| DP CNN | 3 | 300 |
| FlowPolicy | 3 | 300 |
| **Total** | | **900** |

- **Runtime:** ~10–30 detik/episode → ~1–4 jam GPU total
- **Storage video:** ~200 MB per seed (~2 MB/episode)
- **Storage trajectory logs:** ~50–150 MB per seed

## File Implementasi Utama

| File | Peran |
|------|-------|
| `*/env_runner/kitchen_lowdim_eval_runner.py` | Episode loop, video, timing, metrik, trajectory logs |
| `*/eval_kitchen.py` | CLI entry point |
| `scripts/run_kitchen_eval_1000.sh` | Orkestrator multi-model |

## Known Issues

### FlowPolicy — MuJoCo XML error

Jika muncul error:

```
ValueError: XML Error: top-level default class 'main' cannot be renamed
```

Ini karena **mujoco 3.8+** tidak kompatibel dengan XML Kitchen lama. Diffusion Policy (`robodiff`, mujoco 2.3.7) tidak terpengaruh.

Solusi yang dicoba di dokumentasi ReinFlow:

```bash
pip install dm_control==1.0.16 mujoco==3.1.6
```

Lihat `kripsy12/ReinFlow/docs/KnownIssues.md` untuk detail lebih lanjut.

### Memory (MuJoCo leak)

Runner memanggil `env.close()` setiap episode. Jangan reuse env antar episode — ini wajib untuk run panjang per checkpoint.

## Referensi

- Diffusion Policy: Chi et al., 2023 — [paper](https://diffusion-policy.cs.columbia.edu/)
- Kitchen env: relay-policy-learning / `KitchenAllV0`
- Protokol eval porting: `kripsy12/KITCHEN_EVAL_PORTING.md`
