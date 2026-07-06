# Kitchen Multitask Evaluation Experiment

Repositori ini berisi implementasi evaluasi **Franka Kitchen** untuk membandingkan **Diffusion Policy** dan **FlowPolicy** dengan protokol kombinatorial 4-subtask berurutan.

## Struktur Proyek

```
experiment/
├── diffusion_policy/          # Diffusion Policy (Chi et al., 2023)
│   ├── eval_kitchen_combinations.py
│   ├── data/kitchen/          # Dataset + all_init_qpos.npy (566 inits)
│   └── data/kitchen_combo_eval/   # Hasil evaluasi
├── kripsy12/
│   └── FlowPolicy/            # FlowPolicy baseline
│       ├── eval_kitchen_combinations.py
│       └── data/kitchen/      # Dataset (mirror diffusion_policy)
├── scripts/
│   └── run_all_kitchen_combo_eval.sh   # Orkestrator full run
└── logs/                      # Log evaluasi (dibuat saat run)
```

## Protokol Evaluasi

Evaluasi dirancang untuk mengukur performa model pada **semua kombinasi 4-subtask** dari 7 task Kitchen:

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

- **35 kombinasi** — C(7, 4) = 35 set subtask
- **50 episode per kombinasi** — urutan 4 task di-counterbalance (24 permutasi × 2 + 2 acak)
- **3 checkpoint seed per model** — train0/1/2 (DP) atau baseline_42/43/44 (FlowPolicy)
- **1.750 episode per seed**, **5.250 episode per model**

Setiap episode:

- Init state dari `all_init_qpos.npy` / `all_init_qvel.npy`
- Env `KitchenSequential4V0` — 4 subtask **harus** selesai berurutan
- Max 280 steps, video MP4, log joint trajectory (actual vs predicted)

### Metrik

| Metrik | Deskripsi |
|--------|-----------|
| **p1–p4** | Cumulative sequential success: p_k = fraksi episode dengan ≥ k task pertama selesai berurutan |
| **Per-task success** | Success rate tiap subtask di semua episode di mana task tersebut muncul |
| **Per-task duration** | Wall-clock ms dari task sebelumnya selesai sampai task selesai |
| **Inference latency** | Waktu `predict_action` per policy call (ms) |
| **Joint trajectory** | CSV per episode: `seed, combination_id, episode_id, timestep, joint_idx, actual_qpos, predicted_qpos` |

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

## Menjalankan Evaluasi

Set `MUJOCO_GL=egl` untuk headless Linux (GPU rendering).

### Smoke test (1 kombinasi, 2 episode)

```bash
conda activate robodiff
cd diffusion_policy
MUJOCO_GL=egl python eval_kitchen_combinations.py \
  --smoke --device cuda:0 \
  -m diffusion_policy_transformer \
  -c data/diffusion_policy_transformer/train0/epoch=*.ckpt
```

### Diffusion Policy — full run

```bash
conda activate robodiff
cd diffusion_policy

# Transformer (3 seeds)
MUJOCO_GL=egl python eval_kitchen_combinations.py \
  --model diffusion_policy_transformer \
  --output_root data/kitchen_combo_eval \
  --device cuda:0 --resume

# CNN (3 seeds)
MUJOCO_GL=egl python eval_kitchen_combinations.py \
  --model diffusion_policy_cnn \
  --output_root data/kitchen_combo_eval \
  --device cuda:0 --resume
```

### FlowPolicy — full run

```bash
conda activate flowpolicy-kitchen
cd kripsy12/FlowPolicy
MUJOCO_GL=egl python eval_kitchen_combinations.py \
  --output_root data/kitchen_combo_eval/flowpolicy \
  --device cuda:0 --resume
```

### Semua model sekaligus

```bash
bash scripts/run_all_kitchen_combo_eval.sh          # full run
bash scripts/run_all_kitchen_combo_eval.sh --smoke  # smoke test
```

### CLI flags

| Flag | Default | Deskripsi |
|------|---------|-----------|
| `--smoke` | off | 1 kombinasi, 2 episode |
| `--resume` | on | Skip kombinasi/episode yang sudah selesai |
| `--no-resume` | | Paksa re-run semua |
| `--combination_id N` | all 35 | Hanya jalankan kombinasi N |
| `--n_episodes_per_combo` | 50 | Episode per kombinasi |
| `-c / --checkpoints` | auto-scan | Path checkpoint (bisa glob) |
| `--device` | cuda:0 | GPU device |

## Struktur Output

```
data/kitchen_combo_eval/
  diffusion_policy_transformer/
    seed_train0/
      combination_00/
        metrics.json              # mean±std 50 episode: p1-p4, per-task, timing
        episodes/
          ep_000.json               # metadata episode
          ep_000_joints.csv         # joint trajectory
          ep_000.mp4                # video rollout
      combination_01/ ...
      seed_summary.json             # agregat 35 kombinasi
      seed_report.txt               # tabel human-readable
    seed_train1/ ...
    model_summary.json              # mean±std across 3 seeds
    model_summary.txt
  diffusion_policy_cnn/ ...
  flowpolicy/ ...
```

### Contoh `metrics.json` per kombinasi

```json
{
  "combination_id": 0,
  "tasks": ["bottom burner", "top burner", "light switch", "slide cabinet"],
  "n_episodes": 50,
  "sequential_pk": {
    "p1": {"mean": 0.82, "std": 0.39, "n_samples": 50},
    "p2": {"mean": 0.64, "std": 0.49, "n_samples": 50},
    "p3": {"mean": 0.42, "std": 0.50, "n_samples": 50},
    "p4": {"mean": 0.28, "std": 0.45, "n_samples": 50}
  },
  "per_task_success": { "...": {"mean": 0.56, "std": 0.50, "n_samples": 50} },
  "per_task_duration_ms": { "...": {"mean": 12400, "std": 3200, "n_samples": 28} },
  "inference_latency_ms": {"mean": 45.2, "std": 8.1, "n_samples": 3500}
}
```

Cross-seed summary menggunakan **mean-of-seed-means** (bukan pool 5.250 episode).

## Skala & Estimasi

| Model | Seeds | Total episodes |
|-------|-------|----------------|
| DP Transformer | 3 | 5.250 |
| DP CNN | 3 | 5.250 |
| FlowPolicy | 3 | 5.250 |
| **Total** | | **15.750** |

- **Runtime:** ~30–60 detik/episode → ~130–260 jam GPU total
- **Storage video:** ~30 GB per model (~2 MB/episode)
- **Joint CSV:** ~3–4 GB per model (gzip recommended untuk analisis offline)

Gunakan `--resume` agar evaluasi yang terputus dapat dilanjutkan tanpa re-run episode yang sudah selesai.

## File Implementasi Utama

| File | Peran |
|------|-------|
| `*/common/kitchen_combo_protocol.py` | 35 kombinasi, permutasi balanced, agregasi metrik, laporan |
| `*/env/kitchen/kitchen_sequential_v0.py` | Env 4-subtask berurutan |
| `*/env_runner/kitchen_combo_eval_runner.py` | Episode loop, joint CSV, video, timing |
| `*/eval_kitchen_combinations.py` | CLI entry point |
| `scripts/run_all_kitchen_combo_eval.sh` | Orkestrator multi-model |

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

Runner memanggil `env.close()` setiap episode. Jangan reuse env antar episode — ini wajib untuk run 1.750+ episode per seed.

## Referensi

- Diffusion Policy: Chi et al., 2023 — [paper](https://diffusion-policy.cs.columbia.edu/)
- Kitchen env: relay-policy-learning / `KitchenAllV0`
- Protokol eval porting: `kripsy12/KITCHEN_EVAL_PORTING.md`
