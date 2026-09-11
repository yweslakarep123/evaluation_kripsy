# FlowPolicy — Kitchen Lowdim (7-task)

Implementasi **Flow Policy** untuk **Franka Kitchen multitask lowdim**: observasi **60-dim**, action **9-dim**, ~605 demo MJL. Training memakai [Hydra](https://hydra.cc/) (`flowpolicy_kitchen_lowdim`); evaluasi simulasi 7 subgoal dengan metrik **p1–p7** (protokol `KITCHEN_EVAL_PORTING.md`).

Struktur repositori:

```text
<akar-repo>/                # root Git (folder berisi scripts + FlowPolicy)
├── scripts/                # orkestrator eksperimen (baseline + Hyperband)
│   ├── run_experiment.py
│   ├── run_experiment.sh         # pintasan CLI: baseline lalu Hyperband
│   ├── run_baseline_only.sh      # hanya baseline (6 run default)
│   ├── run_hyperband_only.sh     # hanya Hyperband + rerun pemenang top-1
│   ├── verify_hyperband_no_gpu.sh      # cek logika Hyperband tanpa GPU
│   ├── hyperband_search.py       # implementasi Hyperband (Li et al., 2018)
│   ├── cv_splits.py
│   ├── summarize.py
│   ├── plot_results.py
│   ├── plot_training_convergence.py  # overlay kurva loss 3 seed
│   └── experiment_constants.py
└── FlowPolicy/             # train.py, infer_kitchen_lowdim.py, paket flow_policy_3d
    ├── train.py
    ├── infer_kitchen_lowdim.py
    ├── eval_kitchen.py
    ├── setup.py
    ├── requirements-franka-kitchen.txt
    ├── data/kitchen/kitchen_demos_multitask/   # ~605 demo .mjl (wajib ada)
    └── flow_policy_3d/
```

- Perintah **training tunggal** (`train.py`): dari **`FlowPolicy/`** (folder yang berisi `train.py`).
- **Pipeline eksperimen** (`scripts/run_experiment.py`): dijalankan dari **akar repositori** (folder induk `scripts/` dan `FlowPolicy/`).
- **Mulai di cloud GPU:** lihat [Cloud GPU — mulai di sini](#cloud-gpu--mulai-di-sini).

## Prasyarat

- **Linux** (disarankan Ubuntu 22.04+); eval headless memakai MuJoCo + **EGL** (`MUJOCO_GL=egl`, diset otomatis oleh orkestrator).
- **NVIDIA GPU** dengan driver CUDA yang kompatibel dengan PyTorch.
- **Python 3.10**.
- Akun **Weights & Biases** (opsional: `WANDB_API_KEY` atau `WANDB_MODE=offline`).

## Instalasi (lokal atau cloud VM)

### 1. Buat environment

```bash
conda create -n flowpolicy-kitchen python=3.10 -y
conda activate flowpolicy-kitchen
```

### 2. Pasang PyTorch (sesuaikan versi CUDA host)

Contoh CUDA 12.4 ([PyTorch Get Started](https://pytorch.org/get-started/locally/)):

```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### 3. Dependensi proyek + editable install

```bash
cd FlowPolicy
pip install -U pip
pip install -r requirements-franka-kitchen.txt
pip install -e .
```

**PyTorch3D:** jika `pip install pytorch3d` gagal:

```bash
conda install pytorch3d -c pytorch3d
```

## Dataset

Pipeline default memakai **Kitchen lowdim 7-task** (~605 demo `.mjl`) di:

```text
FlowPolicy/data/kitchen/kitchen_demos_multitask/
```

Path relatif terhadap folder berisi `train.py` (`FlowPolicy/`). Pastikan folder ini ada di instance cloud (clone repo + LFS, atau salin data ke volume).

Config Hydra: **`flowpolicy_kitchen_lowdim`** + task **`kitchen_lowdim_all`** (obs 60-dim, action 9-dim, `abs_action=true`, `n_action_steps=8`).

Override lewat orkestrator:

```bash
--dataset-dir FlowPolicy/data/kitchen/kitchen_demos_multitask
```

---

## Cloud GPU — mulai di sini

Bagian ini untuk **Vast.ai, RunPod, Lambda, atau VM GPU** — jalankan eksperimen nyata di sini.

### 1. Clone & install (sekali per instance)

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>

conda create -n flowpolicy-kitchen python=3.10 -y
conda activate flowpolicy-kitchen
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

cd FlowPolicy
pip install -U pip
pip install -r requirements-franka-kitchen.txt
pip install -e .
cd ..   # kembali ke akar repo
```

### 2. Cek GPU

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Pastikan dataset ada:

```bash
ls FlowPolicy/data/kitchen/kitchen_demos_multitask/*.mjl | wc -l
# Harus ~605
```

### 3. Eksperimen yang direkomendasikan — baseline only (Vast.ai)

Konfigurasi yang dipakai: **baseline yang sudah diperbaiki** (split demo **80% train / 20% val / tanpa test**, UNet `down_dims=[256,512,1024]`, checkpoint eval = `best_val.ckpt`). **Tanpa Hyperband.**

**3 seed training: `42`, `43`, `44`** × profil **`standard`** = **3 run**. Setiap run: train 3000 epoch → eval MuJoCo **50 episode × 3 eval-seed** (`0,42,101`) → metrik **p1–p7** + plot konvergensi.

Dari **akar repo** (setelah `conda activate flowpolicy-kitchen`):

```bash
chmod +x scripts/*.sh
export MUJOCO_GL=egl
export WANDB_MODE=offline

./scripts/run_baseline_only.sh \
  --seeds 42 43 44 \
  --output-dir outputs/vast_baseline \
  --max-batch-size 128 \
  --dataloader-num-workers 4 \
  --skip-inference-videos \
  2>&1 | tee vast_baseline.log
```

`--max-batch-size 128` dan `--dataloader-num-workers 4` sesuai baseline (VRAM 32 GB).

Resume jika instance mati: jalankan **perintah yang sama** ( `--output-dir` sama). Run yang sudah `metrics.json` / `status=ok` dilewati; training terputus dilanjutkan dari `latest.ckpt`.

### 4. Eksperimen penuh (baseline + Hyperband + rerun pemenang)

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/experiment \
  --max-batch-size 64 \
  --dataloader-num-workers 2
```

Hyperband hemat waktu (single-bracket SHA, ~≤2 hari tergantung kecepatan GPU):

```bash
./scripts/run_experiment.sh \
  --output-dir outputs/exp_fast \
  --hyperband-s-max 2 \
  --hyperband-s-min 2 \
  --max-batch-size 64 \
  --dataloader-num-workers 2
```

Hanya Hyperband (tanpa baseline):

```bash
./scripts/run_hyperband_only.sh \
  --output-dir outputs/hyperband_only \
  --max-batch-size 64 \
  --dataloader-num-workers 2
```

### 5. Variabel lingkungan (cloud)

| Variabel | Keterangan |
|----------|------------|
| `WANDB_API_KEY` | Logging W&B online (opsional). |
| `WANDB_MODE=offline` | Tanpa upload W&B. |
| `MUJOCO_GL=egl` | Headless rendering eval (orkestrator sudah set). |
| `DATASET_DIR` | Override path dataset di shell script. |

Contoh:

```bash
export WANDB_API_KEY=...
export MUJOCO_GL=egl
export DATASET_DIR=/data/kitchen/kitchen_demos_multitask
./scripts/run_baseline_only.sh --output-dir outputs/baseline_only
```

### 6. Resume setelah instance mati

Jalankan **perintah yang sama** dengan `--output-dir` yang sama. Run selesai (`metrics.json` atau `status=ok` di `results.csv`) dilewati; Hyperband melanjutkan dari `hyperband_state.json`.

### 7. Agregasi hasil (tanpa training ulang)

```bash
python scripts/summarize.py --output-dir outputs/baseline_only
python scripts/plot_results.py --output-dir outputs/baseline_only
python scripts/plot_training_convergence.py --output-dir outputs/baseline_only
```

Keluaran penting:

| File | Isi |
|------|-----|
| `outputs/.../results.csv` | Metrik per run: `test_p1`…`test_p7`, `test_all_7_success`, `test_p4_paper`. |
| `outputs/.../runs/baseline_seed*_*/metrics.json` | Metrik eval lengkap per run. |
| `outputs/.../runs/baseline_seed*_*/checkpoints/best_val.ckpt` | Checkpoint **eval** (val_loss minimum). |
| `outputs/.../runs/baseline_seed*_*/checkpoints/latest.ckpt` | Checkpoint **resume** (epoch terakhir). |
| `outputs/.../runs/baseline_seed*_*/plots/` | Kurva train/val loss, analisis konvergensi, kriteria checkpoint. |
| `outputs/.../seed_convergence.json` | Apakah 3 seed konvergen secara comparable. |
| `outputs/.../summary.csv`, `plots/` | Agregat statistik + grafik eval. |

### 8. Eval manual satu checkpoint (opsional)

Dari **`FlowPolicy/`**:

```bash
cd FlowPolicy
conda activate flowpolicy-kitchen

MUJOCO_GL=egl python infer_kitchen_lowdim.py \
  --checkpoint runs/<run_name>/checkpoints/best_val.ckpt \
  --metrics-json runs/<run_name>/metrics.json \
  --n-infer-episodes 50 \
  --eval-seeds 0,42,101
```

Hemat disk (tanpa video MP4):

```bash
MUJOCO_GL=egl python infer_kitchen_lowdim.py \
  --checkpoint path/ke/best_val.ckpt \
  --metrics-json path/ke/metrics.json \
  --n-infer-episodes 50 \
  --eval-seeds 0,42,101 \
  --skip-inference-videos
```

### 9. Training tunggal (di luar orkestrator)

Dari **`FlowPolicy/`**:

```bash
cd FlowPolicy
python train.py --config-name=flowpolicy_kitchen_lowdim logging.mode=offline
```

Override umum: `training.device=cuda:0`, `dataloader.batch_size=64`, `training.debug=true` (epoch dibatasi).

---

## Referensi perintah penting

Semua perintah di bawah ini dijalankan dari **akar repositori**, kecuali `train.py` / `infer_kitchen_lowdim.py` / `eval_kitchen.py` dari **`FlowPolicy/`**.

### Persiapan (setiap sesi terminal baru)

```bash
cd /path/ke/kripsy12
conda activate flowpolicy-kitchen
export MUJOCO_GL=egl   # opsional; orkestrator set otomatis saat infer
```

```bash
python scripts/run_experiment.py --help
chmod +x scripts/*.sh   # sekali saja
```

### Verifikasi logika Hyperband (tanpa GPU, cepat)

Tidak melatih model — hanya cek bracket / `hyperband_state.json`:

```bash
./scripts/verify_hyperband_no_gpu.sh
```

### Pipeline eksperimen — tiga mode

| Mode | Skrip | Isi |
|------|-------|-----|
| Baseline + Hyperband + rerun pemenang | `./scripts/run_experiment.sh` | Fase 1→2→3 |
| Hanya baseline (3 seed × 1 profil) | `./scripts/run_baseline_only.sh` | Lewati Hyperband; di Vast.ai: `--seeds 42 43 44` |
| Hanya Hyperband + rerun pemenang | `./scripts/run_hyperband_only.sh` | Lewati baseline |

Setara Python langsung:

```bash
python scripts/run_experiment.py \
  --output-dir outputs/experiment \
  --dataset-dir FlowPolicy/data/kitchen/kitchen_demos_multitask \
  --hyperband-max-epochs 3000 \
  --hyperband-eta 3 \
  --hyperband-s-min 0 \
  --hyperband-seed 99 \
  --hyperband-search-train-seed 0 \
  --hyperband-search-profile standard
```

Flag eksklusif: `--baseline-only` atau `--hyperband-only` (maksimal satu).

### Opsi VRAM (GPU cloud)

| VRAM | `--max-batch-size` | `--dataloader-num-workers` |
|------|--------------------|----------------------------|
| **≥ 24 GB** (termasuk 32 GB) | `128` (baseline) | `4` |
| **16 GB** | `64` (mulai di sini) | `2` |
| **8 GB** | `16`–`32` | `0` |

Contoh **16 GB**:

```bash
./scripts/run_baseline_only.sh \
  --output-dir outputs/baseline_only \
  --max-batch-size 64 \
  --dataloader-num-workers 2
```

Hemat waktu eval (tanpa video):

```bash
  --skip-inference-videos
```

---

## Menjalankan training

Dari **`FlowPolicy/`**:

```bash
python train.py --config-name=flowpolicy_kitchen_lowdim
```

| Override | Keterangan |
|----------|------------|
| `training.device=cuda:0` | Device PyTorch. |
| `training.debug=true` | Mode debug (epoch/step dibatasi). |
| `logging.mode=offline` | W&B tanpa upload. |
| `task.dataset.dataset_dir=...` | Path demo MJL jika bukan default. |

**Catatan:** saat training, `env_runner` = `KitchenNullRunner` (tidak spawn MuJoCo). Beban MuJoCo hanya saat **infer/eval** (`infer_kitchen_lowdim.py`).

Checkpoint dan log Hydra: `FlowPolicy/data/outputs/` atau sesuai `hydra.run.dir`.

---

## Pipeline eksperimen (baseline + Hyperband, tanpa k-fold)

Pelatihan **tidak** memakai k-fold. Episode dibagi **sekali** train/val **tanpa test** (`scripts/cv_splits.py`, default **80/20/0**). Eval policy memakai simulasi MuJoCo, bukan holdout demo.

Skrip **`scripts/run_experiment.py`** menjalankan tiga fase **berurutan**:

| Fase | Isi | Jumlah run (default) |
|------|-----|------------------------|
| **1. Baseline** | Hiperparameter default × **3 seed** × **2 profil** | **6** |
| **2. Hyperband** | Random search + Successive Halving, `val_loss`, **1 seed × 1 profil** | tergantung `R`, `eta`, `s_min`, `s_max` |
| **3. Rerun pemenang** | Top-1 Hyperband × **3 seed × 2 profil** @ `R` epoch + eval penuh | **6** |

Profil preprocessing: **`standard`** (noise observasi) dan **`minimal`**.

Eval setelah training: **`infer_kitchen_lowdim.py`** — 7 task KitchenAllV0, metrik **p1–p7**, `test_all_7_success`, `test_p4_paper`, agregasi **3 eval-seed** (`0,42,101`).

**Checkpoint:** kriteria seleksi adalah **`min val_loss`**. Inferensi memakai `best_val.ckpt`; `latest.ckpt` hanya untuk resume. Setiap run menulis `plots/train_val_loss.{png,pdf}`, `plots/convergence_analysis.{png,pdf}`, `plots/checkpoint_selection.{png,pdf}`, plus `checkpoint_selection.json` dan `convergence.json`.

**UNet:** `policy.down_dims=[256,512,1024]` (arsitektur default `ConditionalUnet1D`).

### Hyperband (Li et al., 2018) singkat

- `R` (`--hyperband-max-epochs`): resource = epoch (default **3000**).
- `eta` (`--hyperband-eta`): default **3**.
- Pemenang: **val_loss terkecil** di semua evaluasi intermediate.
- Inference rollout **hanya** pada baseline + rerun pemenang.

### Anggaran waktu Hyperband (R=3000, eta=3)

| `s_max` | `s_min` | Bracket | Baseline-equivalent (~) |
|---|---|---|---|
| 2 | 2 | single-bracket SHA | **~4.7** |
| 2 | 0 | s = 2, 1, 0 | ~19 |
| 7 (native) | 0 | semua bracket | ~44 |

Untuk fit **≤ ~2 hari** (asumsi ~8 jam/baseline run): **`--hyperband-s-max 2 --hyperband-s-min 2`**.

### Opsi CLI yang sering dipakai

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--dataset-dir` | `FlowPolicy/data/kitchen/kitchen_demos_multitask` | Demo MJL (relatif ke `FlowPolicy/`). |
| `--seeds` | `0 42 101` | Seed training. Untuk Vast.ai baseline pakai **`42 43 44`**. |
| `--profiles` | `standard minimal` | Profil preprocessing. |
| `--cv-seed` | `12345` | Seed pembagian episode train/val. |
| `--train-frac` / `--val-frac` / `--test-frac` | `0.8` / `0.2` / `0.0` | Fraksi split demo MJL (tanpa test). |
| `--n-infer-episodes` | `50` | Episode eval per eval-seed. |
| `--output-dir` | `outputs/experiment` | Folder keluaran (relatif akar repo). |
| `--max-batch-size` | `128` | Plafon batch train/val. |
| `--dataloader-num-workers` | `4` | Workers DataLoader. |
| `--baseline-only` | (off) | Hanya baseline. |
| `--hyperband-only` | (off) | Hanya Hyperband + rerun pemenang. |
| `--hyperband-max-epochs` | `3000` | Resource Hyperband `R`. |
| `--hyperband-eta` | `3` | Rasio downsampling antar-rung. |
| `--hyperband-s-min` / `--hyperband-s-max` | `0` / auto | Cap bracket (lihat anggaran waktu). |
| `--skip-inference-videos` | (off) | Tanpa MP4 eval (hemat waktu/disk). |
| `--checkpoint-every` | `200` | Frekuensi checkpoint (resume). |

### Keluaran

Di `--output-dir`:

- `configs.json` — baseline + meta eksperimen.
- `hyperband_state.json` — state Hyperband (resume).
- `cv_splits.json` / `episode_split.json` — partisi episode train/val (tanpa test).
- `results.csv` — metrik baseline (`cfg_idx=-1`) dan rerun Hyperband (`cfg_idx=-3`).
- `seed_convergence.json` + `plots/seed_convergence/` — overlay 3 seed.
- `runs/baseline_seed<seed>_<profile>/` — Hydra output, `checkpoints/best_val.ckpt` (eval) + `latest.ckpt` (resume), `loss_history.csv`, `plots/`, `checkpoint_selection.json`, `training_final.json`, `metrics.json`.
- `runs/hb_best_seed<seed>_<profile>/` — rerun pemenang Hyperband.
- `runs/hb_cfg<idx>/` — run Hyperband intermediate (folder ter-cull dihapus otomatis).

### Resume setelah mesin mati

- Run dilewati jika **`metrics.json`** ada, atau baris **`status=ok`** di `results.csv`.
- Training terputus (`latest.ckpt`, belum `training_final.json`) → **dilanjutkan** (`training.resume=true`).
- Training selesai, infer belum → hanya **`infer_kitchen_lowdim.py`** dijalankan (checkpoint = `best_val.ckpt` jika ada).
- Hyperband intermediate → resume via **`hyperband_state.json`**.

---

## Menjalankan di [Vast.ai](https://vast.ai/)

Copy-paste di bawah ini untuk **baseline yang sudah diperbaiki**, seed training **42, 43, 44**, di instance Vast.ai. Jangan jalankan Hyperband kecuali Anda memang mau.

### Pilih instance

- Template **Ubuntu 22.04 + CUDA 12.x + PyTorch**, atau image minimal lalu install manual.
- Instance ini **VRAM 32 GB** → `--max-batch-size 128` dan `--dataloader-num-workers 4` (nilai baseline).

### 1. Masuk ke folder repo

Sesuaikan path jika clone Anda bukan `/workspace/kripsy12`:

```bash
cd /workspace/kripsy12    # akar repo (ada folder scripts/ dan FlowPolicy/)
source ~/miniforge3/etc/profile.d/conda.sh   # sesuaikan path conda
conda activate flowpolicy-kitchen
```

Kalau environment belum ada, install dulu (lihat [Cloud GPU — mulai di sini](#cloud-gpu--mulai-di-sini) langkah 1–2), lalu:

```bash
cd FlowPolicy
pip install -U pip
pip install -r requirements-franka-kitchen.txt
pip install -e .
cd ..
ls FlowPolicy/data/kitchen/kitchen_demos_multitask/*.mjl | wc -l
# harus ~605
```

### 2. Jalankan baseline (perintah utama)

Tiga run berurutan: `baseline_seed42_standard` → `baseline_seed43_standard` → `baseline_seed44_standard`.

```bash
chmod +x scripts/*.sh
export MUJOCO_GL=egl
export WANDB_MODE=offline

./scripts/run_baseline_only.sh \
  --seeds 42 43 44 \
  --output-dir outputs/vast_baseline \
  --max-batch-size 128 \
  --dataloader-num-workers 4 \
  --skip-inference-videos \
  2>&1 | tee vast_baseline.log
```

Yang sudah aktif di perintah itu:

| Item | Nilai |
|------|--------|
| Mode | `--baseline-only` (tanpa Hyperband) |
| Seed training | **42, 43, 44** |
| Profil | `standard` |
| Split demo | 80% train / 20% val / **tanpa test** |
| UNet | `down_dims=[256,512,1024]` |
| Batch | **128** (sama dengan baseline) |
| Checkpoint eval | `best_val.ckpt` (`min val_loss`); `latest.ckpt` hanya resume |
| Plot | kurva train/val, konvergensi, overlay 3 seed |

`--skip-inference-videos` menghemat disk/waktu eval (metrik p1–p7 tetap dihitung). Hapus flag itu jika Anda butuh MP4.

**GPU 16 GB** (OOM): `--max-batch-size 64 --dataloader-num-workers 2`.

Dataset di volume terpisah:

```bash
export DATASET_DIR=/data/kitchen/kitchen_demos_multitask
./scripts/run_baseline_only.sh --seeds 42 43 44 --output-dir outputs/vast_baseline ...
```

### 3. Resume setelah instance mati

Jalankan **langkah 2 yang sama** (path `--output-dir` sama). Jangan ganti seed atau output-dir.

### 4. Cek hasil di instance

```bash
ls outputs/vast_baseline/runs/
# baseline_seed42_standard  baseline_seed43_standard  baseline_seed44_standard

ls outputs/vast_baseline/runs/baseline_seed42_standard/plots/
# train_val_loss.png  convergence_analysis.png  checkpoint_selection.png

ls outputs/vast_baseline/runs/baseline_seed42_standard/checkpoints/
# best_val.ckpt  latest.ckpt
```

Plot overlay 3 seed (otomatis di akhir orkestrator; bisa diulang tanpa training):

```bash
python scripts/plot_training_convergence.py --output-dir outputs/vast_baseline
python scripts/summarize.py --output-dir outputs/vast_baseline
```

### 5. Unduh ke laptop

```bash
scp -r vast_instance:/workspace/kripsy12/outputs/vast_baseline ./outputs/
scp vast_instance:/workspace/kripsy12/vast_baseline.log .
```

Ganti `vast_instance` dengan host SSH Vast.ai Anda.

### Headless MuJoCo

Eval memakai `MUJOCO_GL=egl`. Training **tidak** membutuhkan display (null runner).

---

## Metrik eval (p1–p7)

| Metrik CSV | Arti singkat |
|------------|--------------|
| `test_p1` … `test_p7` | Fraksi episode yang menyelesaikan ≥ N dari 7 task |
| `test_all_7_success` | Fraksi episode yang menyelesaikan **semua 7 task** |
| `test_p4_paper` | Subset 4-task (microwave, kettle, light switch, slide cabinet) |
| `test_mean_inference_latency_ms` | Latensi inferensi policy |

Detail protokol: `KITCHEN_EVAL_PORTING.md` di akar repo.

---

## Push ke GitHub

```bash
git add README.md scripts FlowPolicy
git commit -m "Kitchen lowdim pipeline"
git push
```

Hindari commit folder besar (`data/outputs/`, checkpoint). Gunakan `.gitignore`.

## Lisensi / atribusi

Sesuaikan dengan lisensi proyek upstream Anda.

## Kontak

Sesuaikan dengan informasi kontributor Anda.
