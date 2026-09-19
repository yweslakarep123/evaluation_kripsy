---
name: Laporan mekanisme gangguan
overview: "Pilot Jalur A selesai: Protocol C, Δ=0.2094 rad, ε=0.05/W=3, 20 pairs × 4 NFE. Hasil + plot tersimpan; perlu keputusan kalibrasi delta sebelum scale-up."
todos:
  - id: gates-closed
    content: "Semua gate pra-protokol ditutup"
    status: completed
  - id: implement-pilot
    content: "Pilot 20 pairs × 4 NFE selesai (plot + tabel)"
    status: completed
  - id: report-pilot
    content: "Laporkan hasil pilot + usulan kalibrasi delta"
    status: completed
  - id: decide-scaleup
    content: "Tunggu keputusan user: scale-up / kalibrasi ulang delta / Jalur B"
    status: pending
isProject: false
---

# Pilot Jalur A — Selesai

## Setup terkunci

- Protocol C, microwave qpos kick `Δ = +0.2094` (10% joint span), ε=0.05, W=3
- 20 pasangan intersection (seed=42), NFE ∈ {1,8,32,100}
- Output: [disturbance_recovery/](data/kitchen_eval_plots/nfe100/disturbance_recovery/)

## Hasil utama (lihat `pilot_jalurA_summary.md`)

| NFE | L (ms) | n_inj | n_rec | mean R | mean T_wall | SR_ctrl | SR_dist |
|----:|-------:|------:|------:|-------:|------------:|--------:|--------:|
| 1 | 13.9 | 16 | 4 | 76.8 | 6512 | 0.70 | 0.50 |
| 8 | 117.2 | 15 | 9 | 72.9 | 8064 | 0.75 | 0.60 |
| 32 | 499.5 | 15 | 10 | 52.9 | 11145 | 0.75 | 0.70 |
| 100 | 1579.8 | 19 | 14 | 47.4 | 23205 | 0.95 | 0.95 |

## Kalibrasi

- Inject rate ~75–80% (Protocol C ≠ log eval lama → ~4–5/20 tidak attempt)
- Recovery-to-nominal rendah (25–67% dari yang di-inject) — Δ mungkin agak besar untuk metrik R ketat
- SR drop sedang (≈0.10–0.20), bukan gagal total → Δ tidak katastrofik
- `R` tidak naik dengan NFE (justru cenderung turun); `T_wall` naik tajam karena L — sesuai pemisahan dua lapis

## Plot

- `01_jalurA_recovery_vs_latency.{png,pdf}`
- `02_jalurA_sr_control_vs_disturbed.{png,pdf}`
