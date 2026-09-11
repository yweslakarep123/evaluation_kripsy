"""Kurva loss, analisis konvergensi, dan artefak pemilihan checkpoint.

Dipakai oleh ``train.py`` di akhir epoch (dan tiap checkpoint) tanpa GUI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOSS_HISTORY_FIELDS = ["epoch", "train_loss", "val_loss", "lr", "is_best_val"]
_PLATEAU_TAIL_FRAC = 0.05
_PLATEAU_REL_THRESH = 0.02
_PLATEAU_ABS_FLOOR = 1e-4
_NEAR_BEST_REL = 0.05
_EPOCH_CACHE: Dict[str, set] = {}


def loss_history_path(output_dir: str) -> Path:
    return Path(output_dir) / "loss_history.csv"


def plots_dir(output_dir: str) -> Path:
    return Path(output_dir) / "plots"


def checkpoint_selection_path(output_dir: str) -> Path:
    return Path(output_dir) / "checkpoint_selection.json"


def convergence_json_path(output_dir: str) -> Path:
    return Path(output_dir) / "convergence.json"


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


def load_history(output_dir: str) -> List[Dict[str, Any]]:
    path = loss_history_path(output_dir)
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            epoch = _to_float(raw.get("epoch"))
            if epoch is None:
                continue
            rows.append(
                {
                    "epoch": int(epoch),
                    "train_loss": _to_float(raw.get("train_loss")),
                    "val_loss": _to_float(raw.get("val_loss")),
                    "lr": _to_float(raw.get("lr")),
                    "is_best_val": int(float(raw.get("is_best_val") or 0)),
                }
            )
    rows.sort(key=lambda r: r["epoch"])
    return rows


def existing_epochs(output_dir: str) -> set:
    key = str(loss_history_path(output_dir).resolve())
    if key not in _EPOCH_CACHE:
        _EPOCH_CACHE[key] = {int(r["epoch"]) for r in load_history(output_dir)}
    return _EPOCH_CACHE[key]


def append_loss_row(
    output_dir: str,
    *,
    epoch: int,
    train_loss: Optional[float],
    val_loss: Optional[float],
    lr: Optional[float],
    is_best_val: bool,
) -> None:
    """Append satu baris epoch. No-op jika epoch sudah ada (resume-aman)."""
    path = loss_history_path(output_dir)
    if int(epoch) in existing_epochs(output_dir):
        return
    new_file = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": int(epoch),
                "train_loss": "" if train_loss is None else f"{float(train_loss):.8g}",
                "val_loss": "" if val_loss is None else f"{float(val_loss):.8g}",
                "lr": "" if lr is None else f"{float(lr):.8g}",
                "is_best_val": 1 if is_best_val else 0,
            }
        )
    existing_epochs(output_dir).add(int(epoch))


def restore_best_val_tracker(
    output_dir: str,
) -> Tuple[Optional[float], Optional[int]]:
    """Pulihkan (best_val_loss, best_val_epoch) dari JSON atau history CSV."""
    sel = checkpoint_selection_path(output_dir)
    if sel.is_file():
        try:
            with sel.open() as f:
                data = json.load(f)
            loss = _to_float(data.get("best_val_loss"))
            epoch = data.get("best_val_epoch")
            if loss is not None and epoch is not None:
                return loss, int(epoch)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    history = load_history(output_dir)
    best_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    for row in history:
        v = row.get("val_loss")
        if v is None:
            continue
        if best_loss is None or v < best_loss:
            best_loss = float(v)
            best_epoch = int(row["epoch"])
    return best_loss, best_epoch


def _moving_average(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    window = max(1, int(window))
    out: List[Optional[float]] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = [v for v in values[lo : i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def compute_convergence(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {
            "n_epochs": 0,
            "plateaued": False,
            "note": "loss_history kosong",
        }

    epochs = [int(r["epoch"]) for r in history]
    train = [r.get("train_loss") for r in history]
    val = [r.get("val_loss") for r in history]
    val_finite = [(e, v) for e, v in zip(epochs, val) if v is not None]
    train_finite = [(e, v) for e, v in zip(epochs, train) if v is not None]

    n = len(history)
    window = max(5, min(50, max(1, n // 20)))
    val_smooth = _moving_average(val, window)

    best_val = None
    best_epoch = None
    for e, v in val_finite:
        if best_val is None or v < best_val:
            best_val = float(v)
            best_epoch = int(e)

    epoch_to_near_best = None
    if best_val is not None and val_finite:
        span = max(v for _, v in val_finite) - min(v for _, v in val_finite)
        thresh = best_val + max(_PLATEAU_ABS_FLOOR, _NEAR_BEST_REL * max(span, abs(best_val)))
        for e, v in val_finite:
            if v <= thresh:
                epoch_to_near_best = int(e)
                break

    tail_n = max(2, int(round(_PLATEAU_TAIL_FRAC * n)))
    tail_vals = [v for v in val_smooth[-tail_n:] if v is not None]
    plateaued = False
    mean_abs_delta = None
    plateau_threshold = None
    if len(tail_vals) >= 2:
        deltas = [abs(tail_vals[i] - tail_vals[i - 1]) for i in range(1, len(tail_vals))]
        mean_abs_delta = sum(deltas) / len(deltas)
        if val_finite:
            span = max(v for _, v in val_finite) - min(v for _, v in val_finite)
            plateau_threshold = max(_PLATEAU_ABS_FLOOR, _PLATEAU_REL_THRESH * span)
            plateaued = mean_abs_delta < plateau_threshold

    final_train = train_finite[-1][1] if train_finite else None
    final_val = val_finite[-1][1] if val_finite else None
    first_val = val_finite[0][1] if val_finite else None
    improved = None
    if first_val is not None and final_val is not None and abs(first_val) > 1e-12:
        improved = (first_val - final_val) / abs(first_val)

    gap_final = None
    if final_train is not None and final_val is not None:
        gap_final = float(final_val) - float(final_train)
    gap_at_best = None
    if best_epoch is not None:
        for r in history:
            if int(r["epoch"]) == int(best_epoch):
                if r.get("train_loss") is not None and r.get("val_loss") is not None:
                    gap_at_best = float(r["val_loss"]) - float(r["train_loss"])
                break

    return {
        "n_epochs": n,
        "epoch_first": epochs[0],
        "epoch_last": epochs[-1],
        "smooth_window": window,
        "best_val_loss": best_val,
        "best_val_epoch": best_epoch,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "train_val_gap_final": gap_final,
        "train_val_gap_at_best": gap_at_best,
        "relative_val_improvement": improved,
        "epoch_to_near_best": epoch_to_near_best,
        "near_best_relative_band": _NEAR_BEST_REL,
        "plateaued": bool(plateaued),
        "plateau_tail_epochs": tail_n,
        "plateau_mean_abs_delta": mean_abs_delta,
        "plateau_threshold": plateau_threshold,
        "val_smooth_last": val_smooth[-1] if val_smooth else None,
    }


def write_checkpoint_selection(
    output_dir: str,
    *,
    criterion: str = "min_val_loss",
    selected: str = "best_val",
    best_val_epoch: Optional[int],
    best_val_loss: Optional[float],
    final_epoch: Optional[int],
    final_train_loss: Optional[float],
    final_val_loss: Optional[float],
    best_val_path: Optional[str],
    latest_path: Optional[str],
    used_for_inference: str = "best_val.ckpt",
) -> Dict[str, Any]:
    payload = {
        "criterion": criterion,
        "selected": selected,
        "used_for_inference": used_for_inference,
        "best_val_epoch": best_val_epoch,
        "best_val_loss": best_val_loss,
        "final_epoch": final_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_checkpoint": best_val_path,
        "latest_checkpoint": latest_path,
        "note": (
            "Eval/infer memakai best_val.ckpt (val_loss minimum). "
            "latest.ckpt hanya untuk resume training."
        ),
    }
    path = checkpoint_selection_path(output_dir)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


def write_convergence_json(output_dir: str, summary: Dict[str, Any]) -> None:
    path = convergence_json_path(output_dir)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)


def _save_fig(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_training_artifacts(
    output_dir: str,
    *,
    best_val_epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    used_for_inference: str = "best_val.ckpt",
) -> Optional[Dict[str, Any]]:
    """Gambar tiga plot dosen. Kembalikan ringkasan konvergensi, atau None jika gagal."""
    history = load_history(output_dir)
    if not history:
        return None

    summary = compute_convergence(history)
    if best_val_epoch is None:
        best_val_epoch = summary.get("best_val_epoch")
    if best_val_loss is None:
        best_val_loss = summary.get("best_val_loss")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        summary["plot_error"] = f"matplotlib tidak tersedia: {exc}"
        write_convergence_json(output_dir, summary)
        return summary

    epochs = [int(r["epoch"]) for r in history]
    train = [r.get("train_loss") for r in history]
    val = [r.get("val_loss") for r in history]
    window = int(summary.get("smooth_window") or 5)
    val_smooth = _moving_average(val, window)
    out = plots_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="train_loss", color="C0", linewidth=1.4)
    if any(v is not None for v in val):
        ax.plot(epochs, val, label="val_loss", color="C1", linewidth=1.4)
    if best_val_epoch is not None:
        ax.axvline(
            best_val_epoch,
            color="C2",
            linestyle="--",
            linewidth=1.1,
            label=f"best val (epoch {best_val_epoch})",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out / "train_val_loss")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(epochs, val, color="C1", alpha=0.35, label="val_loss")
    axes[0].plot(
        epochs,
        val_smooth,
        color="C1",
        linewidth=1.6,
        label=f"val MA(window={window})",
    )
    if best_val_epoch is not None:
        axes[0].axvline(best_val_epoch, color="C2", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("val_loss")
    axes[0].set_title("Convergence analysis")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    deltas: List[Optional[float]] = [None]
    for i in range(1, len(val_smooth)):
        a, b = val_smooth[i - 1], val_smooth[i]
        deltas.append(None if a is None or b is None else abs(b - a))
    axes[1].plot(epochs, deltas, color="C3", linewidth=1.2, label="|Δ| smoothed val")
    thresh = summary.get("plateau_threshold")
    if thresh is not None:
        axes[1].axhline(
            thresh,
            color="gray",
            linestyle=":",
            label=f"plateau threshold ({thresh:.2g})",
        )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("|Δ| val_loss")
    plateau_txt = "plateaued" if summary.get("plateaued") else "not plateaued"
    axes[1].set_title(f"Late-training stability ({plateau_txt})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out / "convergence_analysis")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(v is not None for v in val):
        ax.plot(epochs, val, color="C1", linewidth=1.4, label="val_loss")
    final_epoch = epochs[-1]
    ax.axvline(
        final_epoch,
        color="C0",
        linestyle=":",
        linewidth=1.2,
        label=f"final epoch {final_epoch} (not used for eval)",
    )
    if best_val_epoch is not None:
        ax.axvline(
            best_val_epoch,
            color="C2",
            linestyle="--",
            linewidth=1.2,
            label=f"selected epoch {best_val_epoch}",
        )
        if best_val_loss is not None:
            ax.scatter(
                [best_val_epoch],
                [best_val_loss],
                color="C2",
                s=40,
                zorder=5,
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_loss")
    ax.set_title("Checkpoint selection (min val_loss)")
    note = (
        f"Criterion: min val_loss\n"
        f"Used for inference: {used_for_inference}\n"
        f"Best val: epoch={best_val_epoch}, loss={best_val_loss}\n"
        f"Final epoch: {final_epoch} (resume only)"
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out / "checkpoint_selection")
    plt.close(fig)
    plt.close("all")

    write_convergence_json(output_dir, summary)
    return summary


def write_run_artifacts(
    output_dir: str,
    *,
    best_val_epoch: Optional[int],
    best_val_loss: Optional[float],
    final_epoch: Optional[int],
    final_train_loss: Optional[float],
    final_val_loss: Optional[float],
    used_for_inference: str = "best_val.ckpt",
) -> Optional[Dict[str, Any]]:
    ckpt_dir = Path(output_dir) / "checkpoints"
    best_path = ckpt_dir / "best_val.ckpt"
    latest_path = ckpt_dir / "latest.ckpt"
    selected = "best_val" if best_path.is_file() else "latest"
    used = "best_val.ckpt" if best_path.is_file() else "latest.ckpt"
    write_checkpoint_selection(
        output_dir,
        criterion="min_val_loss",
        selected=selected,
        best_val_epoch=best_val_epoch,
        best_val_loss=best_val_loss,
        final_epoch=final_epoch,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        best_val_path=str(best_path) if best_path.is_file() else None,
        latest_path=str(latest_path) if latest_path.is_file() else None,
        used_for_inference=used,
    )
    try:
        return plot_training_artifacts(
            output_dir,
            best_val_epoch=best_val_epoch,
            best_val_loss=best_val_loss,
            used_for_inference=used_for_inference if best_path.is_file() else used,
        )
    except Exception as exc:
        summary = compute_convergence(load_history(output_dir))
        summary["plot_error"] = str(exc)
        write_convergence_json(output_dir, summary)
        return summary
