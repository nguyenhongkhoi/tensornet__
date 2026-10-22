#!/usr/bin/env python
"""Train Classical CNN / Hybrid QCNN / Full-quantum QCNN on MNIST and plot train/validation convergence.

Protocol (same for all three models)
  * MNIST official train set only: 50,000 train / 10,000 validation, split_seed=20260911
    (first 10,000 of randperm = validation). The test set is never touched.
  * ToTensor only (pixels in [0,1]), no augmentation, Adam, cross-entropy.
  * Per epoch: train loss/acc (running average while training) and validation loss/acc.

Full-quantum "best setting" (validation loss 0.0740 / acc 97.71%, single seed) is a 3-phase recipe:
  phase 1  pretrain  from scratch, Adam lr 2e-3, 20 epochs, batch 64, keep best val-accuracy checkpoint (Stage 3D)
  phase 2  refine    continue from phase-1 best, NEW Adam lr 2e-4, 5 epochs, keep best val-accuracy (Stage 3E)
  phase 3  reupload  add re-upload scale (+24 params), warm-start from phase-2 best, Adam with
                     core lr 1e-4 / readout lr 1e-3, ReduceLROnPlateau(0.5, patience 3), up to 60 epochs,
                     early-stop patience 10, keep best val-loss checkpoint (Stage 9 B_warm_lr0)
  architecture: 16x16 grayscale, 3 branches, 8 final qubits, strongly_entangling n_layers=4,
                merge_rounds=2, pauli_observable readout with pair_correlators=full9, init_scale=0.01 (13,134 params)

Examples
  python run_convergence.py                       # train all 3 models, then plot
  python run_convergence.py --models full_quantum # best full-quantum recipe only
  python run_convergence.py --plot-only           # re-plot from existing history_*.csv
  python run_convergence.py --smoke --fake-data --device cpu   # seconds-long wiring test
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import ClassicalCNN, FullQuantumQCNN, HybridQCNN  # noqa: E402
from models.reupload import add_reupload  # noqa: E402

MODEL_NAMES = ("classical", "hybrid", "full_quantum")
LABELS = {"classical": "Classical CNN", "hybrid": "Hybrid QCNN", "full_quantum": "Full quantum QCNN"}
COLORS = {"classical": "#0072B2", "hybrid": "#E69F00", "full_quantum": "#009E73"}  # Okabe-Ito (CVD-safe)

# Architecture of the best full-quantum model (matches Stage 3D/3E/9; 13,110 params, 13,134 with re-upload).
FQ_ARCH = dict(target_size=(16, 16), grayscale=True, final_qubits=8, readout_type="pauli_observable",
               pair_correlators="full9", init_scale=0.01, merge_rounds=2,
               entangling_type="strongly_entangling", n_layers=4, quantum_channels=3)
# Best hybrid from the variant search (brickwall, 8 final qubits, Pauli readout; 3,507 params).
HYBRID_ARCH = dict(feature_size=(8, 8), quantum_channels=1, final_qubits=8, readout_type="pauli_observable",
                   pair_correlators="zz", init_scale=0.01, merge_rounds=2, entangling_type="brickwall")


# --------------------------------------------------------------------------------------- utilities
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def replace_with_retry(src: Path, dst: Path, tries: int = 60, wait: float = 0.5) -> None:
    """os.replace that survives transient Windows file locks (Defender/OneDrive/indexer)."""
    for attempt in range(tries):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(wait)


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    replace_with_retry(tmp, path)


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    replace_with_retry(tmp, path)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def jsonable_args(args):
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


HISTORY_COLUMNS = ["epoch", "phase", "train_loss", "train_accuracy", "validation_loss",
                   "validation_accuracy", "lr_core", "lr_readout", "epoch_seconds"]


def write_history(path: Path, rows) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(tmp, path)


def read_history(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in HISTORY_COLUMNS:
            if key != "phase":
                row[key] = float(row[key]) if row.get(key) not in (None, "") else float("nan")
    return rows


# ------------------------------------------------------------------------------------------ data
def make_loaders(args, batch_size: int, seed: int):
    """Fixed 50k/10k split of the MNIST train set (identical for all models)."""
    from torchvision import datasets, transforms

    if args.fake_data:
        base = datasets.FakeData(size=2000, image_size=(1, 28, 28), num_classes=10,
                                 transform=transforms.ToTensor(), random_offset=0)
        n_val = 500
    else:
        base = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
        n_val = 10000
    perm = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if args.train_subset:
        train_idx = train_idx[: args.train_subset]
    if args.val_subset:
        val_idx = val_idx[: args.val_subset]
    pin = torch.cuda.is_available() and args.device.startswith("cuda")
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(Subset(base, train_idx), batch_size=batch_size, shuffle=True, generator=generator,
                       num_workers=0, pin_memory=pin)
    val = DataLoader(Subset(base, val_idx), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train, val, generator


# ----------------------------------------------------------------------------------- train/eval
def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(-1) == labels).sum())
            seen += len(labels)
    return loss_sum / seen, correct / seen


def run_phase(*, tag, phase, model, optimizer, scheduler, train, val, generator, epochs, select, patience,
              out_dir: Path, device, epoch_offset=0, resume=True):
    """Train one phase. Returns (rows, best_state_dict). Resumes from <tag>_last.pt after a crash.

    select: 'val_loss' (lower is better) or 'val_acc' (higher is better).
    patience: stop after this many epochs without improvement of the selection metric (None = off).
    """
    done_path, last_path, best_path = out_dir / f"{tag}_done.json", out_dir / f"{tag}_last.pt", out_dir / f"{tag}_best.pt"
    if resume and done_path.exists() and best_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        print(f"[{tag}] already complete ({len(done['rows'])} epochs) - skipping", flush=True)
        return done["rows"], torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]

    phase_started = time.perf_counter()
    sign = 1.0 if select == "val_loss" else -1.0
    best, stale, rows, start = float("inf"), 0, [], 1
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if scheduler is not None and ck.get("scheduler_state_dict"):
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        best, stale, rows, start = ck["best"], ck["stale"], ck["rows"], ck["epoch"] + 1
        generator.set_state(ck["shuffle_rng"])
        print(f"[{tag}] resumed after epoch {ck['epoch']}", flush=True)

    reason = "epoch_budget"
    for epoch in range(start, epochs + 1):
        began = time.perf_counter()
        lrs = [g["lr"] for g in optimizer.param_groups]
        train_loss, train_acc = run_epoch(model, train, device, optimizer)
        val_loss, val_acc = run_epoch(model, val, device)
        row = dict(epoch=epoch_offset + epoch, phase=phase, train_loss=train_loss, train_accuracy=train_acc,
                   validation_loss=val_loss, validation_accuracy=val_acc, lr_core=lrs[0], lr_readout=lrs[-1],
                   epoch_seconds=time.perf_counter() - began)
        rows.append(row)
        metric = sign * (val_loss if select == "val_loss" else val_acc)
        improved = metric < best - 1e-7
        stale = 0 if improved else stale + 1
        best = metric if improved else best
        if scheduler is not None:
            scheduler.step(val_loss)
        state = {"model_state_dict": model.state_dict(), "epoch": epoch, "row": row}
        if improved:
            save_atomic(state, best_path)
        save_atomic({**state, "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                     "best": best, "stale": stale, "rows": rows, "shuffle_rng": generator.get_state()}, last_path)
        print(f"[{tag}] epoch {epoch:3d}/{epochs} train {train_loss:.4f}/{train_acc:.4f} "
              f"val {val_loss:.4f}/{val_acc:.4f} lr {lrs[0]:.2e} {row['epoch_seconds']:.0f}s"
              f"{'  *best' if improved else ''}", flush=True)
        if patience and stale >= patience:
            reason = "early_stop_patience"
            break
    phase_seconds = time.perf_counter() - phase_started
    print(f"[{tag}] finished {len(rows)} epoch(s) in {format_duration(phase_seconds)} ({phase_seconds:.1f}s) "
          f"reason={reason}", flush=True)
    write_json(done_path, {"rows": rows, "stop_reason": reason, "best_metric": sign * best, "select": select,
                           "phase_seconds": phase_seconds, "phase_duration": format_duration(phase_seconds)})
    return rows, torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]


# ----------------------------------------------------------------------------------- model runs
def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_simple(name, args, out_dir: Path, device):
    """Classical / hybrid: one phase, Adam, fixed epochs, keep best validation loss checkpoint."""
    cfg = getattr(args, name)
    seed_all(args.seed)
    train, val, generator = make_loaders(args, cfg["batch_size"], args.seed)
    if name == "classical":
        model = ClassicalCNN(1, 10)
    else:
        model = HybridQCNN(1, 10, **HYBRID_ARCH)
    model.to(device)
    print(f"[{name}] parameters: {count_params(model)}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rows, _ = run_phase(tag=name, phase="train", model=model, optimizer=optimizer, scheduler=None, train=train,
                        val=val, generator=generator, epochs=cfg["epochs"], select="val_loss", patience=None,
                        out_dir=out_dir, device=device, resume=args.resume)
    return rows


def build_fq(args, device, reupload: bool):
    seed_all(args.seed)
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH)
    if reupload:
        add_reupload(model, input_sites=FQ_ARCH["target_size"][0] * FQ_ARCH["target_size"][1],
                     final_qubits=FQ_ARCH["final_qubits"], init_scale=FQ_ARCH["init_scale"],
                     merge_rounds=FQ_ARCH["merge_rounds"], entangling_type=FQ_ARCH["entangling_type"],
                     n_layers=FQ_ARCH["n_layers"])
    return model.to(device)


def load_state(model, state, reupload: bool):
    """Exact warm start: every tensor is loaded; only the new reupload_scale keys (zeros) may be missing."""
    current = model.state_dict()
    expected_missing = {k for k in current if k.endswith(".reupload_scale")} if reupload else set()
    missing, unexpected = set(current) - set(state), set(state) - set(current)
    if missing != expected_missing or unexpected:
        raise RuntimeError(f"checkpoint/model key mismatch: missing={sorted(missing - expected_missing)[:5]} "
                           f"unexpected={sorted(unexpected)[:5]}")
    model.load_state_dict(state, strict=False)


def train_full_quantum(args, out_dir: Path, device):
    fq = args.full_quantum
    all_rows, offset = [], 0
    best_state = None

    if args.fq_warm_start:  # skip phases 1-2, start from an existing Stage-3E style checkpoint
        payload = torch.load(args.fq_warm_start, map_location="cpu", weights_only=False)
        best_state = payload["model_state_dict"]
        print(f"[full_quantum] warm start from {args.fq_warm_start} (phases 1-2 skipped)", flush=True)
    else:
        for phase, label, epochs, lr in (("pretrain", "fq_phase1_pretrain", fq["pretrain_epochs"], fq["pretrain_lr"]),
                                         ("refine", "fq_phase2_refine", fq["refine_epochs"], fq["refine_lr"])):
            if epochs <= 0:
                continue
            train, val, generator = make_loaders(args, fq["batch_size"], args.seed)
            model = build_fq(args, device, reupload=False)
            if best_state is not None:
                load_state(model, best_state, reupload=False)
            print(f"[{label}] parameters: {count_params(model)}  lr={lr}", flush=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # fresh optimizer each phase (as Stage 3D/3E)
            rows, best_state = run_phase(tag=label, phase=phase, model=model, optimizer=optimizer, scheduler=None,
                                         train=train, val=val, generator=generator, epochs=epochs,
                                         select="val_acc", patience=None, out_dir=out_dir, device=device,
                                         epoch_offset=offset, resume=args.resume)
            all_rows += rows
            offset += len(rows)
            write_history(out_dir / "history_full_quantum.csv", all_rows)

    if fq["epochs"] > 0:
        train, val, generator = make_loaders(args, fq["batch_size"], args.seed)
        model = build_fq(args, device, reupload=fq["reupload"])
        if best_state is not None:
            load_state(model, best_state, reupload=fq["reupload"])
        print(f"[fq_phase3_reupload] parameters: {count_params(model)}", flush=True)
        core, readout = [], []
        for pname, param in model.named_parameters():
            (readout if pname.startswith("readout.") or ".readout." in pname else core).append(param)
        optimizer = torch.optim.Adam([dict(params=core, lr=fq["core_lr"]), dict(params=readout, lr=fq["readout_lr"])])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-7)
        rows, _ = run_phase(tag="fq_phase3_reupload", phase="reupload", model=model, optimizer=optimizer,
                            scheduler=scheduler, train=train, val=val, generator=generator, epochs=fq["epochs"],
                            select="val_loss", patience=fq["patience"], out_dir=out_dir, device=device,
                            epoch_offset=offset, resume=args.resume)
        all_rows += rows
        write_history(out_dir / "history_full_quantum.csv", all_rows)
    return all_rows


# ---------------------------------------------------------------------------------------- plots
def plot_all(out_dir: Path, acc_ymin: float = 90.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = {n: read_history(out_dir / f"history_{n}.csv") for n in MODEL_NAMES
                 if (out_dir / f"history_{n}.csv").exists()}
    if not histories:
        raise SystemExit(f"No history_*.csv found in {out_dir}")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.6, "font.size": 10})

    def series(rows, key):
        return [r["epoch"] for r in rows], [r[key] for r in rows]

    # Figure 1: all models on shared axes (loss on log scale; accuracy zoomed to the interesting range).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, rows in histories.items():
        for key, style, kind in (("train", "-", "train"), ("validation", "--", "validation")):
            x, y = series(rows, f"{key}_loss")
            axes[0].plot(x, y, style, color=COLORS[name], lw=1.6, label=f"{LABELS[name]} - {kind}")
            x, y = series(rows, f"{key}_accuracy")
            axes[1].plot(x, [100 * v for v in y], style, color=COLORS[name], lw=1.6, label=f"{LABELS[name]} - {kind}")
    if "full_quantum" in histories:  # mark phase boundaries of the full-quantum recipe
        rows, previous = histories["full_quantum"], None
        for row in rows:
            if previous is not None and row["phase"] != previous:
                for ax in axes:
                    ax.axvline(row["epoch"] - 0.5, color=COLORS["full_quantum"], lw=0.8, ls=":", alpha=0.7)
            previous = row["phase"]
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch (full quantum: cumulative over pretrain -> refine -> reupload)")
    axes[0].set_ylabel("cross-entropy loss (log scale)")
    axes[0].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_ylim(acc_ymin, 100.2)
    axes[1].set_title(f"Accuracy (y-axis zoomed to {acc_ymin:g}-100%; lower with --acc-ymin)")
    axes[1].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("MNIST convergence: solid = train, dashed = validation (dotted vertical lines = full-quantum phase changes)")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_all.png", dpi=160)
    plt.close(fig)

    # Figure 2: one panel per model (own x-range and loss range), validation minimum annotated.
    fig, axes = plt.subplots(1, len(histories), figsize=(5.2 * len(histories), 4.4), squeeze=False)
    for ax, (name, rows) in zip(axes[0], histories.items()):
        for key, style in (("train", "-"), ("validation", "--")):
            x, y = series(rows, f"{key}_loss")
            ax.plot(x, y, style, color=COLORS[name], lw=1.8, label=key)
        best = min(rows, key=lambda r: r["validation_loss"])
        ax.scatter([best["epoch"]], [best["validation_loss"]], color=COLORS[name], zorder=3, s=28)
        ax.annotate(f"min val {best['validation_loss']:.4f}\n(epoch {int(best['epoch'])}, acc {100 * best['validation_accuracy']:.2f}%)",
                    (best["epoch"], best["validation_loss"]), textcoords="offset points",
                    xytext=(8, 18) if best["epoch"] <= rows[len(rows) // 2]["epoch"] else (-8, 18),
                    ha="left" if best["epoch"] <= rows[len(rows) // 2]["epoch"] else "right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(LABELS[name])
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss (log scale)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_per_model.png", dpi=160)
    plt.close(fig)

    summary = {}
    for name, rows in histories.items():
        best = min(rows, key=lambda r: r["validation_loss"])
        summary[name] = dict(epochs=len(rows), best_val_loss=best["validation_loss"], best_val_epoch=int(best["epoch"]),
                             val_accuracy_at_best_loss=best["validation_accuracy"],
                             final_train_loss=rows[-1]["train_loss"], final_val_loss=rows[-1]["validation_loss"])
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_dir / 'convergence_all.png'} and {out_dir / 'convergence_per_model.png'}")


# ------------------------------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--out-dir", default="runs/convergence")
    p.add_argument("--data-root", default="mnist_data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=20260911)
    p.add_argument("--acc-ymin", type=float, default=90.0, help="lower y-limit (%%) of the accuracy panel")
    p.add_argument("--plot-only", action="store_true", help="skip training; plot existing history_*.csv in --out-dir")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="ignore existing checkpoints in --out-dir")
    # classical / hybrid
    for name, epochs, lr in (("classical", 20, 1e-3), ("hybrid", 20, 1e-3)):
        p.add_argument(f"--{name}-epochs", type=int, default=epochs)
        p.add_argument(f"--{name}-batch-size", type=int, default=128)
        p.add_argument(f"--{name}-lr", type=float, default=lr)
    # full quantum: defaults ARE the best setting
    p.add_argument("--fq-batch-size", type=int, default=64)
    p.add_argument("--fq-pretrain-epochs", type=int, default=20)
    p.add_argument("--fq-pretrain-lr", type=float, default=2e-3)
    p.add_argument("--fq-refine-epochs", type=int, default=5)
    p.add_argument("--fq-refine-lr", type=float, default=2e-4)
    p.add_argument("--fq-epochs", type=int, default=60, help="phase 3 (reupload) max epochs")
    p.add_argument("--fq-core-lr", type=float, default=1e-4)
    p.add_argument("--fq-readout-lr", type=float, default=1e-3)
    p.add_argument("--fq-patience", type=int, default=10)
    p.add_argument("--fq-reupload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fq-warm-start", default=None,
                   help="Stage-3E style checkpoint (dict with 'model_state_dict'); skips phases 1-2")
    # testing helpers
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument("--fake-data", action="store_true", help="torchvision FakeData instead of MNIST (wiring test only)")
    p.add_argument("--smoke", action="store_true", help="tiny subsets, 2 epochs per phase, batch<=16")
    args = p.parse_args()

    args.classical = dict(epochs=args.classical_epochs, batch_size=args.classical_batch_size, lr=args.classical_lr)
    args.hybrid = dict(epochs=args.hybrid_epochs, batch_size=args.hybrid_batch_size, lr=args.hybrid_lr)
    args.full_quantum = dict(batch_size=args.fq_batch_size, pretrain_epochs=args.fq_pretrain_epochs,
                             pretrain_lr=args.fq_pretrain_lr, refine_epochs=args.fq_refine_epochs,
                             refine_lr=args.fq_refine_lr, epochs=args.fq_epochs, core_lr=args.fq_core_lr,
                             readout_lr=args.fq_readout_lr, patience=args.fq_patience, reupload=args.fq_reupload)
    if args.smoke:
        args.train_subset, args.val_subset = args.train_subset or 64, args.val_subset or 32
        for cfg in (args.classical, args.hybrid):
            cfg.update(epochs=2, batch_size=16)
        args.full_quantum.update(batch_size=8, pretrain_epochs=2, refine_epochs=1, epochs=2, patience=10)
    return args


def build_run_config(args, out_dir: Path):
    selected = list(args.models)
    phase_plan = {}
    if "classical" in selected:
        phase_plan["classical"] = [dict(tag="classical", phase="train", optimizer="Adam",
                                        lr=args.classical["lr"], epochs=args.classical["epochs"],
                                        batch_size=args.classical["batch_size"], checkpoint_select="val_loss")]
    if "hybrid" in selected:
        phase_plan["hybrid"] = [dict(tag="hybrid", phase="train", optimizer="Adam",
                                     lr=args.hybrid["lr"], epochs=args.hybrid["epochs"],
                                     batch_size=args.hybrid["batch_size"], checkpoint_select="val_loss")]
    if "full_quantum" in selected:
        fq = args.full_quantum
        phase_plan["full_quantum"] = [
            dict(tag="fq_phase1_pretrain", phase="pretrain", optimizer="Adam", lr=fq["pretrain_lr"],
                 epochs=fq["pretrain_epochs"], batch_size=fq["batch_size"], checkpoint_select="val_acc"),
            dict(tag="fq_phase2_refine", phase="refine", optimizer="Adam", lr=fq["refine_lr"],
                 epochs=fq["refine_epochs"], batch_size=fq["batch_size"], checkpoint_select="val_acc"),
            dict(tag="fq_phase3_reupload", phase="reupload", optimizer="Adam + ReduceLROnPlateau",
                 core_lr=fq["core_lr"], readout_lr=fq["readout_lr"], scheduler_factor=0.5,
                 scheduler_patience=3, scheduler_min_lr=1e-7, epochs=fq["epochs"],
                 batch_size=fq["batch_size"], early_stop_patience=fq["patience"],
                 reupload=fq["reupload"], checkpoint_select="val_loss"),
        ]
        if args.fq_warm_start:
            phase_plan["full_quantum"][0]["skipped_when_warm_start"] = True
            phase_plan["full_quantum"][1]["skipped_when_warm_start"] = True

    return dict(
        created_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        command=[Path(sys.argv[0]).name, *sys.argv[1:]],
        working_directory=str(Path.cwd()),
        output_directory=str(out_dir),
        effective_args=jsonable_args(args),
        data=dict(
            dataset="FakeData" if args.fake_data else "MNIST train split",
            data_root=args.data_root,
            split_seed=args.split_seed,
            train_size="all remaining after validation" if args.train_subset is None else args.train_subset,
            validation_size=500 if args.fake_data and args.val_subset is None else (
                10000 if args.val_subset is None else args.val_subset),
            test_set_used=False,
        ),
        architectures=dict(classical="ClassicalCNN(1, 10)", hybrid=HYBRID_ARCH, full_quantum=FQ_ARCH),
        training_plan=phase_plan,
        resume=dict(enabled=args.resume, disable_with="--no-resume"),
        plotting=dict(acc_ymin=args.acc_ymin, plot_only=args.plot_only,
                      outputs=["convergence_all.png", "convergence_per_model.png", "summary.json"]),
        expected_outputs=dict(
            histories=[f"history_{name}.csv" for name in selected],
            checkpoints=["<tag>_best.pt", "<tag>_last.pt", "<tag>_done.json"],
            config="config.json",
        ),
        environment=dict(
            python=sys.version.replace("\n", " "),
            platform=platform.platform(),
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            requested_device=args.device,
            cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        ),
    )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    run_config = build_run_config(args, out_dir)
    write_json(out_dir / "config.json", run_config)
    print(f"[run] output directory: {out_dir}", flush=True)
    print(f"[run] detailed config saved to {out_dir / 'config.json'}", flush=True)
    if not args.plot_only:
        device = torch.device(args.device)
        for name in args.models:
            model_started = time.perf_counter()
            print(f"\n===== {name} =====", flush=True)
            rows = train_full_quantum(args, out_dir, device) if name == "full_quantum" else train_simple(name, args, out_dir, device)
            write_history(out_dir / f"history_{name}.csv", rows)
            model_seconds = time.perf_counter() - model_started
            print(f"[{name}] total time {format_duration(model_seconds)} ({model_seconds:.1f}s)", flush=True)
    plot_all(out_dir, args.acc_ymin)
    run_seconds = time.perf_counter() - run_started
    write_json(out_dir / "run_time.json", dict(total_seconds=run_seconds,
                                               total_duration=format_duration(run_seconds),
                                               completed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds")))
    print(f"[run] total time {format_duration(run_seconds)} ({run_seconds:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
