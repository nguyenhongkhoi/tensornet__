"""Train the 16x16 full-quantum TN model with the original 3-phase recipe, augmentation and MLP readout."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from models import FullQuantumQCNN  # noqa: E402
from models.reupload import add_reupload  # noqa: E402


FQ_ARCH = dict(
    target_size=(16, 16), grayscale=True, final_qubits=8,
    readout_type="pauli_mlp", pair_correlators="full9", init_scale=0.01,
    merge_rounds=2, entangling_type="strongly_entangling", n_layers=4,
    quantum_channels=3,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace, device: torch.device):
    # Keep the same deterministic 50k/10k split as the original 16x16 run.
    split_order = torch.randperm(
        60_000, generator=torch.Generator().manual_seed(args.split_seed)
    ).tolist()
    validation_indices = split_order[:10_000]
    train_indices = split_order[10_000:]

    # Separate dataset objects are essential: augmentation must never touch validation.
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=args.aug_degrees,
                                translate=(args.aug_translate, args.aug_translate),
                                fill=0),
        transforms.ToTensor(),
    ])
    validation_transform = transforms.ToTensor()
    train_base = datasets.MNIST(root=args.data_root, train=True, download=True,
                                transform=train_transform)
    validation_base = datasets.MNIST(root=args.data_root, train=True, download=True,
                                     transform=validation_transform)
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_base, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=pin,
    )
    validation_loader = DataLoader(
        Subset(validation_base, validation_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=pin,
    )
    return train_loader, validation_loader, generator, train_indices, validation_indices


def build_model(device: torch.device, reupload: bool = False) -> FullQuantumQCNN:
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH)
    if reupload:
        add_reupload(model, input_sites=16 * 16, final_qubits=8, init_scale=0.01,
                     merge_rounds=2, entangling_type="strongly_entangling", n_layers=4)
    return model.to(device)


def load_phase_state(model: FullQuantumQCNN, source: dict, allow_new_reupload: bool = False) -> None:
    target = model.state_dict()
    missing, unexpected = set(target) - set(source), set(source) - set(target)
    allowed_missing = {key for key in missing if allow_new_reupload and key.endswith(".reupload_scale")}
    mismatched = [key for key in target.keys() & source.keys() if target[key].shape != source[key].shape]
    if missing != allowed_missing or unexpected or mismatched:
        raise RuntimeError(f"Phase checkpoint mismatch: missing={sorted(missing)}, "
                           f"unexpected={sorted(unexpected)}, shape_mismatch={sorted(mismatched)}")
    model.load_state_dict(source, strict=False)


def run_epoch(model, loader, device, optimizer=None, progress_label: str | None = None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, seen = 0.0, 0, 0
    started = time.perf_counter()
    total_batches = len(loader)
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader, start=1):
            if progress_label:
                print(
                    f"\r[{progress_label}] batch {batch_index:>3}/{total_batches} running...",
                    end="",
                    flush=True,
                )
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss khong huu han")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += labels.numel()
            if progress_label:
                elapsed = time.perf_counter() - started
                print(
                    f"\r[{progress_label}] batch {batch_index:>3}/{total_batches} "
                    f"loss={loss_sum / seen:.4f} acc={correct / seen:.2%} "
                    f"elapsed={elapsed:.0f}s",
                    end="",
                    flush=True,
                )
    if progress_label:
        print(
            f"\r[{progress_label}] batches {total_batches}/{total_batches} "
            f"loss={loss_sum / seen:.4f} acc={correct / seen:.2%} "
            f"elapsed={time.perf_counter() - started:.0f}s",
            flush=True,
        )
    return loss_sum / seen, correct / seen


def write_history(rows: list[dict], out_dir: Path) -> None:
    fields = ["epoch", "train_loss", "train_accuracy", "validation_loss",
              "validation_accuracy", "lr_core", "lr_readout", "epoch_seconds"]
    with (out_dir / "history_full_quantum_mlp_aug.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(rows: list[dict], out_dir: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in rows], label="Validation")
    axes[0].set(title="Cross entropy", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in rows], label="Train")
    axes[1].plot(epochs, [100 * row["validation_accuracy"] for row in rows], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(90, 100))
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.suptitle("16x16 FullQuantum TN + MLP readout + train augmentation")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_mlp_aug.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    for attempt in range(60):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.5)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    for attempt in range(60):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.5)


def write_phase_history(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalized_config(config: dict) -> dict:
    normalized = json.loads(json.dumps(config, sort_keys=True, default=str))
    normalized.pop("output_directory", None)
    args = normalized.get("args", {})
    args.pop("out_dir", None)
    args.pop("no_resume", None)
    return normalized


def validate_run_directory(out_dir: Path, config: dict, *, resume: bool) -> None:
    config_path = out_dir / "config_mlp_aug.json"
    artifact_patterns = (
        "fq_mlp_aug_*_best.pt", "fq_mlp_aug_*_last.pt", "fq_mlp_aug_*_done.json",
        "*_history.csv", "history_full_quantum_mlp_aug.csv", "summary_mlp_aug.json",
        "full_quantum_mlp_aug_best.pt",
    )
    artifacts = [path for pattern in artifact_patterns for path in out_dir.glob(pattern)]
    if artifacts and not config_path.exists():
        raise RuntimeError(
            f"Run artifacts exist without {config_path.name}; choose a new --out-dir."
        )
    if artifacts and not resume:
        raise RuntimeError(
            f"Run artifacts already exist in {out_dir}; choose a new --out-dir."
        )
    if artifacts and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if _normalized_config(previous) != _normalized_config(config):
            raise RuntimeError(
                "Saved run configuration differs from the requested settings. "
                "Keep the original arguments to resume, or choose a new --out-dir."
            )


def train_phase(*, tag, phase, model, optimizer, scheduler, train_loader, validation_loader,
                generator, epochs, select, patience, offset, out_dir, device, rows, resume=True):
    done_path = out_dir / f"{tag}_done.json"
    last_path = out_dir / f"{tag}_last.pt"
    best_path = out_dir / f"{tag}_best.pt"
    if resume and done_path.exists() and best_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        phase_rows = done["rows"]
        rows.extend(phase_rows)
        write_history(rows, out_dir)
        write_phase_history(phase_rows, out_dir / f"{tag}_history.csv")
        print(f"[{tag}] already complete ({len(phase_rows)} epochs) - skipping", flush=True)
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]
        return phase_rows, best_state

    sign = 1.0 if select == "val_loss" else -1.0
    best, stale, phase_rows, start_epoch = float("inf"), 0, [], 1
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best = checkpoint["best"]
        stale = checkpoint["stale"]
        phase_rows = checkpoint["rows"]
        start_epoch = checkpoint["epoch"] + 1
        generator.set_state(checkpoint["shuffle_rng"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        rows.extend(phase_rows)
        write_history(rows, out_dir)
        write_phase_history(phase_rows, out_dir / f"{tag}_history.csv")
        print(f"[{tag}] resumed after epoch {checkpoint['epoch']}", flush=True)

    reason = "epoch_budget"
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, train_loader, device, optimizer,
            progress_label=f"{tag} epoch {epoch}/{epochs} train",
        )
        val_loss, val_acc = run_epoch(
            model, validation_loader, device,
            progress_label=f"{tag} epoch {epoch}/{epochs} validation",
        )
        lr_core, lr_readout = optimizer.param_groups[0]["lr"], optimizer.param_groups[-1]["lr"]
        row = {"epoch": offset + epoch, "phase": phase, "train_loss": train_loss,
               "train_accuracy": train_acc, "validation_loss": val_loss,
               "validation_accuracy": val_acc, "lr_core": lr_core,
               "lr_readout": lr_readout, "epoch_seconds": time.perf_counter() - started}
        rows.append(row)
        phase_rows.append(row)
        metric = val_acc if select == "val_accuracy" else val_loss
        signed_metric = sign * metric
        improved = signed_metric < best - 1e-7
        if improved:
            best, stale = signed_metric, 0
            save_atomic({"model_state_dict": model.state_dict(), "epoch": epoch,
                         "row": row, "phase": phase}, best_path)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        current_state = {"model_state_dict": model.state_dict(), "epoch": epoch,
                         "row": row}
        save_atomic({**current_state,
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                     "best": best, "stale": stale, "rows": phase_rows,
                     "shuffle_rng": generator.get_state(),
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_state_all": torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None}, last_path)
        write_history(rows, out_dir)
        write_phase_history(phase_rows, out_dir / f"{tag}_history.csv")
        print(f"[{tag}] epoch {epoch:3d}/{epochs} train {train_loss:.4f}/{train_acc:.4f} "
              f"val {val_loss:.4f}/{val_acc:.4f} lr {lr_core:.2e}/{lr_readout:.2e} "
              f"{row['epoch_seconds']:.0f}s{'  *best' if improved else ''}", flush=True)
        if patience is not None and stale >= patience:
            reason = "early_stop_patience"
            print(f"[{tag}] early stop after {patience} epochs without improvement.", flush=True)
            break
    write_json(done_path, {"rows": phase_rows, "stop_reason": reason,
                           "best_metric": sign * best, "select": select})
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]
    return phase_rows, best_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["full_quantum"], default=["full_quantum"],
                        help="Accepted for compatibility with run_convergence.py; this script trains full_quantum only.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "mnist_data"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "mlp_aug_3phase"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--phase1-epochs", "--fq-pretrain-epochs", dest="phase1_epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", "--fq-refine-epochs", dest="phase2_epochs", type=int, default=5)
    parser.add_argument("--phase3-epochs", "--fq-epochs", dest="phase3_epochs", type=int, default=60)
    parser.add_argument("--batch-size", "--fq-batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--phase1-lr", "--fq-pretrain-lr", dest="phase1_lr", type=float, default=2e-3)
    parser.add_argument("--phase2-lr", "--fq-refine-lr", dest="phase2_lr", type=float, default=2e-4)
    parser.add_argument("--core-lr", "--fq-core-lr", dest="core_lr", type=float, default=1e-4)
    parser.add_argument("--readout-lr", "--fq-readout-lr", dest="readout_lr", type=float, default=1e-3)
    parser.add_argument("--patience", "--fq-patience", dest="patience", type=int, default=10)
    parser.add_argument("--fq-reupload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aug-degrees", type=float, default=8.0)
    parser.add_argument("--aug-translate", type=float, default=0.05)
    parser.add_argument("--no-resume", action="store_true",
                        help="Refuse existing run artifacts; use a new output directory for a fresh run.")
    args = parser.parse_args()
    if min(args.phase1_epochs, args.phase2_epochs, args.phase3_epochs,
           args.batch_size, args.patience) < 1:
        parser.error("Epoch counts, batch size, and patience must be positive")
    if min(args.phase1_lr, args.phase2_lr, args.core_lr, args.readout_lr) <= 0:
        parser.error("Learning rates must be positive")
    if not 0 <= args.aug_translate < 0.5 or args.aug_degrees < 0:
        parser.error("Invalid augmentation settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA selected but PyTorch cannot access CUDA")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    device = torch.device(args.device)
    train_loader, validation_loader, generator, train_indices, validation_indices = make_loaders(args, device)
    # Match run_convergence: reset the global seed immediately before creating
    # each phase's model, while each phase gets a freshly seeded loader generator.
    seed_all(args.seed)
    model = build_model(device)
    total_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print("Training from scratch with the original 3-phase schedule; no old checkpoint is loaded.", flush=True)
    print(f"Training on {len(train_indices)} examples with RandomAffine; validation uses "
          f"{len(validation_indices)} unaugmented examples.", flush=True)
    print(f"Parameters: {total_params:,} | device: {device} | outputs: {out_dir.resolve()}", flush=True)

    config = {
        "model": "16x16 FullQuantumQCNN, original three-phase schedule, MLP readout",
        "initialization": "phase 1 from scratch; phases 2 and 3 load the previous phase best checkpoint",
        "architecture": FQ_ARCH,
        "readout_mlp": ["Linear(276,256)", "LayerNorm(256)", "GELU()", "Dropout(0.1)", "Linear(256,10)"],
        "data": {"dataset": "MNIST official train split", "train_size": 50000,
                 "validation_size": 10000, "split_seed": args.split_seed,
                 "validation_transform": "ToTensor only", "official_test_used": False},
        "train_augmentation": {"RandomAffine_degrees": args.aug_degrees,
                               "RandomAffine_translate_fraction": args.aug_translate, "fill": 0},
        "optimization": {
            "phase1": {"optimizer": "Adam", "lr": args.phase1_lr, "epochs": args.phase1_epochs,
                       "select": "validation_accuracy"},
            "phase2": {"optimizer": "fresh Adam", "lr": args.phase2_lr, "epochs": args.phase2_epochs,
                       "select": "validation_accuracy"},
            "phase3": {"optimizer": "Adam + ReduceLROnPlateau", "core_lr": args.core_lr,
                       "readout_lr": args.readout_lr, "epochs_max": args.phase3_epochs,
                       "scheduler_factor": 0.5, "scheduler_patience": 3,
                       "early_stop_patience": args.patience, "select": "validation_loss",
                       "reupload": args.fq_reupload},
            "batch_size": args.batch_size,
        },
        "seed": args.seed, "device": str(device), "output_directory": str(out_dir.resolve()),
        "args": vars(args),
    }
    validate_run_directory(out_dir, config, resume=not args.no_resume)
    write_json(out_dir / "config_mlp_aug.json", config)
    with (out_dir / "split_indices.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "dataset_index"])
        writer.writerows(("train", index) for index in train_indices)
        writer.writerows(("validation", index) for index in validation_indices)

    rows, offset = [], 0
    started = time.perf_counter()

    # Phase 1: original fresh-start pretraining.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.phase1_lr)
    phase_rows, best_state = train_phase(
        tag="fq_mlp_aug_phase1_pretrain", phase="pretrain", model=model, optimizer=optimizer,
        scheduler=None, train_loader=train_loader, validation_loader=validation_loader, generator=generator,
        epochs=args.phase1_epochs, select="val_accuracy", patience=None, offset=offset,
        out_dir=out_dir, device=device, rows=rows, resume=not args.no_resume)
    offset += len(phase_rows)
    write_history(rows, out_dir)

    # Phase 2: continue from phase-1 best with a fresh Adam optimizer.
    train_loader, validation_loader, generator, _, _ = make_loaders(args, device)
    seed_all(args.seed)
    model = build_model(device)
    load_phase_state(model, best_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.phase2_lr)
    phase_rows, best_state = train_phase(
        tag="fq_mlp_aug_phase2_refine", phase="refine", model=model, optimizer=optimizer,
        scheduler=None, train_loader=train_loader, validation_loader=validation_loader, generator=generator,
        epochs=args.phase2_epochs, select="val_accuracy", patience=None, offset=offset,
        out_dir=out_dir, device=device, rows=rows, resume=not args.no_resume)
    offset += len(phase_rows)
    write_history(rows, out_dir)

    # Phase 3: add reupload layers and continue from phase-2 best.
    train_loader, validation_loader, generator, _, _ = make_loaders(args, device)
    seed_all(args.seed)
    model = build_model(device, reupload=args.fq_reupload)
    load_phase_state(model, best_state, allow_new_reupload=args.fq_reupload)
    core_params, readout_params = [], []
    for name, parameter in model.named_parameters():
        (readout_params if name.startswith("readout.") or ".readout." in name else core_params).append(parameter)
    optimizer = torch.optim.Adam([{"params": core_params, "lr": args.core_lr},
                                  {"params": readout_params, "lr": args.readout_lr}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-7)
    phase_rows, best_state = train_phase(
        tag="fq_mlp_aug_phase3_reupload", phase="reupload", model=model, optimizer=optimizer,
        scheduler=scheduler, train_loader=train_loader, validation_loader=validation_loader, generator=generator,
        epochs=args.phase3_epochs, select="val_loss", patience=args.patience, offset=offset,
        out_dir=out_dir, device=device, rows=rows, resume=not args.no_resume)
    write_history(rows, out_dir)

    best_path = out_dir / "full_quantum_mlp_aug_best.pt"
    save_atomic({"model_state_dict": best_state, "config": config}, best_path)
    best_row = min(rows, key=lambda item: item["validation_loss"])
    best_accuracy_row = max(rows, key=lambda item: item["validation_accuracy"])
    summary = {
        "epochs_completed": len(rows), "best_validation_loss": best_row["validation_loss"],
        "validation_accuracy_at_best_loss": best_row["validation_accuracy"],
        "best_loss_epoch": best_row["epoch"],
        "best_validation_accuracy": best_accuracy_row["validation_accuracy"],
        "best_accuracy_epoch": best_accuracy_row["epoch"],
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_path.resolve()), "test_set_used": False,
    }
    with (out_dir / "summary_mlp_aug.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_plots(rows, out_dir)
    print(f"Best checkpoint: {best_path}", flush=True)
    print(f"Best validation accuracy: {summary['best_validation_accuracy']:.4%} "
          f"(epoch {summary['best_accuracy_epoch']})", flush=True)
    print(f"Run artifacts: {out_dir.resolve()}", flush=True)

if __name__ == "__main__":
    main()

