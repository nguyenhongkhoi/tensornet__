"""Train a fresh full-quantum QCNN through all three phases using only hard train samples."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import FullQuantumQCNN  # noqa: E402
from models.reupload import add_reupload  # noqa: E402
from run_convergence import (FQ_ARCH, plot_all, save_atomic, seed_all,
                             write_history)  # noqa: E402


def read_hard_train(csv_path: Path, train_indices: set[int]) -> list[tuple[int, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")
    hard: dict[int, int] = {}
    seen: set[int] = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_index", "true_label", "correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV cần có các cột: {sorted(required)}")
        for row in reader:
            if row.get("split") not in (None, "", "train"):
                raise ValueError(f"Cần train_per_sample.csv, gặp split={row['split']!r}")
            index, label = int(row["dataset_index"]), int(row["true_label"])
            if index not in train_indices:
                raise ValueError(f"Index {index} không thuộc train split; kiểm tra split-seed.")
            if index in seen:
                raise ValueError(f"dataset_index bị lặp: {index}")
            seen.add(index)
            if int(row["correct"]) == 0:
                hard[index] = label
    missing = train_indices - seen
    if missing:
        raise ValueError(f"CSV thiếu {len(missing)} index train; cần CSV đầy đủ 50.000 mẫu.")
    if not hard:
        raise ValueError("CSV không có mẫu train nào bị dự đoán sai.")
    return sorted(hard.items())


def build_model(reupload: bool, device: torch.device):
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH)
    if reupload:
        add_reupload(model, input_sites=256, final_qubits=8, init_scale=0.01,
                     merge_rounds=2, entangling_type="strongly_entangling", n_layers=4)
    return model.to(device)


def load_phase_state(model, state: dict, allow_reupload_init: bool = False) -> None:
    current = model.state_dict()
    missing, unexpected = set(current) - set(state), set(state) - set(current)
    allowed = {key for key in missing if key.endswith(".reupload_scale")} if allow_reupload_init else set()
    if missing != allowed or unexpected:
        raise RuntimeError(f"Phase checkpoint/model mismatch. missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.load_state_dict(state, strict=False)


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
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += len(labels)
    return loss_sum / seen, correct / seen


def train_phase(*, tag, phase, model, optimizer, scheduler, train_loader, val_loader,
                epochs, select, patience, offset, rows, out_dir, device):
    sign = 1.0 if select == "val_loss" else -1.0
    best_metric, stale = float("inf"), 0
    best_path, last_path = out_dir / f"{tag}_best.pt", out_dir / f"{tag}_last.pt"
    for epoch in range(1, epochs + 1):
        began = time.perf_counter()
        lrs = [group["lr"] for group in optimizer.param_groups]
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        row = dict(epoch=offset + epoch, phase=phase, train_loss=train_loss,
                   train_accuracy=train_acc, validation_loss=val_loss, validation_accuracy=val_acc,
                   lr_core=lrs[0], lr_readout=lrs[-1], epoch_seconds=time.perf_counter() - began)
        rows.append(row)
        metric_value = val_loss if select == "val_loss" else val_acc
        improved = sign * metric_value < best_metric - 1e-7
        if improved:
            best_metric, stale = sign * metric_value, 0
            save_atomic({"model_state_dict": model.state_dict(), "epoch": epoch,
                         "global_epoch": offset + epoch, "row": row}, best_path)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        save_atomic({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                     "epoch": epoch, "rows": rows}, last_path)
        write_history(out_dir / "history_full_quantum.csv", rows)
        print(f"[{tag}] epoch {epoch:3d}/{epochs} train {train_loss:.4f}/{train_acc:.4f} "
              f"val {val_loss:.4f}/{val_acc:.4f} lr {lrs[0]:.2e}/{lrs[-1]:.2e} "
              f"{row['epoch_seconds']:.0f}s{' *best' if improved else ''}", flush=True)
        if patience and stale >= patience:
            print(f"[{tag}] early stop after {patience} epochs without improvement in {select}.", flush=True)
            break
    best = torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(best)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-csv", default="runs/full_quantum_sample_analysis/train_per_sample.csv")
    parser.add_argument("--data-root", default="mnist_data")
    parser.add_argument("--out-dir", default="runs/full_quantum_hard_from_scratch_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase3-epochs", type=int, default=60)
    parser.add_argument("--phase3-patience", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch_size, args.phase1_epochs, args.phase2_epochs,
           args.phase3_epochs, args.phase3_patience) < 1:
        parser.error("batch-size, epoch counts, and patience must be positive")

    seed_all(args.seed)
    device = torch.device(args.device)
    dataset = datasets.MNIST(root=args.data_root, train=True, download=True,
                             transform=transforms.ToTensor())
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    val_indices, train_indices = permutation[:10000], permutation[10000:]
    hard_samples = read_hard_train(Path(args.hard_csv), set(train_indices))
    for index, label in hard_samples:
        if int(dataset.targets[index]) != label:
            raise ValueError(f"CSV/MNIST label mismatch at dataset_index={index}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(Subset(dataset, [i for i, _ in hard_samples]),
                              batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    config = {"initialization": "random; no pretrained checkpoint loaded",
              "training_data": "only rows with correct=0 from train CSV",
              "hard_count": len(hard_samples), "validation_count": len(val_indices),
              "seed": args.seed, "split_seed": args.split_seed, "args": vars(args),
              "phase_plan": [
                  {"phase": "pretrain", "epochs": args.phase1_epochs, "lr": 2e-3,
                   "selection": "validation accuracy"},
                  {"phase": "refine", "epochs": args.phase2_epochs, "lr": 2e-4,
                   "selection": "validation accuracy"},
                  {"phase": "reupload", "epochs": args.phase3_epochs,
                   "core_lr": 1e-4, "readout_lr": 1e-3,
                   "selection": "validation loss", "patience": args.phase3_patience}],
              "test_set_used": False}
    (out_dir / "from_scratch_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Fresh random initialization; no pretrained checkpoint will be loaded.", flush=True)
    print(f"Training data: {len(hard_samples)} hard TRAIN samples only; validation: all {len(val_indices)} samples.", flush=True)

    rows, offset = [], 0
    model = build_model(reupload=False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    rows = train_phase(tag="phase1_pretrain", phase="pretrain", model=model, optimizer=optimizer,
                       scheduler=None, train_loader=train_loader, val_loader=val_loader,
                       epochs=args.phase1_epochs, select="val_acc", patience=None, offset=offset,
                       rows=rows, out_dir=out_dir, device=device)
    offset = len(rows)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    rows = train_phase(tag="phase2_refine", phase="refine", model=model, optimizer=optimizer,
                       scheduler=None, train_loader=train_loader, val_loader=val_loader,
                       epochs=args.phase2_epochs, select="val_acc", patience=None, offset=offset,
                       rows=rows, out_dir=out_dir, device=device)
    offset = len(rows)

    reupload_model = build_model(reupload=True, device=device)
    load_phase_state(reupload_model, model.state_dict(), allow_reupload_init=True)
    core, readout = [], []
    for name, parameter in reupload_model.named_parameters():
        (readout if name.startswith("readout.") or ".readout." in name else core).append(parameter)
    optimizer = torch.optim.Adam([{"params": core, "lr": 1e-4}, {"params": readout, "lr": 1e-3}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-7)
    train_phase(tag="phase3_reupload", phase="reupload", model=reupload_model,
                optimizer=optimizer, scheduler=scheduler, train_loader=train_loader,
                val_loader=val_loader, epochs=args.phase3_epochs, select="val_loss",
                patience=args.phase3_patience, offset=offset, rows=rows,
                out_dir=out_dir, device=device)

    plot_all(out_dir)
    print(f"Fresh-from-scratch artifacts and plots: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
