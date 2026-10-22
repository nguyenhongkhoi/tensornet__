"""Fine-tune the full-quantum model on training examples it previously misclassified."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import FullQuantumQCNN  # noqa: E402
from models.reupload import add_reupload  # noqa: E402
from run_convergence import FQ_ARCH, plot_all, save_atomic, seed_all, write_history  # noqa: E402


def read_train_rows(csv_path: Path, train_indices: set[int]):
    """Read the complete per-sample train CSV and separate original errors/correct rows."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")
    hard: dict[int, int] = {}
    normal: dict[int, int] = {}
    seen: set[int] = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_index", "true_label", "correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV cần có các cột: {sorted(required)}")
        for row in reader:
            if row.get("split") not in (None, "", "train"):
                raise ValueError(f"CSV phải là train_per_sample.csv; gặp split={row['split']!r}")
            index, label, correct = int(row["dataset_index"]), int(row["true_label"]), int(row["correct"])
            if index not in train_indices:
                raise ValueError(f"Index {index} không thuộc train split; kiểm tra CSV/split seed.")
            if index in seen:
                raise ValueError(f"dataset_index bị lặp trong CSV: {index}")
            if correct not in (0, 1):
                raise ValueError(f"Giá trị correct không hợp lệ tại index {index}: {correct}")
            seen.add(index)
            (normal if correct else hard)[index] = label
    missing_indices = train_indices - seen
    if missing_indices:
        raise ValueError(f"CSV thiếu {len(missing_indices)} index thuộc train split; cần CSV đầy đủ 50.000 mẫu.")
    if not hard or not normal:
        raise ValueError(f"CSV phải có cả mẫu sai và mẫu đúng: {csv_path}")
    return sorted(hard.items()), sorted(normal.items())


def build_model(checkpoint: Path, device: torch.device):
    if not checkpoint.is_file():
        roots = [ROOT / "runs" / "full_quantum", ROOT / "runs" / "full_quantum_v2", ROOT / "full_quantum"]
        matches = [p for root in roots if root.is_dir() for p in root.rglob(checkpoint.name) if p.is_file()]
        if len(matches) == 1:
            checkpoint = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError("Có nhiều checkpoint trùng tên; truyền đường dẫn đầy đủ với --checkpoint:\n" +
                                    "\n".join(str(p.resolve()) for p in matches))
        else:
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH)
    add_reupload(model, input_sites=256, final_qubits=8, init_scale=0.01,
                 merge_rounds=2, entangling_type="strongly_entangling", n_layers=4)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint không có model_state_dict: {checkpoint}")
    state = payload["model_state_dict"]
    current = model.state_dict()
    missing, unexpected = set(current) - set(state), set(state) - set(current)
    allowed_missing = {key for key in missing if key.endswith(".reupload_scale")}
    if missing != allowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint/model không khớp. missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.load_state_dict(state, strict=False)
    return model.to(device)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mistakes-csv", default="runs/full_quantum_sample_analysis/train_per_sample.csv",
                        help="CSV train_per_sample.csv; validation CSV không được dùng làm dữ liệu train")
    parser.add_argument("--checkpoint", default="runs/full_quantum/fq_phase3_reupload_best.pt")
    parser.add_argument("--data-root", default="mnist_data")
    parser.add_argument("--out-dir", default="runs/full_quantum_hard_finetune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normal-per-hard", type=int, default=2,
                        help="Số lần lấy mẫu train đúng kỳ vọng trên mỗi mẫu sai trong mỗi epoch")
    parser.add_argument("--core-lr", type=float, default=1e-5)
    parser.add_argument("--readout-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.normal_per_hard < 1:
        parser.error("--epochs, --batch-size, --patience, và --normal-per-hard phải lớn hơn 0")

    seed_all(args.seed)
    device = torch.device(args.device)
    dataset = datasets.MNIST(root=args.data_root, train=True, download=True,
                             transform=transforms.ToTensor())
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    validation_indices = permutation[:10000]
    train_indices = permutation[10000:]
    hard_samples, normal_pool = read_train_rows(Path(args.mistakes_csv), set(train_indices))
    for index, expected_label in hard_samples + normal_pool:
        actual_label = int(dataset.targets[index])
        if actual_label != expected_label:
            raise ValueError(f"Nhãn CSV không khớp MNIST tại index {index}: {expected_label} != {actual_label}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(Path(args.checkpoint), device)
    core, readout = [], []
    for name, parameter in model.named_parameters():
        (readout if name.startswith("readout.") or ".readout." in name else core).append(parameter)
    optimizer = torch.optim.Adam([{"params": core, "lr": args.core_lr},
                                  {"params": readout, "lr": args.readout_lr}])
    hard_indices = {i for i, _ in hard_samples}
    # Draw a fresh weighted sample every epoch from the full original training split.
    # The weight ratio gives the hard group 1/(1 + normal_per_hard) of expected draws.
    hard_weight = len(normal_pool) / (args.normal_per_hard * len(hard_samples))
    weights = torch.tensor([hard_weight if index in hard_indices else 1.0 for index in train_indices],
                           dtype=torch.double)
    samples_per_epoch = len(hard_samples) * (1 + args.normal_per_hard)
    sampler = WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=args.batch_size,
                                   shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    print(f"Fine-tuning from full TRAIN split with weighted sampling: {len(hard_samples)} hard, "
          f"{len(normal_pool)} originally correct available; {samples_per_epoch} draws/epoch, "
          f"expected ratio 1 hard : {args.normal_per_hard} normal.", flush=True)
    print(f"Validation: all {len(validation_indices)} held-out samples (no training updates).", flush=True)
    rows, best_loss, stale = [], float("inf"), 0
    started = time.perf_counter()
    best_path = out_dir / "hard_finetune_best.pt"
    last_path = out_dir / "hard_finetune_last.pt"
    config = {"source_checkpoint": str(Path(args.checkpoint).resolve()),
              "mistakes_csv": str(Path(args.mistakes_csv).resolve()),
              "seed": args.seed, "split_seed": args.split_seed,
              "hard_count": len(hard_samples), "normal_pool_count": len(normal_pool),
              "normal_per_hard_expected": args.normal_per_hard,
              "samples_per_epoch": samples_per_epoch,
              "validation_count": len(validation_indices), "args": vars(args)}
    (out_dir / "finetune_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, validation_loader, device)
        row = dict(epoch=epoch, phase="hard_example_finetune", train_loss=train_loss,
                   train_accuracy=train_acc, validation_loss=val_loss, validation_accuracy=val_acc,
                   lr_core=args.core_lr, lr_readout=args.readout_lr,
                   epoch_seconds=time.perf_counter() - epoch_start)
        rows.append(row)
        improved = val_loss < best_loss - 1e-7
        if improved:
            best_loss, stale = val_loss, 0
            save_atomic({"model_state_dict": model.state_dict(), "epoch": epoch, "row": row,
                         "source_checkpoint": str(Path(args.checkpoint).resolve()),
                         "hard_indices": [i for i, _ in hard_samples],
                         "normal_pool_count": len(normal_pool),
                         "normal_per_hard_expected": args.normal_per_hard}, best_path)
        else:
            stale += 1
        save_atomic({"model_state_dict": model.state_dict(), "epoch": epoch, "row": row,
                     "optimizer_state_dict": optimizer.state_dict(), "rows": rows}, last_path)
        write_history(out_dir / "history_full_quantum.csv", rows)
        print(f"[hard_finetune] epoch {epoch:2d}/{args.epochs} "
              f"train {train_loss:.4f}/{train_acc:.4f} val {val_loss:.4f}/{val_acc:.4f} "
              f"{row['epoch_seconds']:.0f}s{' *best' if improved else ''}", flush=True)
        if stale >= args.patience:
            print(f"Early stop: validation loss did not improve for {args.patience} epochs.", flush=True)
            break

    plot_all(out_dir)
    print(f"Best checkpoint: {best_path}", flush=True)
    print(f"Plots and history: {out_dir.resolve()} (elapsed {time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
