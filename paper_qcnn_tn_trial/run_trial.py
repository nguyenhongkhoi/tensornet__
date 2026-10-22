"""Train a small MNIST trial of the paper-inspired 8-qubit QCNN/TN model."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from paper_qcnn_tn import PaperMultiClassQCNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def prepare_data(args: argparse.Namespace):
    dataset = datasets.MNIST(root=args.data_root, train=True, download=True)
    split_order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.split_seed))
    validation_indices = split_order[:10_000]
    train_pool_indices = split_order[10_000:]
    if not 5_000 <= args.train_samples <= 10_000:
        raise ValueError("--train-samples phải nằm trong khoảng 5000..10000")
    if args.train_samples > len(train_pool_indices):
        raise ValueError("Số train mẫu yêu cầu lớn hơn train pool")

    local_order = torch.randperm(len(train_pool_indices),
                                 generator=torch.Generator().manual_seed(args.sample_seed))
    selected_indices = train_pool_indices[local_order[:args.train_samples]]
    pixels = dataset.data.reshape(len(dataset), -1).numpy().astype(np.float32) / 255.0
    labels = dataset.targets.numpy().astype(np.int64)
    selected_np = selected_indices.numpy()
    validation_np = validation_indices.numpy()

    # Fit PCA only on the sampled training examples; validation is transform-only.
    pca = PCA(n_components=256, svd_solver="randomized", random_state=args.seed)
    x_train = pca.fit_transform(pixels[selected_np]).astype(np.float32)
    x_validation = pca.transform(pixels[validation_np]).astype(np.float32)
    y_train = labels[selected_np]
    y_validation = labels[validation_np]

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    validation_dataset = TensorDataset(torch.from_numpy(x_validation), torch.from_numpy(y_validation))
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              generator=loader_generator, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0, pin_memory=torch.cuda.is_available())
    return (train_loader, validation_loader, loader_generator, selected_np, y_train,
            validation_np, y_validation, pca.explained_variance_ratio_.sum())


def evaluate(model: PaperMultiClassQCNN, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            probabilities = model(features)
            loss = F.nll_loss(probabilities.clamp_min(1e-12).log(), labels, reduction="sum")
            predictions = model.predict_digits(probabilities)
            total_loss += float(loss)
            total_correct += int((predictions == labels).sum())
            total += labels.numel()
    return total_loss / total, total_correct / total


def save_history(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in rows], marker="o", label="validation")
    axes[0].set(title="Cross entropy", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in rows], marker="o", label="train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in rows], marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.suptitle("Paper-inspired 8-qubit QCNN tensor-network trial")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="mnist_data")
    parser.add_argument("--out-dir", default="runs/paper_qcnn_tn_trial_5k")
    parser.add_argument("--train-samples", type=int, default=5000,
                        help="Random subset size from the 50k train pool: 5000..10000")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--sample-seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("--epochs, --batch-size phải dương và --lr phải lớn hơn 0")
    if not 5000 <= args.train_samples <= 10000:
        parser.error("--train-samples phải từ 5000 đến 10000")

    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (train_loader, validation_loader, loader_generator, train_indices, train_labels,
     validation_indices, validation_labels, pca_variance) = prepare_data(args)

    config = vars(args).copy()
    config.update({"model": "paper-inspired 8-qubit QCNN, exact state-tensor contraction",
                   "train_pool_size": 50_000, "validation_size": 10_000,
                   "num_trainable_angles": 105, "pca_components": 256,
                   "pca_explained_variance_ratio_sum": float(pca_variance),
                   "readout_wires": [0, 2, 4, 6], "num_readout_outcomes": 16,
                   "digit_predictions_use_outcomes": "0..9"})
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (out_dir / "sampled_train_indices.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_index", "true_label"])
        writer.writerows(zip(train_indices.tolist(), train_labels.tolist()))

    class_counts = np.bincount(train_labels, minlength=10).tolist()
    print(f"Device: {device}", flush=True)
    print(f"Random train subset: {len(train_indices)} / 50,000; class counts={class_counts}", flush=True)
    print("Validation: all 10,000 held-out examples from the official MNIST training split.", flush=True)
    print(f"PCA(256) variance retained: {pca_variance:.4f}; fitted on sampled train subset only.", flush=True)

    model = PaperMultiClassQCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows: list[dict] = []
    best_accuracy, best_loss = -1.0, float("inf")
    best_path, last_path = out_dir / "paper_qcnn_tn_best.pt", out_dir / "paper_qcnn_tn_last.pt"
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(features)
            loss = F.nll_loss(probabilities.clamp_min(1e-12).log(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_correct += int((model.predict_digits(probabilities.detach()) == labels).sum())
            total += labels.numel()

        train_loss, train_accuracy = total_loss / total, total_correct / total
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
               "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
               "lr": args.lr, "epoch_seconds": time.perf_counter() - epoch_start}
        rows.append(row)
        improved = (validation_accuracy > best_accuracy or
                    (validation_accuracy == best_accuracy and validation_loss < best_loss))
        if improved:
            best_accuracy, best_loss = validation_accuracy, validation_loss
            save_checkpoint({"model_state_dict": model.state_dict(), "epoch": epoch,
                             "row": row, "config": config}, best_path)
        save_checkpoint({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                         "epoch": epoch, "rows": rows,
                         "loader_generator_state": loader_generator.get_state(), "config": config}, last_path)
        save_history(rows, out_dir)
        print(f"[paper_qcnn_tn] epoch {epoch:2d}/{args.epochs} "
              f"train {train_loss:.4f}/{train_accuracy:.4f} "
              f"val {validation_loss:.4f}/{validation_accuracy:.4f} "
              f"{row['epoch_seconds']:.1f}s{' *best' if improved else ''}", flush=True)

    summary = {"best_epoch": int(max(rows, key=lambda r: (r["validation_accuracy"], -r["validation_loss"]))["epoch"]),
               "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss,
               "train_samples": len(train_indices), "validation_samples": len(validation_indices),
               "elapsed_seconds": time.perf_counter() - started,
               "checkpoint": str(best_path), "note": "MNIST official test set was not used."}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Best checkpoint: {best_path}", flush=True)
    print(f"History and plot: {out_dir.resolve()}", flush=True)
    print(f"Best validation accuracy: {best_accuracy:.4%} at epoch {summary['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
