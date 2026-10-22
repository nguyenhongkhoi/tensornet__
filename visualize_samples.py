"""Visualize per-sample MNIST evaluation CSVs created by analyze_samples.py."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV không có dữ liệu: {path}")
    required = {"dataset_index", "true_label", "predicted_label", "correct",
                "confidence", "true_class_probability", "cross_entropy"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Thiếu cột {sorted(missing)} trong {path}")
    for row in rows:
        for key in ("dataset_index", "true_label", "predicted_label", "correct"):
            row[key] = int(row[key])
        for key in ("confidence", "true_class_probability", "cross_entropy"):
            row[key] = float(row[key])
    return rows


def plot_summary(rows: list[dict], split: str, out_dir: Path) -> None:
    labels = np.array([r["true_label"] for r in rows])
    preds = np.array([r["predicted_label"] for r in rows])
    losses = np.array([r["cross_entropy"] for r in rows])
    correct = np.array([r["correct"] for r in rows], dtype=bool)
    confidence = np.array([r["confidence"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels, preds), 1)
    im = axes[0].imshow(matrix, cmap="Blues")
    axes[0].set(title="Confusion matrix", xlabel="Dự đoán", ylabel="Nhãn thật",
                xticks=range(10), yticks=range(10))
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    per_digit = [correct[labels == digit].mean() if np.any(labels == digit) else 0
                 for digit in range(10)]
    axes[1].bar(range(10), per_digit, color="#4c78a8")
    axes[1].set(title="Accuracy theo chữ số", xlabel="Nhãn thật", ylabel="Accuracy",
                xticks=range(10), ylim=(0, 1.02))
    axes[1].grid(axis="y", alpha=.25)

    axes[2].scatter(confidence[correct], losses[correct], s=10, alpha=.4, label="Đúng")
    axes[2].scatter(confidence[~correct], losses[~correct], s=14, alpha=.7, label="Sai")
    axes[2].set(title="Confidence và cross entropy", xlabel="Confidence dự đoán", ylabel="Cross entropy")
    axes[2].legend()
    fig.suptitle(f"{split}: accuracy {correct.mean():.2%} | {len(rows):,} mẫu")
    fig.tight_layout()
    path = out_dir / f"{split}_summary.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu {path}")


def plot_worst(rows: list[dict], dataset, split: str, count: int, out_dir: Path) -> None:
    # Prioritize mistakes; among them show the largest loss. If all are correct,
    # show the lowest true-class probability (hardest correctly classified cases).
    wrong = [r for r in rows if not r["correct"]]
    candidates = wrong if wrong else rows
    candidates = sorted(candidates, key=lambda r: r["cross_entropy"], reverse=True)[:count]
    if not candidates:
        return
    cols = min(6, len(candidates))
    grid_rows = (len(candidates) + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(2.5 * cols, 2.9 * grid_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, candidates):
        image, _ = dataset[row["dataset_index"]]
        ax.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        status = "SAI" if not row["correct"] else "đúng, khó"
        ax.set_title(
            f'{status}: {row["true_label"]} → {row["predicted_label"]}\n'
            f'p(nhãn thật)={row["true_class_probability"]:.3f}, loss={row["cross_entropy"]:.2f}',
            color="#c62828" if not row["correct"] else "#333333", fontsize=9,
        )
    fig.suptitle(f"{split}: {len(wrong)} mẫu dự đoán sai — hiển thị tối đa {count} mẫu tệ nhất")
    fig.tight_layout()
    path = out_dir / f"{split}_worst_samples.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ và ảnh MNIST từ CSV kết quả theo từng mẫu")
    parser.add_argument("--train-csv", default="runs/full_quantum_sample_analysis/train_per_sample.csv")
    parser.add_argument("--validation-csv", default="runs/full_quantum_sample_analysis/validation_per_sample.csv")
    parser.add_argument("--data-root", default="mnist_data", help="Thư mục MNIST đã tải; cần cho ảnh mẫu")
    parser.add_argument("--out-dir", default="runs/full_quantum_sample_analysis/plots")
    parser.add_argument("--top-k", type=int, default=30, help="Số ảnh mẫu tệ nhất cần vẽ cho mỗi split")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train", "validation"])
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k phải lớn hơn 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = {"train": Path(args.train_csv), "validation": Path(args.validation_csv)}
    # These are the same deterministic indices used by analyze_samples.py.
    permutation = torch.randperm(60000, generator=torch.Generator().manual_seed(20260911)).tolist()
    validation_indices = set(permutation[:10000])
    dataset = datasets.MNIST(root=args.data_root, train=True, download=False,
                             transform=transforms.ToTensor())
    for split in args.splits:
        rows = read_rows(csv_paths[split])
        expected_validation = split == "validation"
        # Warn rather than fail if CSV came from a different split seed.
        mismatch = sum((r["dataset_index"] in validation_indices) != expected_validation for r in rows)
        if mismatch:
            print(f"Lưu ý: {mismatch} dataset_index không khớp split seed mặc định; ảnh vẫn lấy theo index CSV.")
        plot_summary(rows, split, out_dir)
        plot_worst(rows, dataset, split, args.top_k, out_dir)


if __name__ == "__main__":
    main()
