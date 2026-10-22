"""Count and plot true-label -> predicted-label pairs for misclassified MNIST rows."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mistakes(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_index", "true_label", "predicted_label", "correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        rows = []
        for row in reader:
            if row.get("split") not in (None, "", "train"):
                raise ValueError(f"Expected train CSV, found split={row['split']!r}")
            if int(row["correct"]) != 0:
                continue
            true_label, predicted_label = int(row["true_label"]), int(row["predicted_label"])
            if not (0 <= true_label <= 9 and 0 <= predicted_label <= 9):
                raise ValueError(f"Labels must be 0..9; got {true_label}->{predicted_label}")
            if true_label == predicted_label:
                raise ValueError(f"Incorrect row has equal labels at dataset_index={row['dataset_index']}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No misclassified rows (correct=0) in {csv_path}")
    return rows


def write_reports(rows: list[dict], out_dir: Path) -> list[tuple[int, int, int]]:
    pairs = Counter((int(row["true_label"]), int(row["predicted_label"])) for row in rows)
    errors_by_true = Counter(int(row["true_label"]) for row in rows)
    ranked = sorted(((true, pred, count) for (true, pred), count in pairs.items()),
                    key=lambda item: (-item[2], item[0], item[1]))
    total = len(rows)

    with (out_dir / "mistake_pairs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["true_label", "predicted_label", "count", "percent_of_all_errors",
                  "errors_for_true_label", "percent_within_true_label"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for true, pred, count in ranked:
            writer.writerow({"true_label": true, "predicted_label": pred, "count": count,
                             "percent_of_all_errors": f"{100 * count / total:.2f}",
                             "errors_for_true_label": errors_by_true[true],
                             "percent_within_true_label": f"{100 * count / errors_by_true[true]:.2f}"})

    detail_fields = ["dataset_index", "true_label", "predicted_label", "confidence",
                     "true_class_probability", "margin", "cross_entropy"]
    available_fields = [field for field in detail_fields if field in rows[0]]
    detail_rows = sorted(rows, key=lambda row: (-float(row.get("cross_entropy", 0)),
                                                 int(row["true_label"]), int(row["predicted_label"])))
    with (out_dir / "mistake_samples.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=available_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in available_fields} for row in detail_rows)
    return ranked


def plot_pairs(rows: list[dict], ranked: list[tuple[int, int, int]], out_dir: Path, top_k: int) -> None:
    matrix = np.zeros((10, 10), dtype=int)
    for row in rows:
        matrix[int(row["true_label"]), int(row["predicted_label"])] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    image = axes[0].imshow(matrix, cmap="Reds")
    axes[0].set(title="Mistake counts: true label → predicted label",
                xlabel="Predicted label", ylabel="True label",
                xticks=range(10), yticks=range(10))
    for true in range(10):
        for pred in range(10):
            if matrix[true, pred]:
                axes[0].text(pred, true, str(matrix[true, pred]), ha="center", va="center",
                             color="white" if matrix[true, pred] > matrix.max() * .5 else "black", fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=.046, pad=.04, label="Number of errors")

    shown = ranked[:top_k]
    labels = [f"{true} → {pred}" for true, pred, _ in reversed(shown)]
    counts = [count for _, _, count in reversed(shown)]
    axes[1].barh(labels, counts, color="#d95f59")
    axes[1].set(title=f"Top {len(shown)} confusion pairs", xlabel="Number of errors")
    axes[1].grid(axis="x", alpha=.25)
    for y, count in enumerate(counts):
        axes[1].text(count, y, f" {count}", va="center", fontsize=9)

    fig.suptitle(f"{len(rows):,} misclassified training samples")
    fig.tight_layout()
    output_path = out_dir / "mistake_pairs.png"
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="runs/full_quantum_sample_analysis/train_per_sample.csv",
                        help="Per-sample train CSV generated by analyze_samples.py")
    parser.add_argument("--out-dir", default="runs/full_quantum_sample_analysis/mistake_pair_analysis")
    parser.add_argument("--top-k", type=int, default=15, help="Number of frequent pairs shown in the bar chart")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_mistakes(Path(args.csv))
    ranked = write_reports(rows, out_dir)
    plot_pairs(rows, ranked, out_dir, args.top_k)
    print(f"Misclassified samples: {len(rows)}")
    print("Most frequent true → predicted pairs:")
    for true, pred, count in ranked[:args.top_k]:
        print(f"  {true} → {pred}: {count} ({100 * count / len(rows):.2f}% of all errors)")
    print(f"Saved tables under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
