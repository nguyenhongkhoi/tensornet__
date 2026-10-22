from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import FullQuantumQCNN
from models.reupload import add_reupload

ROOT = Path(__file__).resolve().parent

FQ_ARCH = dict(
    target_size=(16, 16), grayscale=True, final_qubits=8,
    readout_type="pauli_observable", pair_correlators="full9",
    init_scale=0.01, merge_rounds=2,
    entangling_type="strongly_entangling", n_layers=4,
    quantum_channels=3,
)


def load_checkpoint(path: Path):
    """Load from the configured path, extracted project files, or the project ZIP."""
    path = path.expanduser()
    if path.is_file():
        resolved = path.resolve()
        return torch.load(resolved, map_location="cpu", weights_only=False), str(resolved)

    # In the user's project, training outputs are under runs/full_quantum.
    search_roots = [ROOT / "runs" / "full_quantum", ROOT / "runs" / "full_quantum_v2",
                    ROOT / "full_quantum"]
    matches = [match for root in search_roots if root.is_dir()
               for match in root.rglob(path.name) if match.is_file()]
    if len(matches) == 1 and matches[0].is_file():
        resolved = matches[0].resolve()
        return torch.load(resolved, map_location="cpu", weights_only=False), str(resolved)
    if len(matches) > 1:
        found = "\n".join(str(match.resolve()) for match in matches)
        raise FileNotFoundError(
            f"Found multiple checkpoints named {path.name}; set --checkpoint to one of these:\n{found}"
        )

    # The project may still have the checkpoint inside full_quantum.zip.
    zip_matches = []
    for archive in ROOT.glob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as zipped:
                entries = [name for name in zipped.namelist()
                           if Path(name).name.lower() == path.name.lower()]
                zip_matches.extend((archive, entry) for entry in entries)
        except zipfile.BadZipFile:
            continue
    if len(zip_matches) == 1:
        archive, entry = zip_matches[0]
        with zipfile.ZipFile(archive) as zipped:
            payload = torch.load(io.BytesIO(zipped.read(entry)), map_location="cpu", weights_only=False)
        return payload, f"{archive.resolve()}::{entry}"
    if len(zip_matches) > 1:
        found = "\n".join(f"{archive}::{entry}" for archive, entry in zip_matches)
        raise FileNotFoundError(f"Found multiple matching checkpoints in ZIP files:\n{found}")
    raise FileNotFoundError(
        f"Checkpoint not found at {path}, under runs/full_quantum, runs/full_quantum_v2, "
        f"or full_quantum, or in ZIP files in {ROOT}."
    )


def evaluate_split(model, dataset, indices, split, batch_size, device, out_dir):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    out_path = out_dir / f"{split}_per_sample.csv"
    fields = ["split", "dataset_index", "true_label", "predicted_label",
              "correct", "confidence", "true_class_probability", "margin", "cross_entropy"]
    offset = 0
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                probs = logits.softmax(dim=1)
                confidence, prediction = probs.max(dim=1)
                true_prob = probs.gather(1, labels[:, None]).squeeze(1)
                top2 = probs.topk(2, dim=1).values
                losses = F.cross_entropy(logits, labels, reduction="none")
                for j in range(len(labels)):
                    writer.writerow({
                        "split": split,
                        "dataset_index": indices[offset + j],
                        "true_label": int(labels[j]),
                        "predicted_label": int(prediction[j]),
                        "correct": int(prediction[j] == labels[j]),
                        "confidence": f"{float(confidence[j]):.8f}",
                        "true_class_probability": f"{float(true_prob[j]):.8f}",
                        "margin": f"{float(top2[j, 0] - top2[j, 1]):.8f}",
                        "cross_entropy": f"{float(losses[j]):.8f}",
                    })
                offset += len(labels)
    print(f"{split}: {len(indices)} samples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-sample MNIST evaluation for Full Quantum QCNN")
    parser.add_argument("--data-root", default="mnist_data")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "runs" / "full_quantum" / "fq_phase3_reupload_best.pt"),
        help="Path to the saved full-quantum checkpoint",
    )
    parser.add_argument("--out-dir", default="runs/full_quantum_sample_analysis")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train", "validation"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH)
    add_reupload(model, input_sites=256, final_qubits=8, init_scale=0.01,
                 merge_rounds=2, entangling_type="strongly_entangling", n_layers=4)
    payload, checkpoint_source = load_checkpoint(Path(args.checkpoint))
    print(f"Using checkpoint: {checkpoint_source}")
    state = payload["model_state_dict"]
    model_state = model.state_dict()
    missing = set(model_state) - set(state)
    unexpected = set(state) - set(model_state)
    allowed_missing = {key for key in missing if key.endswith(".reupload_scale")}
    if missing != allowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    dataset = datasets.MNIST(root=args.data_root, train=True, download=True,
                             transform=transforms.ToTensor())
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(20260911)).tolist()
    val_indices, train_indices = permutation[:10000], permutation[10000:]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        indices = train_indices if split == "train" else val_indices
        evaluate_split(model, dataset, indices, split, args.batch_size, device, out_dir)


if __name__ == "__main__":
    main()
