"""Train the original 16x16 FullQuantumQCNN with train-only image augmentation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from models import FullQuantumQCNN  # noqa: E402
from models.reupload import add_reupload  # noqa: E402
from training_utils import (  # noqa: E402
    plot_all,
    run_phase,
    seed_all,
    write_history,
    write_json,
)


FQ_ARCH = dict(
    target_size=(16, 16),
    grayscale=True,
    final_qubits=8,
    readout_type="pauli_observable",
    pair_correlators="full9",
    init_scale=0.01,
    merge_rounds=2,
    entangling_type="strongly_entangling",
    n_layers=4,
    quantum_channels=3,
)


def make_augmented_loaders(args: argparse.Namespace, batch_size: int, seed: int):
    if args.fake_data:
        dataset_size = 2_000
        validation_size = 500
        train_base = datasets.FakeData(
            size=dataset_size,
            image_size=(1, 28, 28),
            num_classes=10,
            # FakeData already returns tensors. RandomAffine supports tensor
            # inputs, while ToTensor only accepts PIL images/ndarrays.
            transform=transforms.RandomAffine(
                degrees=args.aug_degrees,
                translate=(args.aug_translate, args.aug_translate),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            random_offset=0,
        )
        validation_base = datasets.FakeData(
            size=dataset_size,
            image_size=(1, 28, 28),
            num_classes=10,
            random_offset=0,
        )
    else:
        dataset_size = 60_000
        validation_size = 10_000
        augmentation = transforms.RandomAffine(
            degrees=args.aug_degrees,
            translate=(args.aug_translate, args.aug_translate),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        train_base = datasets.MNIST(
            root=args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose([augmentation, transforms.ToTensor()]),
        )
        validation_base = datasets.MNIST(
            root=args.data_root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    indices = torch.randperm(
        dataset_size, generator=torch.Generator().manual_seed(args.split_seed)
    ).tolist()
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    if args.train_subset is not None:
        train_indices = train_indices[:args.train_subset]
    if args.val_subset is not None:
        validation_indices = validation_indices[:args.val_subset]

    pin_memory = args.device.startswith("cuda") and torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(train_base, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        Subset(validation_base, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader, generator, train_indices, validation_indices


def make_phase_loaders(args: argparse.Namespace, batch_size: int):
    return make_augmented_loaders(args, batch_size, args.seed)


def build_model(device: torch.device, reupload: bool) -> FullQuantumQCNN:
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH)
    if reupload:
        add_reupload(
            model,
            input_sites=16 * 16,
            final_qubits=FQ_ARCH["final_qubits"],
            init_scale=FQ_ARCH["init_scale"],
            merge_rounds=FQ_ARCH["merge_rounds"],
            entangling_type=FQ_ARCH["entangling_type"],
            n_layers=FQ_ARCH["n_layers"],
        )
    return model.to(device)


def load_model_state(model: FullQuantumQCNN, state: dict, allow_new_reupload: bool = False) -> None:
    current = model.state_dict()
    missing = set(current) - set(state)
    unexpected = set(state) - set(current)
    allowed_missing = {
        key for key in missing
        if allow_new_reupload and key.endswith(".reupload_scale")
    }
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.load_state_dict(state, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "mnist_data"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "augmentation"))
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--fq-batch-size", "--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--fq-pretrain-epochs", "--phase1-epochs", dest="phase1_epochs", type=int, default=20)
    parser.add_argument("--fq-refine-epochs", "--phase2-epochs", dest="phase2_epochs", type=int, default=5)
    parser.add_argument("--fq-epochs", "--phase3-epochs", dest="phase3_epochs", type=int, default=60)
    parser.add_argument("--fq-pretrain-lr", "--phase1-lr", dest="phase1_lr", type=float, default=2e-3)
    parser.add_argument("--fq-refine-lr", "--phase2-lr", dest="phase2_lr", type=float, default=2e-4)
    parser.add_argument("--fq-core-lr", "--core-lr", dest="core_lr", type=float, default=1e-4)
    parser.add_argument("--fq-readout-lr", "--readout-lr", dest="readout_lr", type=float, default=1e-3)
    parser.add_argument("--fq-patience", "--phase3-patience", dest="phase3_patience", type=int, default=10)
    parser.add_argument(
        "--fq-reupload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the original phase-3 reupload core (default: enabled)",
    )
    parser.add_argument("--aug-degrees", type=float, default=5.0)
    parser.add_argument("--aug-translate", type=float, default=0.04)
    parser.add_argument("--acc-ymin", type=float, default=90.0)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--val-subset", type=int)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if min(
        args.batch_size,
        args.phase1_epochs,
        args.phase2_epochs,
        args.phase3_epochs,
        args.phase3_patience,
    ) < 1:
        parser.error("Batch size, epoch counts, and patience must be positive")
    if min(args.phase1_lr, args.phase2_lr, args.core_lr, args.readout_lr) <= 0:
        parser.error("Learning rates must be positive")
    if args.aug_degrees < 0 or not 0 <= args.aug_translate < 0.5:
        parser.error("Invalid augmentation settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but torch.cuda.is_available() is False")
    if args.fake_data and not args.smoke:
        parser.error("--fake-data is only intended for --smoke wiring checks")

    if args.smoke:
        args.fake_data = True
        args.train_subset = args.train_subset or 64
        args.val_subset = args.val_subset or 32
        args.batch_size = min(args.batch_size, 8)
        args.phase1_epochs = 2
        args.phase2_epochs = 1
        args.phase3_epochs = 2
        args.phase3_patience = 10
    return args


def save_split_indices(out_dir: Path, train_indices: list[int], validation_indices: list[int]) -> None:
    with (out_dir / "split_indices.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "dataset_index"])
        writer.writerows(("train", index) for index in train_indices)
        writer.writerows(("validation", index) for index in validation_indices)


def _json_normalize(value):
    """Normalize tuples and other JSON-compatible values before config comparison."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _resume_signature(config: dict) -> dict:
    normalized = _json_normalize(config)
    # These affect presentation or invocation mechanics, not the trained run.
    args = normalized.get("args", {})
    for key in ("acc_ymin", "out_dir", "no_resume"):
        args.pop(key, None)
    return normalized


def validate_output_directory(out_dir: Path, config: dict, *, resume: bool) -> None:
    """Prevent accidental resume from, or overwrite of, a different experiment."""
    config_path = out_dir / "config_augmentation.json"
    artifact_patterns = (
        "fq_aug_*_last.pt",
        "fq_aug_*_best.pt",
        "fq_aug_*_done.json",
        "history_full_quantum.csv",
        "split_indices.csv",
        "summary.json",
    )
    artifacts = [path for pattern in artifact_patterns for path in out_dir.glob(pattern)]

    if artifacts and not config_path.exists():
        raise RuntimeError(
            f"Output directory contains run artifacts but no {config_path.name}: {out_dir}. "
            "Choose a new --out-dir so this run cannot mix with unknown checkpoints."
        )

    if artifacts and not resume:
        raise RuntimeError(
            f"Output directory already contains run artifacts: {out_dir}. "
            "Choose a new --out-dir; --no-resume does not delete or overwrite old runs."
        )

    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if artifacts and _resume_signature(previous) != _resume_signature(config):
            raise RuntimeError(
                f"Run configuration does not match the existing checkpoints in {out_dir}. "
                "Keep the original arguments to resume, or choose a new --out-dir for a new experiment."
            )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "Original 16x16 FullQuantumQCNN with train-only input augmentation",
        "architecture": FQ_ARCH,
        "only_model_change": "RandomAffine applied to training images before ToTensor and 16x16 resize",
        "augmentation": {
            "degrees": args.aug_degrees,
            "translate_fraction": args.aug_translate,
            "interpolation": "bilinear",
            "fill": 0,
            "validation_augmented": False,
        },
        "data": {
            "dataset": "MNIST official train split" if not args.fake_data else "FakeData smoke test",
            "train_size": 50_000 if not args.fake_data else "subset of 1,500",
            "validation_size": 10_000 if not args.fake_data else "subset of 500",
            "split_seed": args.split_seed,
            "official_test_used": False,
        },
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "phase1": {"epochs": args.phase1_epochs, "optimizer": "Adam", "lr": args.phase1_lr,
                       "select": "validation accuracy"},
            "phase2": {"epochs": args.phase2_epochs, "optimizer": "fresh Adam", "lr": args.phase2_lr,
                       "select": "validation accuracy"},
            "phase3": {"epochs_max": args.phase3_epochs, "optimizer": "Adam + ReduceLROnPlateau",
                       "core_lr": args.core_lr, "readout_lr": args.readout_lr,
                       "patience": args.phase3_patience, "select": "validation loss",
                       "reupload": args.fq_reupload},
        },
        "device": str(device),
        "args": vars(args),
    }
    validate_output_directory(out_dir, config, resume=not args.no_resume)
    write_json(out_dir / "config_augmentation.json", config)
    print(f"[run] output directory: {out_dir.resolve()}", flush=True)
    print("[run] model architecture and three-phase schedule match the original 16x16 FullQuantumQCNN.",
          flush=True)
    print(f"[run] train-only augmentation: +/-{args.aug_degrees:g} degrees, "
          f"translate={args.aug_translate:g}; validation is unaugmented.", flush=True)

    all_rows = []
    offset = 0
    best_state = None
    phase_settings = [
        ("pretrain", "fq_aug_phase1_pretrain", args.phase1_epochs, args.phase1_lr),
        ("refine", "fq_aug_phase2_refine", args.phase2_epochs, args.phase2_lr),
    ]
    for phase, tag, epochs, learning_rate in phase_settings:
        train, validation, generator, train_indices, validation_indices = make_phase_loaders(
            args, args.batch_size
        )
        if phase == "pretrain":
            save_split_indices(out_dir, train_indices, validation_indices)
        model = build_model(device, reupload=False)
        if best_state is not None:
            load_model_state(model, best_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        rows, best_state = run_phase(
            tag=tag,
            phase=phase,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            train=train,
            val=validation,
            generator=generator,
            epochs=epochs,
            select="val_acc",
            patience=None,
            out_dir=out_dir,
            device=device,
            epoch_offset=offset,
            resume=not args.no_resume,
        )
        all_rows.extend(rows)
        offset += len(rows)
        write_history(out_dir / "history_full_quantum.csv", all_rows)

    train, validation, generator, _, _ = make_phase_loaders(args, args.batch_size)
    model = build_model(device, reupload=args.fq_reupload)
    if best_state is not None:
        load_model_state(model, best_state, allow_new_reupload=args.fq_reupload)
    core_params, readout_params = [], []
    for name, parameter in model.named_parameters():
        (readout_params if name.startswith("readout.") or ".readout." in name else core_params).append(parameter)
    optimizer = torch.optim.Adam([
        {"params": core_params, "lr": args.core_lr},
        {"params": readout_params, "lr": args.readout_lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-7
    )
    rows, _ = run_phase(
        tag="fq_aug_phase3_reupload",
        phase="reupload",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train=train,
        val=validation,
        generator=generator,
        epochs=args.phase3_epochs,
        select="val_loss",
        patience=args.phase3_patience,
        out_dir=out_dir,
        device=device,
        epoch_offset=offset,
        resume=not args.no_resume,
    )
    all_rows.extend(rows)
    write_history(out_dir / "history_full_quantum.csv", all_rows)
    plot_all(out_dir, acc_ymin=args.acc_ymin)
    print(f"[run] completed; artifacts saved under {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
