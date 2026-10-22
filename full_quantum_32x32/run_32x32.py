"""Train a fresh 32x32 padded FullQuantumQCNN with the original three-phase recipe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# This folder contains its own model/data packages and training helpers so it
# can be copied to another checkout without importing the parent project's code.
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from models import FullQuantumQCNN  # noqa: E402
from models.reupload import add_reupload  # noqa: E402
from training_utils import (make_loaders, plot_all, run_phase, seed_all,
                            write_history)  # noqa: E402


FQ_ARCH_32 = dict(
    target_size=(32, 32), grayscale=True, final_qubits=8,
    readout_type="pauli_observable", pair_correlators="full9",
    init_scale=0.01, merge_rounds=2,
    entangling_type="strongly_entangling", n_layers=4,
    quantum_channels=3, preprocess_mode="pad",
)


def build_model(device: torch.device, reupload: bool):
    model = FullQuantumQCNN(in_channels=1, num_classes=10, **FQ_ARCH_32)
    if reupload:
        add_reupload(model, input_sites=32 * 32, final_qubits=8, init_scale=0.01,
                     merge_rounds=2, entangling_type="strongly_entangling", n_layers=4)
    return model.to(device)


def load_state(model, state: dict, allow_reupload_init: bool = False) -> None:
    current = model.state_dict()
    missing, unexpected = set(current) - set(state), set(state) - set(current)
    allowed_missing = {key for key in missing if key.endswith(".reupload_scale")} if allow_reupload_init else set()
    if missing != allowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch. missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.load_state_dict(state, strict=False)


def phase_loaders(args, batch_size: int, phase_seed: int):
    # Reuse the project's deterministic 50k/10k split and ToTensor pipeline.
    loader_args = argparse.Namespace(
        fake_data=False, data_root=args.data_root, split_seed=args.split_seed,
        train_subset=None, val_subset=None, device=args.device,
    )
    return make_loaders(loader_args, batch_size, phase_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "mnist_data"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "full_quantum_32x32"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase3-epochs", type=int, default=60)
    parser.add_argument("--phase3-patience", type=int, default=10)
    parser.add_argument("--acc-ymin", type=float, default=90.0)
    parser.add_argument("--no-resume", action="store_true", help="Ignore per-phase last checkpoints")
    args = parser.parse_args()
    if min(args.batch_size, args.phase1_epochs, args.phase2_epochs,
           args.phase3_epochs, args.phase3_patience) < 1:
        parser.error("batch size, phase epoch counts, and patience must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is False")

    seed_all(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "FullQuantumQCNN",
        "architecture": FQ_ARCH_32,
        "initialization": "random; no pretrained checkpoint",
        "preprocessing": "center-pad 28x28 MNIST to 32x32 with zero pixels; no resize/downsampling",
        "dataset": "MNIST official train split: 50,000 train / 10,000 validation",
        "split_seed": args.split_seed,
        "test_set_used": False,
        "args": vars(args),
        "phases": [
            {"name": "pretrain", "epochs": args.phase1_epochs, "optimizer": "Adam", "lr": 2e-3,
             "checkpoint_selection": "best validation accuracy"},
            {"name": "refine", "epochs": args.phase2_epochs, "optimizer": "fresh Adam", "lr": 2e-4,
             "checkpoint_selection": "best validation accuracy"},
            {"name": "reupload", "epochs": args.phase3_epochs,
             "optimizer": "Adam core=1e-4/readout=1e-3 + ReduceLROnPlateau",
             "early_stop": f"validation loss patience {args.phase3_patience}",
             "checkpoint_selection": "best validation loss"},
        ],
    }
    (out_dir / "config_32x32.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"32x32 model: padded 28x28 images; {FQ_ARCH_32['target_size'][0] ** 2} sites per branch.", flush=True)
    print(f"Training all 50,000 train samples; validating on all 10,000 validation samples.", flush=True)
    print(f"Output: {out_dir.resolve()} | resume={not args.no_resume}", flush=True)

    all_rows = []
    offset = 0
    # Phase 1: a genuinely new model, initialized from random weights.
    model = build_model(device, reupload=False)
    train, val, generator = phase_loaders(args, args.batch_size, args.seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    phase1_rows, phase1_state = run_phase(
        tag="fq32_phase1_pretrain", phase="pretrain", model=model, optimizer=optimizer,
        scheduler=None, train=train, val=val, generator=generator, epochs=args.phase1_epochs,
        select="val_acc", patience=None, out_dir=out_dir, device=device,
        epoch_offset=offset, resume=not args.no_resume,
    )
    all_rows.extend(phase1_rows)
    write_history(out_dir / "history_full_quantum.csv", all_rows)
    offset += len(phase1_rows)

    # Phase 2: continue from the best phase-1 model with a fresh Adam optimizer.
    model = build_model(device, reupload=False)
    load_state(model, phase1_state)
    train, val, generator = phase_loaders(args, args.batch_size, args.seed + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    phase2_rows, phase2_state = run_phase(
        tag="fq32_phase2_refine", phase="refine", model=model, optimizer=optimizer,
        scheduler=None, train=train, val=val, generator=generator, epochs=args.phase2_epochs,
        select="val_acc", patience=None, out_dir=out_dir, device=device,
        epoch_offset=offset, resume=not args.no_resume,
    )
    all_rows.extend(phase2_rows)
    write_history(out_dir / "history_full_quantum.csv", all_rows)
    offset += len(phase2_rows)

    # Phase 3: add zero-initialized re-upload scales and preserve the phase-2 weights.
    model = build_model(device, reupload=True)
    load_state(model, phase2_state, allow_reupload_init=True)
    train, val, generator = phase_loaders(args, args.batch_size, args.seed + 3)
    core, readout = [], []
    for name, parameter in model.named_parameters():
        (readout if name.startswith("readout.") or ".readout." in name else core).append(parameter)
    optimizer = torch.optim.Adam([{"params": core, "lr": 1e-4}, {"params": readout, "lr": 1e-3}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-7)
    phase3_rows, _ = run_phase(
        tag="fq32_phase3_reupload", phase="reupload", model=model, optimizer=optimizer,
        scheduler=scheduler, train=train, val=val, generator=generator, epochs=args.phase3_epochs,
        select="val_loss", patience=args.phase3_patience, out_dir=out_dir, device=device,
        epoch_offset=offset, resume=not args.no_resume,
    )
    all_rows.extend(phase3_rows)
    write_history(out_dir / "history_full_quantum.csv", all_rows)
    plot_all(out_dir, acc_ymin=args.acc_ymin)


if __name__ == "__main__":
    main()
