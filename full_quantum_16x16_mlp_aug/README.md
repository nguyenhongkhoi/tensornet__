# FullQuantum 16x16: original three phases + MLP readout + train augmentation

This isolated experiment keeps the original 16x16 FullQuantumQCNN training recipe and changes only two things: the Pauli readout is an MLP, and training images receive RandomAffine augmentation. It starts from random initialization; it does not load the old phase-3 checkpoint.

## What stays the same

- 16x16 grayscale input, 3 quantum channels, 8 final qubits, strongly entangling core, 4 layers, merge rounds 2, and reupload in phase 3.
- Same fixed MNIST train/validation split: 50,000 train and 10,000 validation examples, split seed 20260911. Official test data is not used.
- Same three-phase schedule and Adam settings as the original 16x16 run:
  - Phase 1: fresh model, Adam LR 2e-3, 20 epochs; carry forward the checkpoint with best validation accuracy.
  - Phase 2: continue from phase-1 best, new Adam LR 2e-4, 5 epochs; carry forward the checkpoint with best validation accuracy.
  - Phase 3: add reupload, continue from phase-2 best, Adam core LR 1e-4/readout LR 1e-3, ReduceLROnPlateau, up to 60 epochs, early stop after 10 epochs without validation-loss improvement; retain best validation-loss checkpoint.
- Batch size 64.

## The two experiment changes

1. Readout per branch: `Linear(276,256) -> LayerNorm(256) -> GELU -> Dropout(0.1) -> Linear(256,10)`.
2. Training only: `RandomAffine(degrees=8, translate=0.05)` before `ToTensor`. Validation remains unaugmented.

## Run on the AnyDesk computer

Copy this entire folder into the AnyDesk project root next to `runs/`, `mnist_data/`, and `venv/`. From any PowerShell location, run the venv Python by its full path (adjust the root path if the AnyDesk project is elsewhere):

```powershell
D:\tensornetwork_mnist\tensornetwork_mnist\venv\Scripts\python.exe D:\tensornetwork_mnist\tensornetwork_mnist\full_quantum_16x16_mlp_aug\run_mlp_aug.py --device cuda --out-dir D:\tensornetwork_mnist\tensornetwork_mnist\full_quantum_16x16_mlp_aug\runs\mlp_aug_3phase
```

The same folder is the default output directory. It must be empty or not exist when starting a run. The script saves separate best checkpoints and histories for each phase, combined history CSV, configuration JSON, split indices, summary, plot, and final best checkpoint under `runs/mlp_aug_3phase/`.

Each epoch log prints train loss/accuracy, validation loss/accuracy, both learning rates, elapsed seconds, and `*best`. Phase 3 can early-stop, just like the original run.

The AnyDesk venv should already contain compatible PyTorch and torchvision. `requirements.txt` lists only NumPy and Matplotlib; do not replace the existing CUDA-enabled PyTorch installation for this experiment.
cd D:\tensornetwork_mnist\tensornetwork_mnist
.\venv\Scripts\Activate.ps1
python .\full_quantum_16x16_mlp_aug\run_mlp_aug.py `  --device cuda`
--core-lr 1e-5 `  --readout-lr 1e-3`
--out-dir .\full_quantum_16x16_mlp_aug\runs\mlp_aug_finetune
