# QCNN_onlyAgurment

Standalone copy of the original 16x16 FullQuantumQCNN project. The only model
input/training change is a mild `RandomAffine` applied to training images before
the existing grayscale/resize/RY-encoding pipeline. The original Pauli full9
Linear readout, three quantum branches, tensor-network code, and three-phase
training recipe are retained. Validation is never augmented.

## Augmentation choice

Defaults are rotation ±5 degrees and translation up to 4% per axis (about one
pixel for a 28x28 MNIST image), with bilinear interpolation and zero fill. This
is intentionally milder than the ±8-degree / 5% proposal: use the mild setting
for the first run, then try the stronger setting only as a separate experiment.
No MLP readout or other breakthrough proposal is included in this folder.

## Run on Windows PowerShell

From the repository root, with the existing environment and MNIST data:

```powershell
.\venv\Scripts\python.exe .\QCNN_onlyAgurment\run_augmentation.py `
  --device cuda `
  --seed 0 --split-seed 20260911 `
  --fq-batch-size 64 `
  --fq-pretrain-epochs 20 --fq-pretrain-lr 2e-3 `
  --fq-refine-epochs 5 --fq-refine-lr 2e-4 `
  --fq-epochs 60 --fq-core-lr 1e-4 --fq-readout-lr 1e-3 `
  --fq-patience 10 --fq-reupload `
  --data-root .\mnist_data `
  --aug-degrees 5 --aug-translate 0.04 `
  --out-dir .\QCNN_onlyAgurment\runs\augmentation_seed0
```

The original schedule is used: phase 1 is 20 epochs at LR 2e-3, phase 2 is 5
epochs at LR 2e-4, and phase 3 runs up to 60 epochs with core/readout LRs
1e-4/1e-3 and patience 10. The default seed is 0 and split seed is 20260911.
Rerun with the same training configuration and output path to resume. The
runner checks the saved configuration before loading checkpoints and stops if
training settings differ. Last checkpoints retain the data-loader and PyTorch
RNG states so stochastic augmentation continues from the saved epoch. Use a
new output path for another seed or augmentation setting. `--no-resume` does
not erase an existing run; it requires an output directory without run artifacts.

For a quick CPU wiring check (not an accuracy experiment):

```powershell
.\venv\Scripts\python.exe .\QCNN_onlyAgurment\run_augmentation.py `
  --smoke --device cpu `
  --out-dir .\QCNN_onlyAgurment\runs\smoke
```

To try the stronger proposal in a separate run:

```powershell
.\venv\Scripts\python.exe .\QCNN_onlyAgurment\run_augmentation.py `
  --device cuda --fq-reupload `
  --fq-batch-size 64 `
  --fq-pretrain-epochs 20 --fq-pretrain-lr 2e-3 `
  --fq-refine-epochs 5 --fq-refine-lr 2e-4 `
  --fq-epochs 60 --fq-core-lr 1e-4 --fq-readout-lr 1e-3 `
  --fq-patience 10 `
  --aug-degrees 8 --aug-translate 0.05 `
  --out-dir .\QCNN_onlyAgurment\runs\augmentation_8deg_5pct
```

Each run stores `config_augmentation.json`, split indices, `history_full_quantum.csv`,
per-phase checkpoints and completion files, `summary.json`, and convergence
plots under its output directory. MNIST official test data is not used.

## Assessment of `forAI/breakthrough_proposals.txt`

- **Spatial diversity `[row, column, morton]`:** best low-risk architectural
  follow-up; the model supports these orders. Keep it out of this augmentation
  run so the effect of augmentation remains measurable.
- **Intermediate entangling at 64 sites:** do not implement as described.
  Keeping a joint density matrix for 64 qubits is intractable, and the proposed
  parameter estimate ignores the per-gate matrix parameters. A scalable
  tensor-network/bond representation is needed first.
- **Multi-scale re-upload:** plausible but needs a precise mapping/shared-weight
  design and a separate ablation; not a drop-in input augmentation change.
- **Residual classical memory:** potentially useful, but makes the model hybrid
  and adds a new learned branch; do not bundle it with the current experiment.
- **Learned positional angles:** feasible as a separate input-encoding ablation,
  but should be initialized to reproduce the current `theta = pi * pixel`
  mapping and tested independently.
- **32x32, more final qubits, optimizer swaps, or MLP:** do not add to this
  folder's run. They change other variables and would confound this comparison.

The first run should change augmentation only. Decide whether to add any
proposal only after comparing validation accuracy and loss with the original
baseline on the same split; repeat with additional seeds before treating a small
difference as a real improvement.
