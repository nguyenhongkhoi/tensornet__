# Paper-inspired QCNN simulated with tensor-network state contraction

This is a standalone pilot folder. It does not modify the project's existing
`models/`, `runs/full_quantum/`, `full_quantum_32x32/`, or checkpoints.

## What this experiment runs

- An 8-qubit, 10-class QCNN based on the circuit structure in *Multi-Class
  Quantum Convolutional Neural Networks*: PCA to 256 values, amplitude
  encoding, F1/CC14, shared SU(4) convolution blocks, pooling, and measurement
  on wires `[0, 2, 4, 6]`.
- The state is an exact rank-8 PyTorch tensor. Each 1-qubit/2-qubit gate is
  contracted against its wire axes; this is a classical tensor-state simulator,
  not execution on quantum hardware.
- No CNN frontend is used. PCA is classical preprocessing. The reference
  circuit has 16 readout outcomes; digit predictions use outcomes 0 through 9.
- Exactly 5,000 or 10,000 randomly selected examples are used for gradient
  updates. The 10,000 validation examples come from a separate held-out portion
  of the official MNIST training set. PCA is fit only on the sampled train set.
  The official MNIST test set is not used.
- The setup is inspired by the article and its linked code; it is a PyTorch
  tensor-contraction port, not a verified bit-for-bit PennyLane reproduction.
  The article's claim is relative to its own small CNN baseline and is not a
  promise of 99% accuracy.

## Run on the AnyDesk machine

Copy the whole `paper_qcnn_tn_trial` folder into the existing project root on
the AnyDesk computer (the root containing `run_convergence.py`, `models/`,
`mnist_data/`, and `venv/`). In PowerShell, from that project root:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r .\paper_qcnn_tn_trial\requirements.txt
python .\paper_qcnn_tn_trial\run_trial.py `
  --data-root .\mnist_data `
  --train-samples 5000 `
  --epochs 10 `
  --batch-size 64 `
  --lr 0.01 `
  --device cuda `
  --out-dir .\runs\paper_qcnn_tn_trial_5k
```

The requirements file installs only the extra PCA/plotting packages and avoids
replacing the AnyDesk environment's existing PyTorch/CUDA build. The project
environment must already have compatible `torch` and `torchvision`, as the
existing training scripts do.

For the 10,000-example pilot, change `--train-samples 5000` to `10000` and
change output directory to `runs\paper_qcnn_tn_trial_10k` so results stay
separate. The script rejects train counts outside 5,000–10,000. If the machine
has no CUDA-capable PyTorch, replace `--device cuda` with `--device cpu`.

Outputs are saved only under the specified new run directory:

- `paper_qcnn_tn_best.pt`, `paper_qcnn_tn_last.pt`
- `history.csv`, `training_curves.png`, `summary.json`, `config.json`
- `sampled_train_indices.csv` (the random subset and true labels)

The split seed is `20260911`, matching the project's existing MNIST
train/validation split. `--sample-seed 43` controls the random subset. Change
the sample seed to run another random subset while keeping validation fixed.

## References

- Paper: https://arxiv.org/abs/2404.12741
- Authors' code: https://github.com/qslab-unipr/qcnn
