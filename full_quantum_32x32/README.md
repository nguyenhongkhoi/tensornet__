# FullQuantumQCNN 32x32

This folder is a self-contained copy of the code needed for the 32x32 run. It
contains its own `models/`, `data/`, and `training_utils.py`; it does not import
Python code from the project root. Python packages (PyTorch, torchvision,
NumPy, and Matplotlib) and the MNIST dataset are still required.

From the project root on the AnyDesk machine, with its existing `venv` active:

```powershell
python .\full_quantum_32x32\run_32x32.py --data-root .\mnist_data --device cuda --out-dir .\full_quantum_32x32\runs\full_quantum_32x32 --batch-size 16 --phase1-epochs 20 --phase2-epochs 5 --phase3-epochs 60 --phase3-patience 10
```

The script starts from random weights, uses the fixed 50,000/10,000 train and
validation split, and center-pads each 28x28 image to 32x32. Checkpoints and
plots are saved under the specified output directory. Re-run the same command
and output path to resume an interrupted phase.

To use a dataset elsewhere, pass its path with `--data-root`. If MNIST is not
present there, torchvision downloads it.
