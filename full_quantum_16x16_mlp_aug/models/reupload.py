"""Re-upload variant of the tensor-network core (used by the best full-quantum setting).

After the conv/pool blocks, the mean RY angle of the input pixels that map to each of the
final qubits is re-applied as a learned-scale RY rotation (``reupload_scale``, 8 params per
branch, initialised to 0 so a warm-started checkpoint is reproduced exactly) before the
merge stage. Copied verbatim from Stage 6/9 (scripts/run_stage9_diagnose.py).
"""
from __future__ import annotations

import torch
from torch import nn

from .layers.tn_core import QuantumTNCore, _pure_to_density


class ReuploadQuantumTNCore(QuantumTNCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reupload_scale = nn.Parameter(torch.zeros(self.final_qubits))

    def forward(self, qubits):
        rho = _pure_to_density(qubits)
        for block in self.blocks:
            rho = block(rho)
        angles = 2 * torch.atan2(qubits[..., 1].real, qubits[..., 0].real)
        angles = angles.reshape(angles.shape[0], -1, self.final_qubits).mean(dim=1)
        angles = angles * self.reupload_scale[None, :]
        c = torch.cos(angles / 2)
        s = torch.sin(angles / 2)
        gate = torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2).to(rho.dtype)
        rho = gate @ rho @ gate.conj().transpose(-1, -2)
        return self.merge_stage(rho)


def add_reupload(model, *, input_sites, final_qubits, init_scale, merge_rounds, entangling_type, n_layers):
    """Swap every branch's tn_core of a FullQuantumQCNN for a ReuploadQuantumTNCore."""
    def make_core():
        return ReuploadQuantumTNCore(
            input_sites=input_sites, final_qubits=final_qubits, init_scale=init_scale,
            merge_rounds=merge_rounds, entangling_type=entangling_type, n_layers=n_layers,
        )

    model.tn_core = make_core()
    for branch in model.extra_branches:
        branch["tn_core"] = make_core()
    return model
