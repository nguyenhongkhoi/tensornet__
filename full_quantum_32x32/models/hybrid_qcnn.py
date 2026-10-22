from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from data.encoding import angle_ry_encode
from .layers.readout import JointBasisProbabilityReadout, PauliObservableReadout
from .layers.tn_core import QuantumTNCore


class HybridQCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        feature_size: Sequence[int] = (8, 8),
        quantum_channels: int = 1,
        final_qubits: int = 4,
        readout_type: str = "basis_prob",
        real_only: bool = False,
        init_scale: float = 1e-2,
        merge_rounds: int = 2,
        entangling_type: str = "strongly_entangling",
        n_layers: int | None = None,
        pair_correlators: str = "zz",
    ):
        super().__init__()
        self.feature_size = tuple(feature_size)
        self.quantum_channels = quantum_channels
        self.frontend = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.feature_size),
            nn.Conv2d(16, quantum_channels, 1),
            nn.Sigmoid(),
        )
        input_sites = self.feature_size[0] * self.feature_size[1] * quantum_channels
        self.tn_core = QuantumTNCore(
            input_sites=input_sites,
            final_qubits=final_qubits,
            real_only=real_only,
            init_scale=init_scale,
            merge_rounds=merge_rounds,
            entangling_type=entangling_type,
            n_layers=n_layers,
        )
        self.readout = _make_readout(readout_type, num_classes, final_qubits, pair_correlators)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.frontend(images).flatten(start_dim=1)
        qubits = angle_ry_encode(features).to(images.device)
        rho = self.tn_core(qubits)
        return self.readout(rho)

    def count_parameters(self) -> Dict[str, int]:
        frontend = sum(p.numel() for p in self.frontend.parameters() if p.requires_grad)
        quantum = self.tn_core.count_parameters()
        readout = sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        return {
            "classical_frontend": frontend,
            "quantum_core": quantum,
            "classical_readout": readout,
            "total": frontend + quantum + readout,
        }

    def estimate_cost(self) -> Dict[str, int]:
        return self.tn_core.estimate_cost()


def _make_readout(readout_type: str, num_classes: int, final_qubits: int, pair_correlators: str = "zz") -> nn.Module:
    if readout_type == "basis_prob":
        return JointBasisProbabilityReadout(num_classes=num_classes, final_qubits=final_qubits)
    if readout_type == "pauli_observable":
        return PauliObservableReadout(num_classes=num_classes, final_qubits=final_qubits, pair_correlators=pair_correlators)
    raise ValueError(f"Unsupported readout_type: {readout_type}")
