from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from data.encoding import angle_ry_encode
from data.preprocessing import preprocess_full_quantum
from .hybrid_qcnn import _make_readout
from .layers.tn_core import QuantumTNCore


def spatial_permutation(height: int, width: int, channels: int, order: str) -> torch.Tensor:
    """Reorder sites without changing pixels; Morton groups dyadic 2D patches."""
    if order not in {"row", "column", "morton"}:
        raise ValueError(f"Unknown spatial order: {order}")
    sites = list(range(height * width))
    if order == "column":
        sites.sort(key=lambda i: (i % width, i // width))
    elif order == "morton":
        def code(i):
            y, x = divmod(i, width)
            return sum(((x >> b) & 1) << (2 * b) |
                       ((y >> b) & 1) << (2 * b + 1)
                       for b in range(max(height, width).bit_length()))
        sites.sort(key=code)
    return torch.tensor([c * height * width + i for c in range(channels) for i in sites])


class FullQuantumQCNN(nn.Module):
    """quantum_channels > 1: chay K nhanh (tn_core, readout) DOC LAP song song tren CUNG 1 dau
    vao qubit (moi nhanh tu khoi tao rieng, hoc rieng), roi trung binh cong LOGITS cua ca K nhanh
    truoc softmax. Day la don bay tang capacity KHONG lam final_qubits/dim ma tran mat do to hon
    (moi nhanh van final_qubits nho, tranh "vach tuong" tensor-rank da phat hien voi final_qubits=9
    - xem PLAN.md muc 22-23), chi phi TUYEN TINH theo K (K lan forward doc lap), khong phai mu.
    Tuong tu 'nhieu dau/mixture-of-experts' duoc train CUNG LUC qua 1 ham loss chung, KHAC voi
    ensemble hau-ky (train K seed rieng roi trung binh sau) - o day cac nhanh co the hoc CHUYEN
    MON HOA/BO SUNG cho nhau ngay trong qua trinh train (xem PLAN.md muc 27).

    quantum_channels=1 (mac dinh, tuong thich nguoc HOAN TOAN): giu nguyen ten thuoc tinh
    `tn_core`/`readout` nhu truoc day, nen checkpoint cu (train truoc khi co tham so nay) van
    load duoc binh thuong khong can sua gi.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        target_size: Sequence[int] = (8, 8),
        grayscale: bool = False,
        final_qubits: int = 4,
        readout_type: str = "basis_prob",
        real_only: bool = False,
        init_scale: float = 1e-2,
        merge_rounds: int = 2,
        entangling_type: str = "strongly_entangling",
        n_layers: int | None = None,
        pair_correlators: str = "zz",
        quantum_channels: int = 1,
        spatial_orders: Sequence[str] | None = None,
        preprocess_mode: str = "resize",
    ):
        super().__init__()
        self.target_size = tuple(target_size)
        self.grayscale = grayscale or in_channels == 1
        if preprocess_mode not in {"resize", "pad"}:
            raise ValueError("preprocess_mode must be 'resize' or 'pad'")
        self.preprocess_mode = preprocess_mode
        effective_channels = 1 if self.grayscale else in_channels
        input_sites = self.target_size[0] * self.target_size[1] * effective_channels
        self.quantum_channels = max(1, quantum_channels)
        self.spatial_orders = tuple(spatial_orders or ["row"] * self.quantum_channels)
        if len(self.spatial_orders) != self.quantum_channels:
            raise ValueError("Provide one spatial order per quantum branch")
        self.register_buffer("site_permutations", torch.stack([
            spatial_permutation(*self.target_size, effective_channels, order)
            for order in self.spatial_orders
        ]), persistent=False)

        def _make_branch():
            tn_core = QuantumTNCore(
                input_sites=input_sites,
                final_qubits=final_qubits,
                real_only=real_only,
                init_scale=init_scale,
                merge_rounds=merge_rounds,
                entangling_type=entangling_type,
                n_layers=n_layers,
            )
            readout = _make_readout(readout_type, num_classes, final_qubits, pair_correlators)
            return tn_core, readout

        # Nhanh dau tien luon giu ten thuoc tinh cu (tn_core/readout) de tuong thich checkpoint.
        self.tn_core, self.readout = _make_branch()
        self.extra_branches = nn.ModuleList(
            [
                nn.ModuleDict({"tn_core": tn_core, "readout": readout})
                for _ in range(self.quantum_channels - 1)
                for tn_core, readout in [_make_branch()]
            ]
        )

    def _all_branches(self):
        yield self.tn_core, self.readout
        for branch in self.extra_branches:
            yield branch["tn_core"], branch["readout"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = preprocess_full_quantum(images, self.target_size, grayscale=self.grayscale,
                                         mode=self.preprocess_mode)
        qubits = angle_ry_encode(values).to(images.device)
        logits_sum = None
        for index, (tn_core, readout) in enumerate(self._all_branches()):
            rho = tn_core(qubits.index_select(1, self.site_permutations[index]))
            logits = readout(rho)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        return logits_sum / self.quantum_channels

    def count_parameters(self) -> Dict[str, int]:
        quantum = 0
        readout_params = 0
        for tn_core, readout in self._all_branches():
            quantum += tn_core.count_parameters()
            readout_params += sum(p.numel() for p in readout.parameters() if p.requires_grad)
        return {"quantum_core": quantum, "classical_readout": readout_params, "total": quantum + readout_params}

    def estimate_cost(self) -> Dict[str, int]:
        combined: Dict[str, int] = {}
        for tn_core, _ in self._all_branches():
            for key, value in tn_core.estimate_cost().items():
                combined[key] = combined.get(key, 0) + value
        return combined
