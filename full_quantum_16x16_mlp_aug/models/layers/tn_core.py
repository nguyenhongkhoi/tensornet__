from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn

from .unitary import LearnedUnitary1Q, LearnedUnitary2Q


def _pure_to_density(qubits: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bni,bnj->bnij", qubits, qubits.conj())


def _pair_density(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    pair = torch.einsum("bij,bkl->bikjl", left, right)
    return pair.reshape(left.shape[0], 4, 4)


def _apply_unitary_density(rho: torch.Tensor, unitary: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ai,bij,kj->bak", unitary, rho, unitary.conj())


def _trace_second_qubit(pair_rho: torch.Tensor) -> torch.Tensor:
    pair = pair_rho.reshape(pair_rho.shape[0], 2, 2, 2, 2)
    return torch.einsum("bijkj->bik", pair)


def _product_density(rho: torch.Tensor) -> torch.Tensor:
    """Build the joint density matrix for independent one-qubit states."""
    if rho.ndim != 4 or rho.shape[-2:] != (2, 2):
        raise ValueError("Expected per-qubit density matrices with shape (batch, sites, 2, 2)")

    joint = rho[:, 0]
    for site in range(1, rho.shape[1]):
        joint = torch.einsum("bij,bkl->bikjl", joint, rho[:, site])
        joint = joint.reshape(rho.shape[0], joint.shape[1] * 2, joint.shape[3] * 2)
    return joint


def _apply_gate_to_axes(tensor: torch.Tensor, gate: torch.Tensor, axes: List[int], right_action: bool) -> torch.Tensor:
    total_axes = list(range(tensor.ndim))
    rest_axes = [axis for axis in total_axes if axis not in axes]
    permuted = tensor.permute(*rest_axes, *axes)
    rest_shape = permuted.shape[:-2]
    flat = permuted.reshape(-1, 4)
    matrix = gate.conj().T if right_action else gate.T
    updated = flat @ matrix
    updated = updated.reshape(*rest_shape, 2, 2)

    inverse = [0] * len(total_axes)
    for new_axis, old_axis in enumerate([*rest_axes, *axes]):
        inverse[old_axis] = new_axis
    return updated.permute(*inverse)


def apply_two_qubit_gate(
    rho: torch.Tensor,
    gate: torch.Tensor,
    qubit_i: int,
    qubit_j: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a 2-qubit unitary to selected qubits of an n-qubit density matrix."""
    if qubit_i == qubit_j:
        raise ValueError("qubit_i and qubit_j must be different")
    if not (0 <= qubit_i < n_qubits and 0 <= qubit_j < n_qubits):
        raise ValueError("Target qubits are out of range")
    dim = 2**n_qubits
    if rho.ndim != 3 or rho.shape[-2:] != (dim, dim):
        raise ValueError(f"Expected rho with shape (batch, {dim}, {dim})")
    if gate.shape != (4, 4):
        raise ValueError("Expected a 2-qubit gate with shape (4, 4)")

    tensor = rho.reshape(rho.shape[0], *([2] * n_qubits), *([2] * n_qubits))
    tensor = _apply_gate_to_axes(tensor, gate, [1 + qubit_i, 1 + qubit_j], right_action=False)
    tensor = _apply_gate_to_axes(tensor, gate, [1 + n_qubits + qubit_i, 1 + n_qubits + qubit_j], right_action=True)
    return tensor.reshape(rho.shape[0], dim, dim)


def _apply_one_qubit_gate(rho: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    if not (0 <= qubit < n_qubits):
        raise ValueError("Target qubit is out of range")
    dim = 2**n_qubits
    if rho.ndim != 3 or rho.shape[-2:] != (dim, dim):
        raise ValueError(f"Expected rho with shape (batch, {dim}, {dim})")
    if gate.shape != (2, 2):
        raise ValueError("Expected a 1-qubit gate with shape (2, 2)")

    tensor = rho.reshape(rho.shape[0], *([2] * n_qubits), *([2] * n_qubits))
    tensor = _apply_single_gate_to_axis(tensor, gate, 1 + qubit, right_action=False)
    tensor = _apply_single_gate_to_axis(tensor, gate, 1 + n_qubits + qubit, right_action=True)
    return tensor.reshape(rho.shape[0], dim, dim)


def _apply_single_gate_to_axis(tensor: torch.Tensor, gate: torch.Tensor, axis: int, right_action: bool) -> torch.Tensor:
    total_axes = list(range(tensor.ndim))
    rest_axes = [candidate for candidate in total_axes if candidate != axis]
    permuted = tensor.permute(*rest_axes, axis)
    rest_shape = permuted.shape[:-1]
    flat = permuted.reshape(-1, 2)
    matrix = gate.conj().T if right_action else gate.T
    updated = flat @ matrix
    updated = updated.reshape(*rest_shape, 2)

    inverse = [0] * len(total_axes)
    for new_axis, old_axis in enumerate([*rest_axes, axis]):
        inverse[old_axis] = new_axis
    return updated.permute(*inverse)


def _renormalize_hermitize(joint: torch.Tensor) -> torch.Tensor:
    trace = joint.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real.clamp_min(1e-8)
    joint = joint / trace[:, None, None]
    return 0.5 * (joint + joint.conj().transpose(-1, -2))


class QuantumConvPoolBlock(nn.Module):
    def __init__(self, real_only: bool = False, init_scale: float = 1e-2):
        super().__init__()
        self.conv = LearnedUnitary2Q(init_scale=init_scale, real_only=real_only)
        self.pool = LearnedUnitary2Q(init_scale=init_scale, real_only=real_only)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        batch, sites, _, _ = rho.shape
        if sites % 2 != 0:
            raise ValueError(f"QuantumTNCore requires an even site count before pooling, got {sites}")

        left = rho[:, 0::2]
        right = rho[:, 1::2]
        pairs = _pair_density(left.reshape(-1, 2, 2), right.reshape(-1, 2, 2))
        pairs = _apply_unitary_density(pairs, self.conv())
        pairs = _apply_unitary_density(pairs, self.pool())
        survivors = _trace_second_qubit(pairs).reshape(batch, sites // 2, 2, 2)

        trace = survivors.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real.clamp_min(1e-8)
        survivors = survivors / trace[..., None, None]
        return 0.5 * (survivors + survivors.conj().transpose(-1, -2))


class QuantumMergeStage(nn.Module):
    """Entangle final independent qubits without tracing any qubit out."""

    def __init__(
        self,
        final_qubits: int = 4,
        merge_rounds: int = 2,
        real_only: bool = False,
        init_scale: float = 1e-2,
    ):
        super().__init__()
        if final_qubits < 1:
            raise ValueError("final_qubits must be positive")
        if merge_rounds < 0:
            raise ValueError("merge_rounds must be non-negative")

        self.final_qubits = final_qubits
        self.merge_rounds = merge_rounds
        self.pairs: List[List[Tuple[int, int]]] = []
        self.rounds = nn.ModuleList()
        for round_idx in range(merge_rounds):
            start = round_idx % 2
            pairs = [(i, i + 1) for i in range(start, final_qubits - 1, 2)]
            self.pairs.append(pairs)
            self.rounds.append(
                nn.ModuleList(
                    [LearnedUnitary2Q(init_scale=init_scale, real_only=real_only) for _ in pairs]
                )
            )

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        joint = _product_density(rho)
        for round_pairs, round_gates in zip(self.pairs, self.rounds):
            for (qubit_i, qubit_j), gate in zip(round_pairs, round_gates):
                joint = apply_two_qubit_gate(joint, gate(), qubit_i, qubit_j, self.final_qubits)

        trace = joint.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real.clamp_min(1e-8)
        joint = joint / trace[:, None, None]
        return 0.5 * (joint + joint.conj().transpose(-1, -2))

    def estimate_cost(self) -> Dict[str, int]:
        merge_gates = sum(len(pairs) for pairs in self.pairs)
        return {"merge_two_qubit_gates": merge_gates}


class _IdentityJointStage(nn.Module):
    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return _product_density(rho)

    def estimate_cost(self) -> Dict[str, int]:
        return {}


class StronglyEntanglingCore(nn.Module):
    """PennyLane-style StronglyEntanglingLayers over final qubits."""

    def __init__(
        self,
        final_qubits: int = 4,
        n_layers: int | None = None,
        real_only: bool = False,
        init_scale: float = 1e-2,
    ):
        super().__init__()
        if final_qubits < 1:
            raise ValueError("final_qubits must be positive")
        if n_layers is None:
            n_layers = max(final_qubits - 1, 0)
        if n_layers < 0:
            raise ValueError("n_layers must be non-negative")

        self.final_qubits = final_qubits
        self.n_layers = n_layers
        self.rotations = nn.ModuleList()
        self.entanglers = nn.ModuleList()
        for _ in range(n_layers):
            self.rotations.append(
                nn.ModuleList(
                    [LearnedUnitary1Q(init_scale=init_scale, real_only=real_only) for _ in range(final_qubits)]
                )
            )
            self.entanglers.append(
                nn.ModuleList(
                    [LearnedUnitary2Q(init_scale=init_scale, real_only=real_only) for _ in range(final_qubits)]
                )
            )

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        joint = _product_density(rho)
        if self.final_qubits == 1 or self.n_layers == 0:
            return _renormalize_hermitize(joint)

        for layer_idx, (layer_rotations, layer_entanglers) in enumerate(zip(self.rotations, self.entanglers)):
            for qubit, gate in enumerate(layer_rotations):
                joint = _apply_one_qubit_gate(joint, gate(), qubit, self.final_qubits)
                joint = _renormalize_hermitize(joint)

            if self.final_qubits == 2:
                layer_range = 1
            else:
                layer_range = (layer_idx % (self.final_qubits - 1)) + 1
            for qubit_i, gate in enumerate(layer_entanglers):
                qubit_j = (qubit_i + layer_range) % self.final_qubits
                joint = apply_two_qubit_gate(joint, gate(), qubit_i, qubit_j, self.final_qubits)
                joint = _renormalize_hermitize(joint)

        return joint

    def estimate_cost(self) -> Dict[str, int]:
        if self.final_qubits == 1:
            one_qubit_gates = 0
            two_qubit_gates = 0
        else:
            one_qubit_gates = self.n_layers * self.final_qubits
            two_qubit_gates = self.n_layers * self.final_qubits
        return {
            "entangling_one_qubit_gates": one_qubit_gates,
            "entangling_two_qubit_gates": two_qubit_gates,
        }


class QuantumTNCore(nn.Module):
    """Tree/QCNN-style tensor-network core shared by hybrid and full-Q arms."""

    def __init__(
        self,
        input_sites: int,
        final_qubits: int = 4,
        real_only: bool = False,
        init_scale: float = 1e-2,
        merge_rounds: int = 2,
        entangling_type: str = "strongly_entangling",
        n_layers: int | None = None,
    ):
        super().__init__()
        if input_sites < final_qubits:
            raise ValueError("input_sites must be >= final_qubits")
        ratio = input_sites / final_qubits
        if ratio < 1 or int(ratio) != ratio or int(ratio) & (int(ratio) - 1):
            raise ValueError("input_sites/final_qubits must be a power of two")

        self.input_sites = input_sites
        self.final_qubits = final_qubits
        self.blocks = nn.ModuleList(
            [QuantumConvPoolBlock(real_only=real_only, init_scale=init_scale) for _ in range(int(math.log2(ratio)))]
        )
        self.entangling_type = entangling_type
        if entangling_type == "none":
            self.merge_stage = _IdentityJointStage()
        elif entangling_type == "brickwall":
            self.merge_stage = QuantumMergeStage(
                final_qubits=final_qubits,
                merge_rounds=merge_rounds,
                real_only=real_only,
                init_scale=init_scale,
            )
        elif entangling_type == "strongly_entangling":
            self.merge_stage = StronglyEntanglingCore(
                final_qubits=final_qubits,
                n_layers=n_layers,
                real_only=real_only,
                init_scale=init_scale,
            )
        else:
            raise ValueError(f"Unsupported entangling_type: {entangling_type}")

    def forward(self, qubits: torch.Tensor) -> torch.Tensor:
        if qubits.ndim != 3 or qubits.shape[-1] != 2:
            raise ValueError("Expected encoded qubits with shape (batch, sites, 2)")
        if qubits.shape[1] != self.input_sites:
            raise ValueError(f"Expected {self.input_sites} sites, got {qubits.shape[1]}")

        rho = _pure_to_density(qubits)
        for block in self.blocks:
            rho = block(rho)
        return self.merge_stage(rho)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_cost(self) -> Dict[str, int]:
        sites = self.input_sites
        gates = 0
        pair_contractions = 0
        for _ in self.blocks:
            pairs = sites // 2
            gates += 2 * pairs
            pair_contractions += pairs
            sites //= 2
        merge_cost = self.merge_stage.estimate_cost()
        gates += merge_cost.get("merge_two_qubit_gates", 0)
        gates += merge_cost.get("entangling_two_qubit_gates", 0)
        return {
            "two_qubit_gates": gates,
            "pair_contractions": pair_contractions,
            **merge_cost,
        }
