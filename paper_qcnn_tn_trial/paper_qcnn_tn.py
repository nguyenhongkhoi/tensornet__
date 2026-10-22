"""8-qubit multi-class QCNN from the paper, contracted as a PyTorch tensor state."""
from __future__ import annotations

import torch
from torch import nn


def _phase(angle: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.complex(torch.zeros_like(angle), angle))


def _ry(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(torch.complex64)


def _rx(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle / 2).to(torch.complex64), -1j * torch.sin(angle / 2)
    return torch.stack((torch.stack((c, s)), torch.stack((s, c))))


def _rz(angle: torch.Tensor) -> torch.Tensor:
    phases = torch.stack((_phase(-angle / 2), _phase(angle / 2)))
    return torch.diag(phases)


def _crx(angle: torch.Tensor) -> torch.Tensor:
    rx = _rx(angle)
    z = torch.zeros((), dtype=torch.complex64, device=angle.device)
    o = torch.ones((), dtype=torch.complex64, device=angle.device)
    return torch.stack((torch.stack((o, z, z, z)), torch.stack((z, o, z, z)),
                        torch.stack((z, z, rx[0, 0], rx[0, 1])),
                        torch.stack((z, z, rx[1, 0], rx[1, 1]))))


def _crz(angle: torch.Tensor) -> torch.Tensor:
    rz = _rz(angle)
    z = torch.zeros((), dtype=torch.complex64, device=angle.device)
    o = torch.ones((), dtype=torch.complex64, device=angle.device)
    return torch.stack((torch.stack((o, z, z, z)), torch.stack((z, o, z, z)),
                        torch.stack((z, z, rz[0, 0], rz[0, 1])),
                        torch.stack((z, z, rz[1, 0], rz[1, 1]))))


def _u3(theta: torch.Tensor, phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack((torch.stack((c.to(torch.complex64), -_phase(lam) * s)),
                        torch.stack((_phase(phi) * s, _phase(phi + lam) * c))))


def _cnot(device: torch.device) -> torch.Tensor:
    return torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=torch.complex64, device=device)


def _x(device: torch.device) -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)


def _apply_1q(state: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply a gate to one physical wire. State axes are ordered wire 0..7."""
    axis = wire + 1  # batch occupies axis zero
    permutation = [0] + [i for i in range(1, 9) if i != axis] + [axis]
    inverse = [permutation.index(i) for i in range(9)]
    moved = state.permute(permutation)
    moved = moved.reshape(-1, 2) @ gate.transpose(0, 1)
    return moved.reshape([state.shape[0]] + [2] * 7 + [2]).permute(inverse)


def _apply_2q(state: torch.Tensor, gate: torch.Tensor, wire_a: int, wire_b: int) -> torch.Tensor:
    """Apply a two-qubit gate; wire_a is the most-significant gate input."""
    a, b = wire_a + 1, wire_b + 1
    permutation = [0] + [i for i in range(1, 9) if i not in (a, b)] + [a, b]
    inverse = [permutation.index(i) for i in range(9)]
    moved = state.permute(permutation)
    moved = moved.reshape(-1, 4) @ gate.transpose(0, 1)
    return moved.reshape([state.shape[0]] + [2] * 6 + [2, 2]).permute(inverse)


def _cc14(state: torch.Tensor, params: torch.Tensor, wires: tuple[int, ...]) -> torch.Tensor:
    """Paper F1/CC14 ansatz; params has 4 * len(wires) elements."""
    n = len(wires)
    for i, wire in enumerate(wires):
        state = _apply_1q(state, _ry(params[i]), wire)
    for i in range(n):
        state = _apply_2q(state, _crx(params[i + n]), wires[(i - 1) % n], wires[i])
    for i, wire in enumerate(wires):
        state = _apply_1q(state, _ry(params[i + 2 * n]), wire)
    if n % 3 == 0 or n == 2:
        for i in reversed(range(n)):
            state = _apply_2q(state, _crx(params[i + 3 * n]), wires[i], wires[(i - 1) % n])
    else:
        control, target = n - 1, (n + 2) % n
        for i in reversed(range(n)):
            state = _apply_2q(state, _crx(params[i + 3 * n]), wires[control], wires[target])
            control, target = target, (target + 2) % n
    return state


def _su4(state: torch.Tensor, params: torch.Tensor, wire_a: int, wire_b: int) -> torch.Tensor:
    """Paper U_SU4 block (15 angles)."""
    state = _apply_1q(state, _u3(params[0], params[1], params[2]), wire_a)
    state = _apply_1q(state, _u3(params[3], params[4], params[5]), wire_b)
    state = _apply_2q(state, _cnot(state.device), wire_a, wire_b)
    state = _apply_1q(state, _ry(params[6]), wire_a)
    state = _apply_1q(state, _rz(params[7]), wire_b)
    state = _apply_2q(state, _cnot(state.device), wire_b, wire_a)
    state = _apply_1q(state, _ry(params[8]), wire_a)
    state = _apply_2q(state, _cnot(state.device), wire_a, wire_b)
    state = _apply_1q(state, _u3(params[9], params[10], params[11]), wire_a)
    state = _apply_1q(state, _u3(params[12], params[13], params[14]), wire_b)
    return state


def _conv_layer(state: torch.Tensor, params: torch.Tensor, wires: tuple[int, ...]) -> torch.Tensor:
    n = len(wires)
    f1_size = 4 * (n // 2) if n == 8 else 4 * n
    if n == 8:
        # The source circuit shares one F1 parameter block between both halves.
        state = _cc14(state, params[:f1_size], wires[:4])
        state = _cc14(state, params[:f1_size], wires[4:])
    else:
        state = _cc14(state, params[:f1_size], wires)
    pair_params = params[f1_size:f1_size + 15]
    for i in range(0, n, 2):
        state = _su4(state, pair_params, wires[i], wires[(i + 1) % n])
    for i in range(1, n, 2):
        state = _su4(state, pair_params, wires[i], wires[(i + 1) % n])
    return state


def _pool(state: torch.Tensor, params: torch.Tensor, control: int, target: int) -> torch.Tensor:
    state = _apply_2q(state, _crz(params[0]), control, target)
    state = _apply_1q(state, _x(state.device), control)
    return _apply_2q(state, _crx(params[1]), control, target)


class PaperMultiClassQCNN(nn.Module):
    """Paper's 8-qubit, 10-class amplitude-encoded QCNN (105 trainable angles).

    The 256 PCA values are contracted as an exact rank-8 state tensor. Outcomes
    are measured on physical wires [0, 2, 4, 6], yielding 16 probabilities;
    classes 0..9 use the first ten outcomes, as in the reference implementation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.params = nn.Parameter(torch.randn(105))

    def forward(self, amplitudes: torch.Tensor) -> torch.Tensor:
        if amplitudes.ndim != 2 or amplitudes.shape[1] != 256:
            raise ValueError(f"Expected [batch, 256] PCA amplitudes, got {tuple(amplitudes.shape)}")
        amplitudes = amplitudes / amplitudes.norm(dim=1, keepdim=True).clamp_min(1e-12)
        state = amplitudes.to(torch.complex64).reshape(-1, *([2] * 8))
        p = self.params
        state = _conv_layer(state, p[0:31], tuple(range(8)))
        state = _pool(state, p[31:33], 7, 6)
        state = _pool(state, p[31:33], 1, 0)
        state = _conv_layer(state, p[33:72], (0, 2, 3, 4, 5, 6))
        state = _pool(state, p[72:74], 3, 2)
        state = _pool(state, p[72:74], 5, 4)
        state = _conv_layer(state, p[74:105], (0, 2, 4, 6))

        # PennyLane probabilities over the requested wire order [0,2,4,6].
        measured_first = state.permute(0, 1, 3, 5, 7, 2, 4, 6)
        probs = measured_first.abs().square().reshape(-1, 16, 16).sum(dim=-1)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def predict_digits(probabilities: torch.Tensor) -> torch.Tensor:
        return probabilities[:, :10].argmax(dim=1)
