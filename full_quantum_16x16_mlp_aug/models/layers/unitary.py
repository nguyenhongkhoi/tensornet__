from __future__ import annotations

import torch
from torch import nn


class LearnedUnitary1Q(nn.Module):
    """Learned 1-qubit unitary parameterized by matrix exponential."""

    def __init__(self, init_scale: float = 1e-2, real_only: bool = False):
        super().__init__()
        self.real_only = real_only
        self.raw_real = nn.Parameter(init_scale * torch.randn(2, 2))
        self.raw_imag = None if real_only else nn.Parameter(init_scale * torch.randn(2, 2))

    def matrix(self) -> torch.Tensor:
        if self.real_only:
            skew = self.raw_real - self.raw_real.T
            return torch.matrix_exp(skew).to(torch.complex64)

        a = torch.complex(self.raw_real, self.raw_imag)
        hermitian = 0.5 * (a + a.conj().T)
        return torch.matrix_exp(1j * hermitian)

    def forward(self) -> torch.Tensor:
        return self.matrix()


class LearnedUnitary2Q(nn.Module):
    """Learned 2-qubit unitary parameterized by matrix exponential.

    Complex mode learns an unconstrained complex matrix A, forms a Hermitian
    H=(A+A^dagger)/2, then returns exp(iH). Real-only mode returns an
    orthogonal matrix exp(A-A^T), kept as complex for downstream consistency.
    """

    def __init__(self, init_scale: float = 1e-2, real_only: bool = False):
        super().__init__()
        self.real_only = real_only
        self.raw_real = nn.Parameter(init_scale * torch.randn(4, 4))
        self.raw_imag = None if real_only else nn.Parameter(init_scale * torch.randn(4, 4))

    def matrix(self) -> torch.Tensor:
        if self.real_only:
            skew = self.raw_real - self.raw_real.T
            return torch.matrix_exp(skew).to(torch.complex64)

        a = torch.complex(self.raw_real, self.raw_imag)
        hermitian = 0.5 * (a + a.conj().T)
        return torch.matrix_exp(1j * hermitian)

    def forward(self) -> torch.Tensor:
        return self.matrix()
