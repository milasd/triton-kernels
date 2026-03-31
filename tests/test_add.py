"""Tests for vector addition kernel."""
import pytest
import torch
from triton_kernels.config import DEVICE
from triton_kernels.ops.add import add


@pytest.mark.parametrize("size", [96, 1024, 4096, 4097])
def test_add_kernel(size: int, atol: float = 1e-3, rtol: float = 1e-3):
    torch.manual_seed(42)
    m = torch.randn(size=(size,), device=DEVICE)
    n = torch.randn(size=(size,), device=DEVICE)

    m_plus_n = torch.add(m, n)
    add_kernel_out = add(m, n)

    torch.testing.assert_close(actual=add_kernel_out, expected=m_plus_n, rtol=rtol, atol=atol)
