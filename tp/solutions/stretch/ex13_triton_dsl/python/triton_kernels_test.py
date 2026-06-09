from __future__ import annotations

import torch

from python.triton_kernels import triton_matmul, triton_saxpy, triton_vector_add


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Triton stretch tests require a Linux NVIDIA CUDA GPU")
    return torch.device("cuda")


def test_vector_add_masked_edge_size() -> None:
    device = _require_cuda()
    torch.manual_seed(0)
    a = torch.randn(2057, device=device)
    b = torch.randn(2057, device=device)
    torch.testing.assert_close(triton_vector_add(a, b), a + b)


def test_saxpy_masked_edge_size() -> None:
    device = _require_cuda()
    torch.manual_seed(1)
    a = torch.randn(3001, device=device)
    b = torch.randn(3001, device=device)
    alpha = 2.5
    torch.testing.assert_close(triton_saxpy(a, b, alpha), alpha * a + b)


def test_square_matmul() -> None:
    device = _require_cuda()
    torch.manual_seed(2)
    a = torch.randn((32, 32), device=device)
    b = torch.randn((32, 32), device=device)
    torch.testing.assert_close(triton_matmul(a, b), a @ b, atol=1e-2, rtol=1e-2)


def test_rectangular_matmul_masked_edges() -> None:
    device = _require_cuda()
    torch.manual_seed(3)
    a = torch.randn((37, 29), device=device)
    b = torch.randn((29, 41), device=device)
    torch.testing.assert_close(triton_matmul(a, b), a @ b, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_vector_add_masked_edge_size()
    test_saxpy_masked_edge_size()
    test_square_matmul()
    test_rectangular_matmul_masked_edges()
