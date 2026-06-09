from __future__ import annotations

import torch
import triton

from python.triton_kernels import triton_matmul, triton_saxpy, triton_vector_add


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Triton benchmark requires a Linux NVIDIA CUDA GPU")
    return torch.device("cuda")


def _bench(label: str, torch_fn, triton_fn, bytes_moved: int) -> None:
    torch_ms = triton.testing.do_bench(torch_fn)
    triton_ms = triton.testing.do_bench(triton_fn)

    def gbps(ms: float) -> float:
        return bytes_moved * 1e-9 / (ms * 1e-3)

    print(f"{label}")
    print(f"  torch_ms={torch_ms:.4f}  torch_gb_s={gbps(torch_ms):.2f}")
    print(f"  triton_ms={triton_ms:.4f} triton_gb_s={gbps(triton_ms):.2f}")


def main() -> None:
    device = _require_cuda()
    torch.manual_seed(0)

    n = 1 << 24
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)
    _bench(
        "vector_add",
        lambda: a + b,
        lambda: triton_vector_add(a, b),
        bytes_moved=3 * a.numel() * a.element_size(),
    )

    alpha = 2.5
    _bench(
        "saxpy",
        lambda: alpha * a + b,
        lambda: triton_saxpy(a, b, alpha),
        bytes_moved=3 * a.numel() * a.element_size(),
    )

    m, k, n_cols = 512, 384, 640
    lhs = torch.randn((m, k), device=device)
    rhs = torch.randn((k, n_cols), device=device)
    matmul_bytes = (lhs.numel() + rhs.numel() + m * n_cols) * lhs.element_size()
    _bench(
        "matmul",
        lambda: lhs @ rhs,
        lambda: triton_matmul(lhs, rhs),
        bytes_moved=matmul_bytes,
    )

    print("Run this target twice before interpreting timings; first launch pays JIT/cache cost.")


if __name__ == "__main__":
    main()
