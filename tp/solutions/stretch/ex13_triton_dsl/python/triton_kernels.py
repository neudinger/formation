from __future__ import annotations

import torch
import triton
import triton.language as tl


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@triton.jit
def _vector_add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@triton.jit
def _saxpy_kernel(a_ptr, b_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, alpha * a + b, mask=mask)


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(k, BLOCK_K)):
        k_offsets = k_start * BLOCK_K + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < m) & (k_offsets[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="tf32")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out_ptrs = out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_vector_add(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    _check_cuda_tensor("a", a)
    _check_cuda_tensor("b", b)
    if a.shape != b.shape:
        raise ValueError("triton_vector_add: input shape mismatch")
    out = torch.empty_like(a)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=block_size)
    return out


def triton_saxpy(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    block_size: int = 1024,
) -> torch.Tensor:
    _check_cuda_tensor("a", a)
    _check_cuda_tensor("b", b)
    if a.shape != b.shape:
        raise ValueError("triton_saxpy: input shape mismatch")
    out = torch.empty_like(a)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _saxpy_kernel[grid](a, b, out, float(alpha), n_elements, BLOCK_SIZE=block_size)
    return out


def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 32,
) -> torch.Tensor:
    _check_cuda_tensor("a", a)
    _check_cuda_tensor("b", b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_matmul: inputs must be 2D")
    if a.shape[1] != b.shape[0]:
        raise ValueError("triton_matmul: incompatible dimensions")
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out
