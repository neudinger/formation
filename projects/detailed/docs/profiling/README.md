# MatMul Profiling Notes

This directory contains local profiling notes and generated Nsight Systems
reports for `//linear_algebra:matmul_benchmark`.

## Tool Availability on This Machine

```bash
nsys --version
ncu --version
command -v nvprof || true
```

Observed:

- `nsys`: available, version `2025.6.3`.
- `ncu`: available, version `2026.1.1`.
- `nvprof`: not installed on `PATH`.

`nvprof` is a legacy CUDA profiler. Use Nsight Systems for application timelines
and Nsight Compute for kernel metrics on modern CUDA setups.

## Build the Benchmark

```bash
bazel build //linear_algebra:matmul_benchmark
```

Run without profiler first:

```bash
bazel-bin/linear_algebra/matmul_benchmark
```

Observed sample output:

```text
shape      : M=256 N=256 K=256
cpu        : 39.414 ms
cuda basic : 242.505 ms
cuda tiled : 0.740 ms
sample     : 512
```

The exact numbers vary. The important teaching point is that app-level timing
includes runtime setup, allocation, copies, launch, synchronization, and kernel
work.

## Nsight Systems

Collect a CUDA timeline and summary:

```bash
nsys profile \
  --force-overwrite=true \
  --stats=true \
  --trace=cuda,osrt \
  --output=docs/profiling/matmul_nsys \
  bazel-bin/linear_algebra/matmul_benchmark
```

Generated here:

```text
docs/profiling/matmul_nsys.nsys-rep
docs/profiling/matmul_nsys.sqlite
```

Reprint selected stats:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
  docs/profiling/matmul_nsys.sqlite
```

Observed CUDA kernel summary:

```text
matmul_tiled_kernel : 62.592 us
matmul_basic_kernel : 62.079 us
```

Observed CUDA memory summary:

```text
Host-to-Device : 6 copies, 1.573 MB total
Device-to-Host : 2 copies, 0.524 MB total
```

Observed API summary highlight:

```text
cudaStreamCreate : dominates this short run because CUDA context/runtime setup
cudaLaunchKernel : 2 calls
cudaMemcpyAsync  : 8 calls
cudaMalloc       : 8 calls
cudaFree         : 8 calls
```

For teaching: explain why first-use CUDA runtime/context setup can dominate a
small benchmark and why warm-up runs matter.

CPU profiling was partially unavailable on this machine:

```bash
nsys status --environment
```

Reported `perf_event_open` and CPU sampling failures due system configuration.
CUDA tracing still worked.

## Nsight Compute

Attempted command:

```bash
ncu \
  --target-processes all \
  --kernel-name regex:matmul_.* \
  --set basic \
  --force-overwrite \
  --export docs/profiling/matmul_ncu \
  bazel-bin/linear_algebra/matmul_benchmark
```

This machine reported:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

To use Nsight Compute metrics, run with appropriate privileges or ask the system
administrator to enable GPU performance counter access for non-admin users.

## Legacy nvprof

If a legacy CUDA environment provides `nvprof`, the historical command shape is:

```bash
nvprof bazel-bin/linear_algebra/matmul_benchmark
```

For this project and modern CUDA toolchains, prefer:

```bash
nsys profile --trace=cuda,osrt ...
ncu --set basic --kernel-name regex:matmul_.* ...
```

