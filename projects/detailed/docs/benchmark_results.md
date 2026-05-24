# Benchmark Results

Measured on 2026-05-24 with:

```text
GPU                 : NVIDIA GeForce RTX 5090
CUDA runtime version: 12.8 (12080)
CUDA driver version : 13.0 (13000)
```

Commands:

```bash
bazel test //...
bazel run //runtime:device_info
bazel run //cpp:vector_cpu_benchmark
bazel run //gpu:vector_add_benchmark
bazel run //gpu:vector_launch_benchmark
bazel run //linear_algebra:matmul_benchmark
```

| Version | Target | Time | Lesson |
| --- | --- | ---: | --- |
| CPU serial | `//cpp:vector_cpu_benchmark` | 116.910 ms | correctness baseline |
| CPU threaded | `//cpp:vector_cpu_benchmark` | 70.634 ms | CPU parallelism |
| CUDA runtime API | `//gpu:vector_add_benchmark` | 218.355 ms | launch and transfer model |
| CUDA `<<< >>>` | `//gpu:vector_add_benchmark` | 94.239 ms | launch syntax comparison |
| Launch API host enqueue | `//gpu:vector_launch_benchmark` | 1.878 us | warm launch overhead without allocation/copy |
| Launch `<<< >>>` host enqueue | `//gpu:vector_launch_benchmark` | 1.878 us | warm launch overhead without allocation/copy |
| Launch API enqueue + sync | `//gpu:vector_launch_benchmark` | 2.092 us | warm launch plus stream drain |
| Launch `<<< >>>` enqueue + sync | `//gpu:vector_launch_benchmark` | 2.091 us | warm launch plus stream drain |
| MatMul CPU | `//linear_algebra:matmul_benchmark` | 37.994 ms | GEMM oracle |
| MatMul basic CUDA | `//linear_algebra:matmul_benchmark` | 252.155 ms | one thread per output |
| MatMul tiled CUDA | `//linear_algebra:matmul_benchmark` | 0.741 ms | shared-memory reuse |

Notes:

- `bazel test //...` passed all 3 tests.
- The vector CPU numbers come from `//cpp:vector_cpu_benchmark`.
- The CUDA vector numbers come from `//gpu:vector_add_benchmark`, which also
  reported `cpu=4 cuda=4` for the sample value.
- The launch microbenchmark comes from `//gpu:vector_launch_benchmark`. It warms
  up CUDA first, runs both launch variants repeatedly, alternates order each
  round, and excludes `cudaMalloc`, `cudaFree`, and `cudaMemcpy` from the timed
  region.
- The MatMul benchmark used `M=256`, `N=256`, `K=256` and reported sample `512`.
- These are single-run lab measurements, not statistically stable final
  performance numbers.
