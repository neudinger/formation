# Detailed CUDA Bazel Course Project

This is the repository-backed implementation for `../../detailed.md`: a
2-day CUDA, Bazel, Clang, C++26, and JAX/XLA course project. It is intentionally
isolated from `../simple` and has its own Bazel module, lockfile, toolchains,
targets, and documentation.

The project builds from a CPU correctness baseline to CUDA kernels, benchmark
binaries, a runtime device-info tool, Python/JAX integration hooks, and a
separate dense linear algebra module for MatMul/GEMM lessons.

## Hermetic Build Model

The repository defines the build environment:

- C++ compiler: hermetic LLVM/Clang 20.1.8 from `toolchains_llvm`.
- C++ mode: C++26, configured in `.bazelrc` and repeated where CUDA compilation
  needs it.
- CUDA toolkit: CUDA 12.8.0 redistributable components from `rules_cuda`.
- CUDA compiler mode: Clang CUDA, configured through `rules_cuda`.
- Python: hermetic Python 3.13 from `rules_python`.

Do not point `CC`, `CXX`, or `CUDA_HOME` at host paths for normal builds. The
runtime machine still needs an NVIDIA driver and CUDA-capable GPU.

## Repository Map

| Directory | Purpose |
| --- | --- |
| `cpp/` | CPU serial and threaded vector baselines. |
| `gpu/` | CUDA vector-add implementation, tests, and benchmark. |
| `runtime/` | CUDA error helpers, launch helpers, and device info binary. |
| `python/` | Hermetic Python targets and optional JAX/XLA lesson hook. |
| `linear_algebra/` | Standalone MatMul module: CPU, basic CUDA, tiled CUDA. |
| `deploy/` | Deployment-oriented file aggregation target. |
| `docs/` | Benchmark worksheet/results notes. |

## Quick Start

```bash
bazel test //...
bazel run //runtime:device_info
bazel run //cpp:vector_cpu_benchmark
bazel run //gpu:vector_add_benchmark
bazel run //gpu:vector_launch_benchmark
bazel run //linear_algebra:matmul_benchmark
```

Run the Python lesson hooks:

```bash
bazel run //python:benchmark
bazel run //python:jax_xla_demo
```

`//python:jax_xla_demo` uses a hermetic Python interpreter. JAX itself is left
as an explicit packaging exercise because GPU JAX wheels depend on the chosen
CUDA/JAX wheel matrix.

## Important Targets

| Target | What it does |
| --- | --- |
| `//runtime:device_info` | Prints CUDA runtime, driver, and device properties. |
| `//cpp:vector_cpu_test` | Verifies serial and threaded CPU vector addition. |
| `//cpp:vector_cpu_benchmark` | Times CPU serial and threaded vector addition. |
| `//gpu:vector_add_test` | Compares CUDA vector addition with the CPU oracle. |
| `//gpu:vector_add_benchmark` | Times CPU serial, CPU threaded, `cudaLaunchKernel`, and `<<< >>>`. |
| `//gpu:vector_launch_benchmark` | Compares warm launch overhead for `cudaLaunchKernel` and `<<< >>>` without allocation/copy timing. |
| `//linear_algebra:matmul_test` | Checks CUDA MatMul implementations against CPU output. |
| `//linear_algebra:matmul_benchmark` | Times CPU, basic CUDA, and tiled CUDA MatMul. |
| `//python:benchmark` | Prints the benchmark command sequence for the lab. |
| `//python:jax_xla_demo` | Optional JAX/XLA comparison hook. |
| `//deploy:vector_add_runtime_files` | Collects runtime binaries for deployment exercises. |

## CUDA Architecture

The default `.bazelrc` targets `compute_86` PTX plus `sm_86` code:

```text
build --@rules_cuda//cuda:archs=compute_86:compute_86,sm_86
```

That keeps the course runnable on Ampere-class machines and newer GPUs that can
JIT from compatible PTX. For a specific lab GPU, update the architecture in
`.bazelrc` before benchmarking so the generated device code matches the hardware
you want to teach against.

## Benchmark Workflow

1. Confirm the runtime environment:

   ```bash
   bazel run //runtime:device_info
   ```

2. Run correctness tests:

   ```bash
   bazel test //...
   ```

3. Collect timings:

   ```bash
   bazel run //cpp:vector_cpu_benchmark
   bazel run //gpu:vector_add_benchmark
   bazel run //gpu:vector_launch_benchmark
   bazel run //linear_algebra:matmul_benchmark
   ```

4. Record results in `docs/benchmark_results.md`.

## Course Flow

The core lesson path is:

```text
CPU oracle
  -> threaded CPU baseline
  -> CUDA vector-add kernel
  -> runtime API launch helper
  -> triple-chevron launch comparison
  -> benchmarks
  -> deployment target
  -> Python/JAX/XLA comparison
```

The MatMul/GEMM work is intentionally separate:

```text
CPU MatMul
  -> one-thread-per-output CUDA MatMul
  -> tiled shared-memory CUDA MatMul
  -> benchmark and correctness comparison
```

This separation keeps the main 2-day course focused while still giving the
linear algebra material a realistic standalone module.

## Troubleshooting

- `cudaGetDeviceCount` fails: check that the machine has an NVIDIA driver and
  the user can access the GPU.
- CUDA code builds but does not run on the lab GPU: check the `archs` setting in
  `.bazelrc`.
- Python target runs but JAX is missing: expected for the minimal hermetic
  Python toolchain; use `//python:jax_xla_demo` as the integration target for
  the JAX packaging exercise.
- A build appears to use a host compiler: inspect with `bazel aquery` and look
  for `external/toolchains_llvm` in the compile action.
