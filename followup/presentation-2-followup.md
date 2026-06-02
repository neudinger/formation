# Presentation 2 Follow-Up: Slide to Code Demo Map

## Recommended Demo Flow

1. Start in presentation mode for slides 1-3.
2. Switch to VS Code at slide 4 to run `device_info` and ground the day in the real GPU.
3. Stay in code for slides 6-20 to explain timing, memory movement, allocation, streams, and CUDA error behavior.
4. Switch to `projects/simple` for slides 21-28 and demonstrate the OCI image target.
5. Use slides 31-33 for Python/JAX as a comparison layer, not as a replacement for the CUDA code.
6. Spend the main second-half code walkthrough on slides 34-43: CPU MatMul, basic CUDA MatMul, tiled CUDA MatMul, benchmark, and cuBLAS hook.
7. Close with the final end-to-end command sequence on slide 45.

## Slide to Code Map

| Slides | VS Code files | What to explain or demo |
| --- | --- | --- |
| 1-3 | `projects/detailed/README.md`, `projects/detailed/gpu/BUILD.bazel` | Day 2 framing: measure, package, compare, then extend from vector add to MatMul. |
| 4-5 | `projects/detailed/runtime/device_info.cc`, `projects/detailed/runtime/BUILD.bazel` | Run `bazel run //runtime:device_info`. Show runtime version, driver version, device count, SM count, warp size, async engines, and memory fields. |
| 6 | `projects/detailed/gpu/vector_add_benchmark.cc`, `projects/detailed/docs/benchmark_results.md` | Measurement discipline: warmup, sync scope, allocation/copy inclusion, sample correctness. |
| 7 | `projects/detailed/runtime/cuda_launch.h`, `projects/detailed/gpu/vector_launch_benchmark.cc` | CUDA work is asynchronous. Explain stream sync, event timing helper, and launch-only benchmark structure. |
| 8-9 | `projects/detailed/gpu/vector_add_benchmark.cc`, `projects/detailed/docs/benchmark_results.md` | Four benchmark paths: CPU serial, CPU threaded, CUDA Runtime API, CUDA triple-chevron. Read output responsibly. |
| 10-13 | `projects/detailed/gpu/vector_add_cuda.cc` | Host/device memory flow: allocate device buffers, H2D copies for inputs, kernel writes output, D2H copy for result, stream synchronize. |
| 11 | `projects/detailed/docs/benchmark_results.md`, `projects/detailed/gpu/vector_add_benchmark.cc` | Lab: estimate bytes moved for `N = 1 << 24`: `3 * N * sizeof(float)`. |
| 12 | `projects/detailed/gpu/vector_add_cuda.cc`, `projects/detailed/gpu/vector_launch_benchmark.cc` | Allocation cost: full function timing includes malloc/free and copies; launch microbenchmark separates launch overhead. |
| 14 | `projects/detailed/gpu/vector_launch_benchmark.cc`, `projects/detailed/gpu/vector_add_device.cu` | Kernel launch cost: enqueue, synchronize, amortization, tiny-work overhead. |
| 15-18 | `projects/detailed/runtime/cuda_launch.h`, `projects/detailed/gpu/vector_add_cuda.cc`, `projects/detailed/runtime/device_info.cc` | Streams and overlap design. The current code uses one explicit stream as a serialized baseline; slide 18 is the future multi-stream extension. |
| 19-20 | `projects/detailed/runtime/cuda_errors.h`, `projects/detailed/gpu/vector_add_test.cc`, `projects/detailed/runtime/device_info.cc`, `projects/detailed/.bazelrc` | Async error handling and debugging order: device info, smallest correctness test, `CUDA_CHECK`, input reduction, architecture flags. |
| 21-24 | `projects/simple/src/BUILD.bazel`, `projects/simple/src/main.cc`, `projects/simple/README.md` | Packaging motivation and simple deployment specimen: `cc_binary`, `pkg_tar`, `oci_image`, `oci_load`, runfiles. |
| 25 | `projects/simple/MODULE.bazel` | Digest-pinned OCI base image and declared repository inputs. |
| 26-27 | `projects/simple/src/BUILD.bazel`, `projects/simple/README.md` | Live demo: load the image and run it with GPU access; build the image tarball artifact. |
| 28 | `projects/detailed/deploy/BUILD.bazel` | Detailed project deploy target: explicit runtime file grouping for future image work. |
| 29-30 | `projects/detailed/MODULE.bazel`, `projects/detailed/.bazelrc`, `projects/simple/MODULE.bazel` | Remote build model: hermetic build inputs are declared, but runtime GPU driver/device access remains environment-specific. |
| 31 | `projects/detailed/python/benchmark.py`, `projects/detailed/python/BUILD.bazel` | Python benchmark orchestration: Python points to Bazel-built binaries instead of owning the compute path. |
| 32 | `projects/detailed/python/jax_xla_demo.py`, `projects/detailed/python/BUILD.bazel` | JAX/XLA hook: `jax.jit`, `lower()`, `compile()`, and the explicit note that JAX GPU packaging is a separate exercise. |
| 33 | `projects/detailed/gpu/vector_add_device.cu`, `projects/detailed/gpu/vector_add_cuda.cc`, `projects/detailed/python/jax_xla_demo.py` | Manual CUDA vs JAX/XLA: explicit memory and launch geometry versus compiler-managed array programs. |
| 34-35 | `projects/detailed/linear_algebra/BUILD.bazel`, `projects/detailed/README.md` | Transition from vector add to MatMul. Show the separate linear algebra module and target graph. |
| 36 | `projects/detailed/linear_algebra/matmul_cpu.cc`, `projects/detailed/linear_algebra/matmul_cpu.h`, `projects/detailed/linear_algebra/matmul_test.cc` | CPU MatMul oracle: nested loops over M, N, K and correctness baseline for CUDA. |
| 37-38 | `projects/detailed/linear_algebra/matmul_basic_device.cu`, `projects/detailed/linear_algebra/matmul_basic_cuda.cc` | Basic CUDA MatMul: one thread computes one output element, loops over K, poor global-memory reuse. |
| 39-40 | `projects/detailed/linear_algebra/matmul_tiled_device.cu`, `projects/detailed/linear_algebra/matmul_tiled_cuda.cc` | Tiled MatMul: `__shared__` tiles, cooperative loading, `__syncthreads()`, reuse inside a block. |
| 41-42 | `projects/detailed/linear_algebra/matmul_benchmark.cc`, `projects/detailed/docs/benchmark_results.md` | MatMul benchmark ladder and benchmark worksheet: CPU, basic CUDA, tiled CUDA. |
| 43 | `projects/detailed/linear_algebra/matmul_cublas.cc`, `projects/detailed/linear_algebra/matmul_cublas.h` | Production GEMM hook: cuBLAS is intentionally not linked yet; it marks the production baseline lesson. |
| 44 | `projects/detailed/README.md`, `projects/detailed/python/jax_xla_demo.py`, `projects/detailed/linear_algebra/matmul_cublas.cc` | Choosing abstraction level: custom CUDA, CUDA libraries, JAX/XLA, and framework-managed math. |
| 45 | `projects/detailed/README.md`, `projects/simple/README.md` | Final end-to-end demo: tests, vector benchmark, MatMul benchmark, load/run GPU image. |
| 46-48 | `projects/detailed/README.md`, `projects/detailed/docs/benchmark_results.md`, `projects/detailed/deploy/BUILD.bazel` | Reusable patterns, capstone options, and closing summary. |

## Live Commands

### Device Reality

```bash
cd /Users/kevin/formation/projects/detailed
bazel run //runtime:device_info
```

### Runtime and Benchmark Measurement

```bash
cd /Users/kevin/formation/projects/detailed
bazel test //...
bazel run //gpu:vector_add_benchmark
bazel run //gpu:vector_launch_benchmark
```

### Python and JAX/XLA Hooks

```bash
cd /Users/kevin/formation/projects/detailed
bazel run //python:benchmark
bazel run //python:jax_xla_demo
```

### MatMul

```bash
cd /Users/kevin/formation/projects/detailed
bazel test //linear_algebra:matmul_test
bazel run //linear_algebra:matmul_benchmark
```

### Simple GPU Image

```bash
cd /Users/kevin/formation/projects/simple
bazel run //src:gpu_info
bazel run //src:load_gpu_info_image
docker run --rm --gpus all simple-gpu-info:latest
bazel build //src:gpu_info_image_tar
docker image ls simple-gpu-info:latest
```

### Final End-to-End Sequence

```bash
cd /Users/kevin/formation/projects/detailed
bazel test //...
bazel run //gpu:vector_add_benchmark
bazel run //linear_algebra:matmul_benchmark

cd /Users/kevin/formation/projects/simple
bazel run //src:load_gpu_info_image
docker run --rm --gpus all simple-gpu-info:latest
```

## Best VS Code Open Order

Open these tabs before the talk:

1. `projects/detailed/runtime/device_info.cc`
2. `projects/detailed/gpu/vector_add_benchmark.cc`
3. `projects/detailed/gpu/vector_launch_benchmark.cc`
4. `projects/detailed/gpu/vector_add_cuda.cc`
5. `projects/detailed/runtime/cuda_launch.h`
6. `projects/detailed/runtime/cuda_errors.h`
7. `projects/simple/src/BUILD.bazel`
8. `projects/simple/MODULE.bazel`
9. `projects/detailed/deploy/BUILD.bazel`
10. `projects/detailed/python/benchmark.py`
11. `projects/detailed/python/jax_xla_demo.py`
12. `projects/detailed/linear_algebra/BUILD.bazel`
13. `projects/detailed/linear_algebra/matmul_cpu.cc`
14. `projects/detailed/linear_algebra/matmul_basic_device.cu`
15. `projects/detailed/linear_algebra/matmul_tiled_device.cu`
16. `projects/detailed/linear_algebra/matmul_benchmark.cc`
17. `projects/detailed/docs/benchmark_results.md`

## Short Presenter Notes

For slides 1-5, make the point that performance discussion starts with the actual runtime machine, not an abstract GPU.

For slides 6-14, keep the benchmark story precise: say what the timing includes before interpreting whether something is fast or slow.

For slides 15-20, describe the current implementation as a clear serialized baseline. Multi-stream overlap is the next design exercise, and it needs pinned memory, independent chunks, non-default streams, copy engines, and explicit synchronization.

For slides 21-28, emphasize that deployment is represented in the repository. The image is built by Bazel targets, not by manual Dockerfile steps.

For slides 31-33, position Python/JAX as a comparison of abstraction layers. Manual CUDA teaches control; JAX/XLA teaches compiler-managed array programming.

For slides 34-43, narrate the MatMul ladder:

1. CPU oracle: `matmul_cpu.cc`
2. Basic CUDA: `matmul_basic_device.cu`
3. Tiled CUDA: `matmul_tiled_device.cu`
4. Benchmark and worksheet: `matmul_benchmark.cc`, `docs/benchmark_results.md`
5. Production library hook: `matmul_cublas.cc`

For slide 45, finish with commands. The final proof is that the repository can test, benchmark, and package.

