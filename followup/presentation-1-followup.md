# Presentation 1 Follow-Up: Slide to Code Demo Map

## Demo Flow

1. Start in presentation mode for slides 1-8.
2. Switch to VS Code at slide 9 for the first live CUDA runtime demo.
3. Stay close to code for slides 17-32: CPU oracle, CUDA kernel, host wrapper, tests.
4. Return to presentation mode for recap slides.
5. Switch again for slides 38-43 to run benchmarks and the end-to-end target sequence.

## Slide to Code Map

| Slides | VS Code files | What to explain or demo |
| --- | --- | --- |
| 1-3 | `projects/detailed/README.md` | Course framing, two-day roadmap, runnable Bazel targets. |
| 4-5 | `projects/detailed/README.md` | Repository shape, learning contract, correctness before performance. |
| 6 | `projects/detailed/runtime/device_info.cc` | Host code asks CUDA runtime about the GPU. Explain CPU orchestration vs GPU execution. |
| 7 | `projects/detailed/cpp/vector_threaded.cc`, `projects/detailed/gpu/vector_add_device.cu` | Compare CPU threaded chunks with CUDA's one-thread-per-element model. |
| 8 | `projects/detailed/cpp/vector_cpu.cc` | Show the course operation: `out[i] = alpha * a[i] + b[i]`. |
| 9-10 | `projects/simple/src/main.cc`, `projects/simple/src/BUILD.bazel` | First live demo: `bazel run //src:gpu_info`. Show `cudaGetDeviceCount`, versions, and `cudaDeviceProp`. |
| 11-15 | `projects/detailed/MODULE.bazel`, `projects/detailed/.bazelrc` | Hermetic build setup: rules, LLVM/Clang, CUDA redistributable toolkit, CUDA compiler mode. |
| 16 | `projects/detailed/README.md` | Lab checkpoint: `bazel test //...` from `projects/detailed`. |
| 17 | `projects/detailed/cpp/vector_cpu.cc`, `projects/detailed/cpp/vector_cpu.h` | CPU oracle. Size validation, simple loop, reference output. |
| 18 | `projects/detailed/cpp/vector_cpu_test.cc`, `projects/detailed/cpp/BUILD.bazel` | CPU test target and contract capture before CUDA. |
| 19 | `projects/detailed/cpp/vector_threaded.cc`, `projects/detailed/cpp/vector_threaded.h` | Threaded CPU baseline: worker count, chunks, `std::thread`, join. |
| 20 | `projects/detailed/cpp/vector_cpu_benchmark.cc` | CPU benchmark: input size, serial timing, threaded timing. |
| 21 | `projects/detailed/gpu/vector_add.h`, `projects/detailed/gpu/vector_add_cuda.cc`, `projects/detailed/gpu/vector_add_device.h`, `projects/detailed/gpu/vector_add_device.cu` | CUDA source separation: API, host wrapper, device launcher, kernel. |
| 22 | `projects/detailed/gpu/BUILD.bazel` | CUDA target graph: `cuda_library`, `cc_library`, tests, benchmarks. |
| 23-25 | `projects/detailed/gpu/vector_add_device.cu` | Kernel walkthrough: `blockIdx`, `blockDim`, `threadIdx`, bounds check, block size 256, rounded grid size. |
| 26-27 | `projects/detailed/gpu/vector_add_cuda.cc` | Host wrapper: validate inputs, allocate device buffers, copy, launch, copy back, synchronize. |
| 28 | `projects/detailed/runtime/cuda_errors.h` | Error handling with `CUDA_CHECK`, file and line context. |
| 29 | `projects/detailed/runtime/cuda_launch.h` | RAII helpers for streams and events. |
| 30 | `projects/detailed/gpu/vector_add.h`, `projects/detailed/gpu/vector_add_device.cu` | Two launch styles: `LaunchApi::kRuntimeApi` and `LaunchApi::kTripleChevron`. |
| 31-32 | `projects/detailed/gpu/vector_add_test.cc` | Correctness test: CPU output compared to both CUDA launch styles. Demo `bazel test --nocache_test_results //gpu:vector_add_test`. |
| 33 | `projects/detailed/.bazelrc`, `projects/detailed/MODULE.bazel` | Demo `bazel aquery 'mnemonic("CppCompile", //gpu:vector_add_test)'` and point to hermetic toolchain paths. |
| 34 | `projects/detailed/gpu/BUILD.bazel`, `projects/detailed/runtime/BUILD.bazel`, `projects/detailed/cpp/BUILD.bazel` | Build graph as architecture diagram. |
| 35 | `projects/detailed/.bazelrc` | CUDA architecture flag: `compute_86`, `sm_86`. |
| 36 | `projects/detailed/runtime/device_info.cc`, `projects/detailed/runtime/cuda_errors.h` | Troubleshooting: no GPU, driver problem, architecture mismatch, unchecked CUDA failures. |
| 37 | `projects/detailed/gpu/vector_add_device.cu` | Lab: explain the kernel line by line. |
| 38-39 | `projects/detailed/gpu/vector_add_benchmark.cc` | Change benchmark vector size, rerun CUDA test, run benchmark. |
| 40 | `projects/simple/src/BUILD.bazel`, `projects/detailed/deploy/BUILD.bazel` | Visual Studio workflow replacement: source files stay editor-agnostic, Bazel defines build/package flow. |
| 41-43 | `projects/detailed/README.md`, `projects/detailed/docs/benchmark_results.md` | Day 1 integration demo and recap. |
| 44 | `projects/detailed/python/jax_xla_demo.py`, `projects/detailed/linear_algebra/BUILD.bazel` | Day 2 preview: JAX/XLA and MatMul. |
| 45-46 | `projects/detailed/README.md` | Exercise pack and final close: repository declares build, CPU verifies CUDA, Clang compiles host/device code. |

## Live Commands

### Simple Project

```bash
cd /Users/kevin/formation/projects/simple
bazel run //src:gpu_info
```

### Detailed Project

```bash
cd /Users/kevin/formation/projects/detailed
bazel test //...
bazel run //runtime:device_info
bazel test //cpp:vector_cpu_test
bazel test --nocache_test_results //gpu:vector_add_test
bazel run //gpu:vector_add_benchmark
```

### Hermetic Compiler Inspection

```bash
cd /Users/kevin/formation/projects/detailed
bazel aquery 'mnemonic("CppCompile", //gpu:vector_add_test)'
```

Look for LLVM/toolchain paths from Bazel-managed external repositories rather than an ambient host compiler path.

## Best VS Code Open Order

Open these tabs before the talk:

1. `projects/detailed/README.md`
2. `projects/simple/src/main.cc`
3. `projects/detailed/MODULE.bazel`
4. `projects/detailed/.bazelrc`
5. `projects/detailed/cpp/vector_cpu.cc`
6. `projects/detailed/cpp/vector_threaded.cc`
7. `projects/detailed/gpu/vector_add_device.cu`
8. `projects/detailed/gpu/vector_add_cuda.cc`
9. `projects/detailed/runtime/cuda_errors.h`
10. `projects/detailed/runtime/cuda_launch.h`
11. `projects/detailed/gpu/vector_add_test.cc`
12. `projects/detailed/gpu/vector_add_benchmark.cc`
13. `projects/detailed/gpu/BUILD.bazel`

## Short Presenter Notes

For slides 1-8, keep the story high level: this is no longer an IDE-specific CUDA sample, it is a repository-defined CUDA engineering workflow.

For slides 9-10, prove the GPU is visible before discussing kernels.

For slides 17-20, emphasize that the CPU implementation is not throwaway code. It is the correctness oracle.

For slides 21-32, show that the CUDA feature is made of four clean layers:

1. Public API: `gpu/vector_add.h`
2. Host orchestration: `gpu/vector_add_cuda.cc`
3. Device launcher and kernel: `gpu/vector_add_device.{h,cu}`
4. Test and benchmark targets: `gpu/vector_add_test.cc`, `gpu/vector_add_benchmark.cc`

For slides 33-35, use Bazel introspection to show that the build is inspectable and reproducible.

For slides 38-43, close with runnable commands, not just explanation.

