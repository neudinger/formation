# Simple CUDA GPU Info

Small Bazel + Clang CUDA project for the course foundation in `reworked.md`.

The CUDA toolkit used by the build is declared through NVIDIA CUDA 13.2.1
redistributable components, so Bazel downloads the CUDA headers, runtime,
`nvcc`, `ptxas`, CRT, and NVVM inputs instead of relying on a locally installed
CUDA toolkit. The C++ compiler is hermetic too: Bazel downloads LLVM/Clang
20.1.8 through `toolchains_llvm`. The runtime machine still needs an NVIDIA
driver.

Run:

```bash
bazel build //src:gpu_info
bazel run //src:gpu_info
```

Build the tiny GPU-enabled container image:

```bash
bazel build //src:gpu_info_image
bazel run //src:load_gpu_info_image
docker run --rm --gpus all simple-gpu-info:latest
```

For a loadable Docker tarball artifact:

```bash
bazel build //src:gpu_info_image_tar
```

The tarball is emitted at `bazel-bin/src/load_gpu_info_image/tarball.tar`.
The image uses a digest-pinned `gcr.io/distroless/cc-debian12` base and packages
the Bazel runfiles that contain the hermetic CUDA runtime shared library.

The binary calls `cudaGetDeviceCount`, `cudaRuntimeGetVersion`,
`cudaDriverGetVersion`, and `cudaGetDeviceProperties`, then prints every public
field from `cudaDeviceProp` for each CUDA-capable device.
