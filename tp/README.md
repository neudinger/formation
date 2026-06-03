# CUDA Bazel TP Exercises

This directory contains a two-day CUDA/Bazel training pack. Every exercise is an
independent Bazel module with its own `MODULE.bazel`, `.bazelrc`, root
`BUILD.bazel`, local build files, starter code, and mirrored solution.

## Layout

```text
tp/
  day1/       # required starter exercises 1-5
  day2/       # required starter exercises 6-10
  stretch/    # optional starter exercises
  solutions/  # complete mirrored answers
```

## Workflow

```bash
cd tp/day1/ex02_cpu_vector_oracle
bazel build //...
bazel test //...
```

`bazel build //...` should succeed. `bazel test //...` is expected to fail until
the TODOs are completed. Compare with the matching project under
`tp/solutions/...` when reviewing or demonstrating the answer.

CUDA runtime binaries and GPU tests require an NVIDIA driver and CUDA-capable
GPU. Linux uses the hermetic LLVM/Clang and CUDA redistributable toolchain;
Windows uses local Visual C++ plus NVCC through `rules_cuda`.

## Minimum Windows Setup

Install these on Windows 10/11 64-bit:

- Git for Windows
- Bazelisk
- Visual Studio 2022 Build Tools with the C++ workload
- NVIDIA driver
- CUDA-capable NVIDIA GPU

You normally do not need to install the CUDA Toolkit manually. The exercises
declare CUDA redistributable components in `MODULE.bazel`. Install a local CUDA
Toolkit only if Windows reports a missing `nvcc` or CUDA toolchain.

Run PowerShell as Administrator:

```powershell
winget source update

winget install -e --id Git.Git --source winget
winget install -e --id Bazel.Bazelisk --source winget

winget install -e --id Microsoft.VisualStudio.2022.BuildTools --source winget `
  --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Install the NVIDIA driver from:

```text
https://www.nvidia.com/Download/index.aspx
```

Then reboot and verify:

```powershell
git --version
bazelisk version
nvidia-smi
```

Check that the Visual C++ compiler is visible:

```powershell
cmd /c "`"%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 && cl"
```

Run an exercise:

```powershell
cd C:\path\to\formation\tp\day1\ex02_cpu_vector_oracle
bazelisk build //...
bazelisk test //...
```

Run a CUDA exercise:

```powershell
cd C:\path\to\formation\tp\day1\ex04_cuda_vector_add
bazelisk build //...
bazelisk test //src:vector_add_test
```

The deployment exercise has Linux-only OCI image targets. On native Windows, do
not run:

```powershell
bazelisk run //src:load_gpu_info_image
bazelisk build //src:gpu_info_image_tar
```

References:

- Bazel install overview: `https://preview.bazel.build/install`
- Bazel Windows setup: `https://docs.bazel.build/versions/main/install-windows.html`
- Visual Studio Build Tools workload IDs: `https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools`
- Git for Windows: `https://git-scm.com/install/windows`
- NVIDIA drivers: `https://www.nvidia.com/Download/index.aspx`
