# CUDA Bazel Devcontainer

This devcontainer gives the CUDA/Bazel training repo a repeatable Ubuntu 24.04
VS Code environment with NVIDIA GPU passthrough. It intentionally does not
install a system CUDA Toolkit: the exercises fetch CUDA components through
`rules_cuda`, and the host only needs the NVIDIA driver/runtime support.

## Host Prerequisites

Linux:

- NVIDIA driver installed and visible on the host with `nvidia-smi`.
- Docker Engine or Docker Desktop.
- NVIDIA Container Toolkit configured for Docker.
- This command succeeds before opening VS Code:

  ```bash
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi
  ```

Windows:

- Windows 10/11 with an NVIDIA GPU and a current NVIDIA driver that supports
  CUDA on WSL2.
- Docker Desktop using the WSL2 backend.
- WSL kernel updated:

  ```powershell
  wsl --update
  ```

- This command succeeds in PowerShell before opening VS Code:

  ```powershell
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi
  ```

Native Windows containers and macOS GPU passthrough are not supported by this
setup.

## Open In VS Code

Install the VS Code Dev Containers extension, open this repository, and run:

```text
Dev Containers: Reopen in Container
```

The first build downloads the Ubuntu base image, devcontainer feature layers,
Bazelisk, and later Bazel/CUDA dependencies when you build an exercise.

## Smoke Checks

Inside the devcontainer:

```bash
lsb_release -ds
bazelisk version
nvidia-smi
docker version
```

Run a CUDA exercise:

```bash
cd tp/solutions/day1/ex01_gpu_bazel_foundation
bazel run //src:gpu_info
```

Run the deployment container exercise:

```bash
cd tp/solutions/day2/ex07_deployment_container
bazel run //src:load_gpu_info_image
docker run --rm --gpus all tp-gpu-info:latest
```

If `--gpus all` fails, fix Docker/NVIDIA runtime support on the host first; the
devcontainer is configured to require GPU access.
