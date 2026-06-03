# Exercise 07: Deployment Container

## Goal

Package a CUDA runtime binary into a Bazel-built OCI image while keeping the normal binary runnable on Windows and Linux.

## What to finish

- Complete the image repository tag metadata.
- Check the Linux-only OCI image targets.
- Run the direct binary before trying the container.

## Commands

```bash
bazel build //...
bazel test //src:packaging_metadata_test
bazel run //src:gpu_info
bazel run //src:load_gpu_info_image
docker run --rm --gpus all tp-gpu-info:latest
```

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/day2/ex07_deployment_container`.
## Notes

- The OCI image targets are Linux-only, matching the course `projects/simple` module.

