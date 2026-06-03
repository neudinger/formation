# Exercise 01: GPU Environment and Bazel Foundation

## Goal

Complete a small CUDA runtime information binary and inspect the isolated Bazel module that builds it.

## What to finish

- Finish the CUDA version formatter.
- Finish the compact device summary helper.
- Run the binary to confirm the runtime machine can see the GPU.

## Commands

```bash
bazel build //...
bazel test //src:gpu_info_test
bazel run //src:gpu_info
```

The starter project should build before you edit it. Its tests are expected to
fail until the TODO implementations are complete.

The complete answer is mirrored at `tp/solutions/day1/ex01_gpu_bazel_foundation`.
## Notes

- The test does not require a GPU; the `//src:gpu_info` binary does.

