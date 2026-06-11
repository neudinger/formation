# Exercise 08: Python Benchmark Orchestration and JAX/XLA Hook

## Goal

Use hermetic Python as an orchestration layer and expose a Bazel-runnable JAX/XLA SAXPY demo. Python dependencies are resolved from the single `requirement.txt` file with the `rules_python` uv lock rule.

## What to finish

- Return the benchmark commands in execution order.
- Keep the JAX dependency graph pinned in `requirement.txt`.
- Run the copied JAX/XLA SAXPY demo through Bazel.
- Use the `cuda12` JAX extra so the Bazel Python environment vendors the CUDA 12 NVIDIA wheels through uv/PyPI.

## Commands

```bash
bazel build //...
bazel test //python:python_hooks_test
bazel run //python:benchmark
bazel run //:requirement.update
bazel run //python:jax_xla_demo
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day2/ex08_python_jax_xla`.
