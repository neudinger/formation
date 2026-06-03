# Exercise 08: Python Benchmark Orchestration and JAX/XLA Hook

## Goal

Use hermetic Python as an orchestration layer and expose a JAX/XLA comparison hook that degrades clearly when JAX is absent.

## What to finish

- Return the benchmark commands in execution order.
- Detect whether JAX is importable.
- Print a clear fallback message when JAX is unavailable.

## Commands

```bash
bazel build //...
bazel test //python:python_hooks_test
bazel run //python:benchmark
bazel run //python:jax_xla_demo
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day2/ex08_python_jax_xla`.

