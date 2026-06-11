#!/usr/bin/env python3

"""Optional JAX/XLA comparison hook for the course.

The Bazel target provides a hermetic Python interpreter. JAX GPU wheels are left
as an explicit packaging exercise because they depend on the target CUDA/JAX
wheel matrix.
"""

from __future__ import annotations


def main() -> None:
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError:
        print("JAX is not installed in this minimal hermetic Python toolchain.")
        print("Use this target as the integration point for the JAX/XLA lesson.")
        return

    @jax.jit
    def saxpy(alpha, a, b):
        return alpha * a + b

    x = jnp.ones((1_000_000,), dtype=jnp.float32)
    y = jnp.ones((1_000_000,), dtype=jnp.float32)
    compiled = saxpy.lower(2.0, x, y).compile()
    print(compiled(2.0, x, y))


if __name__ == "__main__":
    main()
