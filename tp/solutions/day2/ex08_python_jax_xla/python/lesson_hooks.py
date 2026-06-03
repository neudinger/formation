from __future__ import annotations


def benchmark_commands() -> list[str]:
    return [
        "bazel run //src:vector_cpu_benchmark",
        "bazel run //src:vector_add_benchmark",
        "bazel run //src:matmul_benchmark",
    ]


def jax_status() -> str:
    try:
        import jax  # type: ignore
    except ModuleNotFoundError:
        return "JAX is not installed in this hermetic Python environment; keep this as a packaging exercise."
    return f"JAX is available: {jax.__version__}"
