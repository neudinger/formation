from __future__ import annotations

from python.lesson_hooks import benchmark_commands, jax_status


def test_benchmark_commands() -> None:
    assert benchmark_commands() == [
        "bazel run //src:vector_cpu_benchmark",
        "bazel run //src:vector_add_benchmark",
        "bazel run //src:matmul_benchmark",
    ]


def test_jax_status_is_informative() -> None:
    status = jax_status()
    assert "TODO" not in status
    assert "JAX" in status


if __name__ == "__main__":
    test_benchmark_commands()
    test_jax_status_is_informative()
