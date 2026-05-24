#!/usr/bin/env python3

"""Small benchmark runner used by the Day 2 slides.

The heavy measurements live in Bazel-built C++ binaries. This script keeps the
orchestration layer explicit without adding a non-hermetic Python package stack.
"""

from __future__ import annotations


def main() -> None:
    print("Run the Bazel-built benchmark binaries directly:")
    print("  bazel run //cpp:vector_cpu_benchmark")
    print("  bazel run //gpu:vector_add_benchmark")
    print("  bazel run //linear_algebra:matmul_benchmark")


if __name__ == "__main__":
    main()
