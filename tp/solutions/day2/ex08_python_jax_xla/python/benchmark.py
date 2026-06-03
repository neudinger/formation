from __future__ import annotations

from python.lesson_hooks import benchmark_commands


def main() -> None:
    print("Run the Bazel-built benchmark binaries directly:")
    for command in benchmark_commands():
        print(f"  {command}")


if __name__ == "__main__":
    main()
