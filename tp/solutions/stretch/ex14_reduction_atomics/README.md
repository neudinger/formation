# Exercise 14: Reduction and Atomics

## Goal

Sum a float vector in two ways: one global atomic update per element, then a
shared-memory reduction with one global atomic update per block. This is an
optional extension after Ex04 and Ex10; timing reuses the CUDA events from Ex06.
See CUDA Programming Guide §2.3.5, pp. 69–72 ([local PDF](../../../../cuda-programming-guide.pdf)).

## What to finish

Only the two kernels in `src/reduction.cu` need work. The host wrapper already
copies the input, initializes the result to zero, launches, and checks CUDA errors.

1. In `sum_atomic`, guard the input index and call `atomicAdd(sum, input[i])`.
   Plain `*sum += input[i]` is a read-modify-write race: simultaneous threads can
   read the same old value, then overwrite one another's additions.
2. In `sum_block`, stage one value per thread in a 256-float shared array. Use
   zero for out-of-range lanes, then synchronize. Reduce with strides 128, 64,
   ..., 1: participating lanes add their partner's partial sum, and **all** lanes
   synchronize after each step. Finally, thread 0 atomically adds the block sum.

Keep the block size at 256 for this power-of-two reduction. Do not return early
in the partial block: inactive lanes must still reach the barriers. A block
barrier cannot protect updates from other blocks, so the final update still
needs an atomic operation.

## Commands

Run inside this directory; it is an independent Bazel module.

```bash
bazel build //...
bazel test //src:reduction_test
bazel run //src:reduction_benchmark
```

This is the complete reference solution. Build, tests, and benchmark should pass.

Starter: `tp/stretch/ex14_reduction_atomics`.

Tests cover empty and single-element inputs, 255/256/257 elements, larger partial
blocks, mixed signs, zeros, and repeated calls. Both kernels use the same inputs
and are checked against a CPU sum.

## Compare the two kernels

The benchmark uses 1,048,576 elements, one warm-up per kernel, and five measured
runs. CUDA events report mean **kernel-only** time, excluding allocations, copies,
and result initialization. Every result is checked before reporting timings.

| Method | Global atomic updates | Mean kernel time (ms) |
| --- | ---: | ---: |
| Per-element atomic | 1,048,576 | measure locally |
| Block-local reduction | 4,096 | measure locally |

Explain why fewer updates to one global address can reduce contention, and what
the shared-memory tree costs in additions and barriers. Measure rather than
assuming a speedup for every input size or GPU.

Floating-point addition depends on order; atomics prevent lost updates but do not
make arbitrary float sums reproducible. These tests use bounded integer/quarter
values for exact comparisons. With arbitrary floats, compare against a double CPU
sum using a tolerance appropriate to the input and expected rounding error.
This teaching reduction is deliberately simple; production code can use CUB.

CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.
