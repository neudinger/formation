# Exercise 10: Tiled MatMul and Profiling Notes

## Goal

Use shared memory tiling to improve MatMul and record benchmark/profiling observations.

## What to finish

- Load A and B tiles into `__shared__` memory.
- Use `__syncthreads()` around tile reuse.
- Explain coalescing and shared-memory reuse using the access trace below.
- Fill in the benchmark notes worksheet.

## Commands

```bash
bazel build //...
bazel test //src:matmul_test
bazel run //src:matmul_benchmark
```

This is the complete reference solution. It should build and pass its tests.

Starter exercise: `tp/day2/ex10_tiled_matmul_profiling`.

## Memory access patterns

Read CUDA Programming Guide §2.3.4, pp. 58–69 ([local PDF](../../../../cuda-programming-guide.pdf)).
Matrices here use row-major storage: `matrix[row * width + col]`.
Coalescing combines nearby addresses requested by threads in one warp into fewer
global-memory transactions. Follow addresses across threads at one instruction,
not just the sequence of addresses visited by one thread.

### Trace the tiled MatMul

In `src/matmul_tiled_device.cu`, `threadIdx.x` selects the column. With a
16 × 16 block, a warp covers two groups of 16 threads on consecutive rows.
Within each group, increasing `threadIdx.x` by one gives:

| Access | Address | Pattern across neighboring x threads |
| --- | --- | --- |
| Load A tile | `a[row * k + tile * 16 + threadIdx.x]` | consecutive floats |
| Load B tile | `b[(tile * 16 + threadIdx.y) * n + col]` | consecutive floats |
| Store output | `out[row * n + col]` | consecutive floats |

These contiguous row segments allow coalescing; row alignment and matrix widths
influence the number of transactions. Each full tile loads 256 values from A and
256 from B, then reuses each value for 16 output calculations in shared memory.
The first barrier makes the loaded tiles visible to the block; the second keeps
threads from overwriting them before all threads finish using them. Zero-fill
out-of-range tile entries, and keep both barriers outside per-thread bounds checks.

Compare with `src/matmul_basic_device.cu`: at a fixed `inner`, neighboring x
threads read the same A value and consecutive B values. Although one thread walks
down a column of B over time, the warp's B reads are contiguous at each instruction.
Tiling adds explicit shared-memory reuse; it does not fix strided B reads in this
basic kernel.

For a quick address experiment, take 32 threads with lane numbers 0–31 and an
aligned float array. Compare `a[lane]` with `a[lane * 32]`: both request 128 useful
bytes, but touch 4 versus 32 separate 32-byte segments. Record this and the MatMul
access patterns in [the worksheet](docs/benchmark_results.md).

The current benchmark times complete calls, including allocations and transfers.
Use warm-up runs and Ex06's CUDA events around kernel launches if you want to
measure the kernel effect separately; small inputs may hide any benefit.

### Optional: transpose and shared-memory padding

Use the guide's transpose example as a short extension within this Bazel project:

1. Transpose a row-major matrix with `out[col * height + row] = in[row * width + col]`.
   Map x threads to input columns: reads are contiguous, while writes are strided.
2. Use a 32 × 32 thread block to stage a 32 × 32 tile in shared memory.
   Synchronize and transpose the shared tile while swapping block coordinates
   for output, so that both global reads and writes are contiguous.
3. Compare `__shared__ float tile[32][32]` with `tile[32][33]`. For float data,
   shared memory has 32 banks: a column access `tile[threadIdx.x][fixed_col]`
   hits one bank with stride 32, but different banks with stride 33. Padding
   removes this column-access conflict; it does not change the output shape.
4. Check both versions against a CPU transpose, including a non-square size such
   as 1003 × 517. Guard boundary loads/stores and let every thread reach the
   barrier. Warm up and compare kernel-only CUDA event timings at a fixed size.

Padding addresses shared-memory bank conflicts; coalescing concerns global-memory
transactions. Apply padding when the actual access pattern needs it, rather than
assuming it improves the existing 16 × 16 MatMul.

## Notes

- CUDA tests and benchmarks require an NVIDIA driver and CUDA-capable GPU.

