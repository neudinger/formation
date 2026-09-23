# Exercise 16: CUDA Graphs

Reuse Ex06's stream pattern with a tiny kernel: add 1 to each vector element.
One batch contains eight kernels. Compare direct launches with replaying a graph
containing the same eight kernels, all on one stream.

## Three TODOs

In `src/graph_replay.cu`:

1. Begin stream capture, call `launch_kernels()`, then end capture into `work.graph`.
2. Instantiate the graph into `work.executable` once.
3. Launch that executable on `work.stream` for each replay.

Capture records work; it does not run it. Keep buffers alive during replay.
Stream creation, synchronization, warm-up, and cleanup are supplied.

## Run

This directory is an independent Bazel project.

```bash
bazel build //...
bazel test //src:graph_replay_test
bazel run //src:graph_replay_benchmark
```

Complete solution. Starter: `tp/stretch/ex16_cuda_graphs`.

## Compare

The benchmark runs 1,000 batches on 4,096 elements: 8,000 kernel launches versus
1,000 graph launches, each executing eight kernels. Both results must equal 8,000.

| Path | Wall time for all batches (ms) |
| --- | ---: |
| Direct launches | measure locally |
| Graph replay | measure locally |

Timing includes CPU submission and waiting for GPU completion. Allocations,
copies, graph capture/instantiation, first-launch warm-up, and validation are
outside the timed interval. Graph setup still costs time and must be amortized.
Try fewer batches or a larger vector: graphs can reduce launch overhead but do
not reduce the work inside each kernel. A speedup is not guaranteed.

Requires a CUDA-capable GPU. See the
[CUDA Programming Guide, §2.5.9.2 and §4.2](../../../../cuda-programming-guide.pdf).
