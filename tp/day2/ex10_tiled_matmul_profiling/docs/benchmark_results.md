# Benchmark Results

Record the machine, commands, timings, and interpretation here.

| Version | Time | Lesson |
| --- | ---: | --- |
| CPU | TODO | correctness oracle |
| Basic CUDA | TODO | one thread per output |
| Tiled CUDA | TODO | shared-memory reuse |
| cuBLAS | TODO | production library baseline |

## Memory access observations

| Question | Observation |
| --- | --- |
| Tiled A/B loads and output stores | TODO: trace neighboring x threads. |
| Basic kernel B reads at a fixed `inner` | TODO: contiguous or strided across threads? |
| Main benefit of tiling | TODO: how many times is each full-tile value reused? |
| `a[lane]` versus `a[lane * 32]` | TODO: count 32-byte segments for 32 lanes and aligned floats. |
| Optional transpose: why `[32][33]`? | TODO if attempted: explain bank mapping and record kernel timings. |
