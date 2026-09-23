# Benchmark Results

Record the machine, commands, timings, and interpretation here.

| Version | Time | Lesson |
| --- | ---: | --- |
| CPU | measure locally | correctness oracle |
| Basic CUDA | measure locally | one thread per output |
| Tiled CUDA | measure locally | shared-memory reuse |
| cuBLAS | measure locally | production library baseline |

## Memory access observations

| Question | Observation |
| --- | --- |
| Tiled A/B loads and output stores | Consecutive floats within each 16-thread row group; alignment affects transactions. |
| Basic kernel B reads at a fixed `inner` | Consecutive across x threads, despite each thread walking a column over time. |
| Main benefit of tiling | Each A/B value in a full tile is reused 16 times from shared memory. |
| `a[lane]` versus `a[lane * 32]` | For 32 lanes and aligned floats: 4 versus 32 separate 32-byte segments. |
| Optional transpose: why `[32][33]`? | Column stride 33 distributes float accesses across banks instead of hitting one bank. |
