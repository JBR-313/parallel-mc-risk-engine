Real data, top-200 portfolio, 100,000 paths x 10 steps (strong) / 10,000 paths per worker (weak); median of 3 runs. Serial C baseline: 11.38 s.

| Implementation | 1 worker | 8 workers | 16 workers | Speedup vs serial C (8 / 16) | Strong eff. (8 / 16) | Weak eff. (8 / 16) |
| --- | --- | --- | --- | --- | --- | --- |
| OpenMP (C) | 11.44 s | 1.88 s | 1.37 s | 6.1x / 8.3x | 0.76 / 0.52 | 0.78 / 0.58 |
| MPI (C) | 11.39 s | 1.86 s | 1.31 s | 6.1x / 8.7x | 0.76 / 0.54 | 0.78 / 0.56 |
| JAX, threefry PRNG | 6.43 s | 2.89 s | 2.78 s | 3.9x / 4.1x | 0.28 / 0.14 | 0.29 / 0.16 |
| JAX, rbg PRNG | 11.25 s | 2.34 s | 1.71 s | 4.9x / 6.6x | 0.60 / 0.41 | 0.61 / 0.43 |
