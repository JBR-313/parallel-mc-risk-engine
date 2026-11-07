# Parallel Monte Carlo Risk Engine

Final project for DCCS 410 Bigdata Distributed Processing System (Korea University).

Estimates portfolio Value-at-Risk for a top-200 KRX portfolio and European option prices with correlated geometric Brownian motion, and measures the speedup from OpenMP, MPI, and JAX.

## Pipeline

1. `ingest/`: collect daily prices of every KRX-listed stock (~2,500 stocks x ~750 days, ~1.9 M rows) through the data.go.kr OpenAPI, one request per trading day, cached with a SHA-256 manifest.
2. `spark/`: log returns and corporate-action cleaning for the whole market, per-stock statistics, and the 200 x 200 covariance of the top-200 stocks by market cap via a distributed self-join (~10 M pair rows).
3. `sim/`: serial, OpenMP, MPI (C) and NumPy/JAX simulators; `validate.py` checks them.
4. `bench/`: speedup, efficiency, and strong/weak scaling runs.
5. `results/`: benchmark tables and figures.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your data.go.kr API key (금융위원회_주식시세정보)
```

MPI needs a system toolchain: `sudo apt install libopenmpi-dev openmpi-bin`, then `pip install mpi4py`.

## Running

```bash
python ingest/fetch_prices.py          # real data (needs key); add --synthetic for offline GBM data
python spark/stats.py                  # returns, mu, sigma, covariance -> data/stats/
make -C sim                            # build/mc_serial, build/mc_omp, build/mc_mpi
python sim/validate.py                 # Black-Scholes, closed-form VaR, OpenMP == serial, C vs NumPy
python bench/scaling.py --impl omp     # strong/weak scaling -> results/
python bench/scaling.py --impl mpi
```

Simulator CLI (all implementations share it):

```bash
./build/mc_omp --mode var --paths 4000000 --steps 10 --T 0.0397 --value 1e8   # 10-day VaR/CVaR
./build/mc_serial --mode option --paths 1000000 --steps 52 --T 1 --r 0.03 --K 1.0
```

Path `p` always draws from RNG stream `(seed, p)`, so serial, OpenMP and MPI runs simulate
exactly the same paths and parallel VaR results are bit-identical to serial.

## Status

| Date | Milestone | State |
| --- | --- | --- |
| Nov 7 | OpenAPI ingestion, Spark statistics, serial simulator validated against Black–Scholes | done (real-data run pending API key) |
| Nov 8 | Scope extended to the full KRX universe (top-200 portfolio) | done |
| Nov 14 | OpenMP version and benchmarks | done: 6.1x on 8 threads, 8.8x on 16 (Ryzen 7 7840HS, 8C/16T) |
| Nov 21 | MPI version and benchmarks | done: 8.0x on 16 ranks, communication < 1 % of wall time |
| Dec 5 | JAX version, final benchmarks, slides | |
| Dec 8 | Presentation | |
| Dec 16 | Final report | |
