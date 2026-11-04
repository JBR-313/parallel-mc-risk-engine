# Parallel Monte Carlo Risk Engine

Final project for DCCS 410 Bigdata Distributed Processing System (Korea University).

Estimates portfolio Value-at-Risk and European option prices with correlated geometric Brownian motion, and measures the speedup from OpenMP, MPI, and JAX.

## Pipeline

1. `ingest/`: collect daily prices of listed Korean stocks through the data.go.kr OpenAPI.
2. `spark/`: compute returns, volatilities, and the covariance matrix.
3. `sim/`: serial, OpenMP, MPI, and JAX simulators.
4. `bench/`: speedup, efficiency, and strong/weak scaling runs.
5. `results/`: benchmark tables and figures.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your data.go.kr API key
```

## Status

Work in progress. Target milestones:

| Date | Milestone |
| --- | --- |
| Nov 7 | OpenAPI ingestion, Spark statistics, serial simulator validated against Black–Scholes |
| Nov 14 | OpenMP version and benchmarks |
| Nov 21 | MPI version and benchmarks |
| Dec 5 | JAX version, final benchmarks, slides |
| Dec 8 | Presentation |
| Dec 16 | Final report |
