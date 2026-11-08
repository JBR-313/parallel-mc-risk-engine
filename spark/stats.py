"""Market-wide return statistics and the portfolio covariance matrix with PySpark.

Input : data/raw/prices/ (Parquet, every KRX stock, partitioned by year)
Output: data/stats/returns.parquet         clean daily log returns, all stocks
        data/stats/universe.parquet        per-stock annualised mu / sigma, market cap, coverage
        data/stats/params.{json,h5,txt}    S0, mu, cov for the top-N portfolio (C reads .txt)

Steps
 1. Log returns per stock with a window function (lag over date).
 2. Cleaning on unadjusted prices: a return is dropped if listed shares changed
    by more than 10 % that day (split / bonus issue / large rights issue) or if it lies outside the KRX
    +/-30 % daily limit [ln 0.7, ln 1.3], which no genuine move can do.
 3. Per-stock statistics for the whole market (groupBy).
 4. Portfolio = top-N stocks by latest market cap among those that trade on
    >= --min-coverage of all days. Covariance is computed as a distributed
    self-join on date: (date, i, r_i) x (date, j, r_j), i <= j, then
    groupBy(i, j).sum(r_i r_j) -> cov_ij = (S_ij - S_i S_j / N) / (N - 1),
    using only dates on which all N stocks have a clean return (so the
    matrix is a proper sample covariance, hence positive semi-definite).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS = 252
LIMIT_LO, LIMIT_HI = math.log(0.7), math.log(1.3)
SHARE_CHANGE_TOL = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/raw/prices"))
    ap.add_argument("--outdir", default=str(ROOT / "data/stats"))
    ap.add_argument("--top", type=int, default=200, help="portfolio size N")
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--driver-memory", default="3g")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    spark = (
        SparkSession.builder.master(args.master)
        .appName("mc-risk-stats")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    prices = spark.read.parquet(args.input).where(F.col("close") > 0)
    n_rows = prices.count()
    n_days = prices.select("date").distinct().count()

    # 1-2. returns + cleaning
    w = Window.partitionBy("ticker").orderBy("date")
    rets = (
        prices.withColumn("prev_close", F.lag("close").over(w))
        .withColumn("prev_shares", F.lag("shares").over(w))
        .where(F.col("prev_close").isNotNull())
        .withColumn("r", F.log(F.col("close") / F.col("prev_close")))
        # Small share-count changes (CB conversions, stock options) are frequent and
        # do not move the price; only a >10 % change marks a split / bonus issue /
        # large rights offering. Big ones are also caught by the price-limit test.
        .withColumn("corp_action",
                    F.abs(F.col("shares") / F.col("prev_shares") - 1) > SHARE_CHANGE_TOL)
        .withColumn("out_of_limit", (F.col("r") < LIMIT_LO) | (F.col("r") > LIMIT_HI))
    ).cache()
    flags = rets.agg(F.count("*").alias("n"),
                     F.sum(F.col("corp_action").cast("int")).alias("corp"),
                     F.sum((F.col("out_of_limit") & ~F.col("corp_action")).cast("int")).alias("limit")).first()
    clean = (rets.where(~F.col("corp_action") & ~F.col("out_of_limit"))
             .select("date", "ticker", "r").cache())
    clean.write.mode("overwrite").parquet(str(out / "returns.parquet"))

    # 3. per-stock statistics, whole market
    last_date = prices.agg(F.max("date")).first()[0]
    latest = prices.where(F.col("date") == last_date).select("ticker", "name", "market", "close", "mktcap")
    per = clean.groupBy("ticker").agg(
        F.count("r").alias("n_ret"),
        F.mean("r").alias("mean_d"),
        F.stddev_samp("r").alias("std_d"),
    )
    universe = (
        per.join(latest, "ticker", "inner")
        .withColumn("sigma", F.col("std_d") * math.sqrt(TRADING_DAYS))
        .withColumn("mu", F.col("mean_d") * TRADING_DAYS + 0.5 * F.col("sigma") ** 2)
        .withColumn("coverage", F.col("n_ret") / F.lit(n_days - 1))
    ).cache()
    universe.write.mode("overwrite").parquet(str(out / "universe.parquet"))
    by_market = (universe.groupBy("market")
                 .agg(F.count("*").alias("stocks"),
                      F.expr("percentile_approx(sigma, 0.5)").alias("median_sigma"),
                      (F.sum("mktcap") / 1e12).alias("mktcap_tn_krw"))
                 .orderBy("market").collect())

    # 4. portfolio selection and covariance
    sel = (universe.where(F.col("coverage") >= args.min_coverage)
           .orderBy(F.col("mktcap").desc()).limit(args.top)
           .select("ticker", "name", "close", "mktcap").collect())
    tickers = sorted(r.ticker for r in sel)
    info = {r.ticker: r for r in sel}
    n = len(tickers)
    idx = spark.createDataFrame([(t, i) for i, t in enumerate(tickers)], "ticker STRING, i INT")

    port = clean.join(F.broadcast(idx), "ticker")
    full_dates = port.groupBy("date").count().where(F.col("count") == n).select("date")
    port = port.join(full_dates, "date").select("date", "i", "r").cache()
    N = full_dates.count()

    a, b = port.alias("a"), port.alias("b")
    pairs = (
        a.join(b, (F.col("a.date") == F.col("b.date")) & (F.col("a.i") <= F.col("b.i")))
        .groupBy(F.col("a.i").alias("i"), F.col("b.i").alias("j"))
        .agg(F.sum(F.col("a.r") * F.col("b.r")).alias("p"))
    )
    sums = dict(port.groupBy("i").agg(F.sum("r")).collect())
    s = np.array([sums[i] for i in range(n)])
    cov_d = np.empty((n, n))
    for row in pairs.collect():
        i, j = row.i, row.j
        cov_d[i, j] = cov_d[j, i] = (row.p - s[i] * s[j] / N) / (N - 1)
    n_pair_rows = N * n * (n + 1) // 2

    # Cross-check against a local numpy computation on the aligned matrix.
    wide = port.groupBy("date").pivot("i", list(range(n))).agg(F.first("r")).toPandas()
    local = wide[[str(i) for i in range(n)]].to_numpy()
    assert local.shape == (N, n)
    assert np.allclose(np.cov(local, rowvar=False), cov_d, rtol=1e-8, atol=1e-14), "cov mismatch"

    mean_d = s / N
    cov = cov_d * TRADING_DAYS
    sigma = np.sqrt(np.diag(cov))
    mu = mean_d * TRADING_DAYS + 0.5 * sigma**2  # E[log r] = (mu - sigma^2/2) dt
    corr = cov / np.outer(sigma, sigma)
    eig = np.linalg.eigvalsh(cov)
    np.linalg.cholesky(cov)  # fails loudly if not positive definite
    s0 = np.array([info[t].close for t in tickers])

    params = {
        "tickers": tickers,
        "names": [info[t].name for t in tickers],
        "S0": s0.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(),
        "cov": cov.tolist(), "corr": corr.tolist(),
        "n_obs": int(N), "trading_days": TRADING_DAYS, "as_of": str(last_date),
        "cond_number": float(eig[-1] / eig[0]),
    }
    (out / "params.json").write_text(json.dumps(params, ensure_ascii=False))

    import h5py
    with h5py.File(out / "params.h5", "w") as h:
        h.attrs.update(n_obs=int(N), trading_days=TRADING_DAYS, as_of=str(last_date))
        h["tickers"] = np.array(tickers, dtype="S")
        for k, v in (("S0", s0), ("mu", mu), ("sigma", sigma), ("cov", cov), ("corr", corr)):
            h[k] = np.asarray(v, dtype=np.float64)

    with open(out / "params.txt", "w") as f:
        f.write(f"{n}\n")
        for i, t in enumerate(tickers):
            f.write(f"{t} {s0[i]:.17g} {mu[i]:.17g}\n")
        for i in range(n):
            f.write(" ".join(f"{c:.17g}" for c in cov[i]) + "\n")

    summary = {
        "input_rows": n_rows, "trading_days": n_days, "returns": flags.n,
        "dropped_corporate_action": flags.corp, "dropped_out_of_limit": flags.limit,
        "universe_stocks": universe.count(), "portfolio_size": n, "aligned_days": N,
        "self_join_pair_rows": n_pair_rows, "cond_number": params["cond_number"],
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "by_market": [r.asDict() for r in by_market],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    print(f"{n_rows:,} price rows, {n_days} trading days, {flags.n:,} returns "
          f"({flags.corp} corporate-action and {flags.limit} out-of-limit returns dropped)")
    for r in by_market:
        print(f"  {r.market:7} {r.stocks:5} stocks  median sigma {r.median_sigma:.2f}  "
              f"mktcap {r.mktcap_tn_krw:,.0f} tn KRW")
    print(f"portfolio: top {n} by market cap, {N} aligned days, "
          f"self-join {n_pair_rows:,} pair rows, cond(cov) = {params['cond_number']:.0f}")
    print(f"sigma: min {sigma.min():.2f} median {np.median(sigma):.2f} max {sigma.max():.2f}; "
          f"mean pairwise corr {corr[np.triu_indices(n, 1)].mean():.2f}")
    print(f"done in {summary['elapsed_s']}s")
    spark.stop()


if __name__ == "__main__":
    main()
