"""Collect daily prices of every KRX-listed stock from the data.go.kr OpenAPI.

API: 금융위원회_주식시세정보 (GetStockSecuritiesInfoService_V2/getStockPriceInfo_V2).
One request per trading day returns every listed stock (KOSPI, KOSDAQ, KONEX),
so three years is ~750 requests and ~1.9 M rows.

Reproducibility (Week 5 pattern):
  * every response is cached (gzip) under data/raw/cache/<basDt>_<page>.json.gz, so an
    interrupted run resumes where it stopped and re-runs make no requests,
  * data/ingest_manifest.json records the request parameters (key redacted),
    the SHA-256 of every page's JSON and of the output files.

Output: data/raw/prices/year=YYYY/*.parquet (partitioned by year)

Usage:
  python ingest/fetch_prices.py                     # real API, needs DATA_GO_KR_API_KEY in .env
  python ingest/fetch_prices.py --synthetic         # offline data with the same schema
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = RAW / "cache"
OUT = RAW / "prices"
MANIFEST = DATA / "ingest_manifest.json"

# V2 endpoint (the old .../service/GetStockSecuritiesInfoService/getStockPriceInfo
# path rejects keys issued for V2 with SERVICE_KEY_IS_NOT_REGISTERED_ERROR).
ENDPOINT = (
    "https://apis.data.go.kr/1160100/"
    "GetStockSecuritiesInfoService_V2/getStockPriceInfo_V2"
)
ROWS_PER_PAGE = 5000  # > number of listed stocks, so normally one page per day

SCHEMA = {
    "date": "datetime64[ns]", "ticker": "string", "name": "string", "market": "string",
    "open": "float64", "high": "float64", "low": "float64", "close": "float64",
    "volume": "int64", "shares": "int64", "mktcap": "float64",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class ApiError(RuntimeError):
    pass


def fetch_page(session: requests.Session, key: str, day: str, page: int, retries: int = 4):
    """Fetch one page for one day, using the on-disk cache. Returns (doc, sha256, cached)."""
    cache_file = CACHE / f"{day}_{page}.json.gz"
    if cache_file.exists():
        body = gzip.decompress(cache_file.read_bytes())
        return json.loads(body), sha256_bytes(body), True

    params = {"resultType": "json", "numOfRows": ROWS_PER_PAGE, "pageNo": page, "basDt": day}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(ENDPOINT, params={**params, "serviceKey": key}, timeout=60)
            # data.go.kr may answer HTTP 200 with an XML error body (e.g. bad key),
            # so check both the status code and the body's resultCode.
            if r.status_code != 200:
                raise ApiError(f"HTTP {r.status_code}: {r.text[:200]}")
            try:
                doc = r.json()
            except ValueError:
                raise ApiError(f"non-JSON response (likely auth error): {r.text[:300]}")
            header = doc.get("response", {}).get("header", {})
            if header.get("resultCode") != "00":
                raise ApiError(f"resultCode={header.get('resultCode')} {header.get('resultMsg')}")
            body = json.dumps(doc, ensure_ascii=False, sort_keys=True).encode()
            tmp = cache_file.with_name(cache_file.name + ".tmp")
            tmp.write_bytes(gzip.compress(body, mtime=0))  # mtime=0: deterministic bytes
            tmp.replace(cache_file)  # atomic: no half-written cache entries
            return doc, sha256_bytes(body), False
        except (requests.RequestException, ApiError) as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise ApiError(f"{day} page {page}: failed after {retries} attempts: {last_err}")


def items_of(doc: dict) -> list[dict]:
    items = doc["response"]["body"].get("items") or {}
    item = items.get("item", []) if isinstance(items, dict) else []
    return [item] if isinstance(item, dict) else list(item)


_local = threading.local()


def fetch_day(key, day: str):
    session = getattr(_local, "session", None)
    if session is None:  # requests.Session is not guaranteed thread-safe: one per thread
        session = _local.session = requests.Session()
    records, pages, page = [], [], 1
    while True:
        doc, digest, cached = fetch_page(session, key, day, page)
        total = int(doc["response"]["body"].get("totalCount", 0))
        its = items_of(doc)
        pages.append({"basDt": day, "pageNo": page, "sha256": digest, "n_items": len(its)})
        records.extend(its)
        if page * ROWS_PER_PAGE >= total or not its:
            return records, pages, cached
        page += 1


def fetch_day_frame(key, day: str):
    records, pages, cached = fetch_day(key, day)
    return (to_frame(records) if records else None), pages, cached


def to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["basDt"], format="%Y%m%d"),
        "ticker": df["srtnCd"].astype(str).str.lstrip("A"),
        "name": df["itmsNm"],
        "market": df["mrktCtg"],
        "open": pd.to_numeric(df["mkp"], errors="coerce"),
        "high": pd.to_numeric(df["hipr"], errors="coerce"),
        "low": pd.to_numeric(df["lopr"], errors="coerce"),
        "close": pd.to_numeric(df["clpr"], errors="coerce"),
        "volume": pd.to_numeric(df["trqu"], errors="coerce").fillna(0).astype("int64"),
        "shares": pd.to_numeric(df["lstgStCnt"], errors="coerce").fillna(0).astype("int64"),
        "mktcap": pd.to_numeric(df["mrktTotAmt"], errors="coerce"),
    })
    return out


def synthetic(begin: str, end: str, n_stocks: int, seed: int) -> pd.DataFrame:
    """Factor-model market with the same schema as the API output.

    r_it = beta_i * market_t + gamma_i * sector_{s(i),t} + idio_it, with
    listings / delistings (ragged histories), occasional trading halts and a
    few stock splits (unadjusted prices), so the Spark cleaning steps are exercised.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(pd.to_datetime(begin), pd.to_datetime(end))
    T, n, n_sec = len(dates), n_stocks, 20
    dtau = 1 / 252
    mkt = rng.normal(0.06 * dtau, 0.18 * np.sqrt(dtau), T)
    sec = rng.normal(0.0, 0.12 * np.sqrt(dtau), (T, n_sec))
    sector = rng.integers(0, n_sec, n)
    beta = rng.uniform(0.5, 1.5, n)
    gamma = rng.uniform(0.3, 1.2, n)
    idio_vol = rng.uniform(0.15, 0.6, n)
    r = (mkt[:, None] * beta + sec[:, sector] * gamma
         + rng.normal(0, 1, (T, n)) * idio_vol * np.sqrt(dtau))
    r = np.clip(r, np.log(0.7) + 1e-6, np.log(1.3) - 1e-6)   # KRX +/-30 % limit
    shares = rng.lognormal(16.5, 1.2, n).astype("int64")
    s0 = np.exp(rng.uniform(np.log(1_000), np.log(500_000), n))
    logp = np.log(s0) + np.cumsum(r, axis=0)
    close = np.exp(logp)

    # 2 % of names split 1:5 once (price / 5, shares x 5 on unadjusted data).
    split = rng.random(n) < 0.02
    split_day = rng.integers(T // 4, T, n)
    for i in np.where(split)[0]:
        close[split_day[i]:, i] /= 5

    # Ragged histories: 15 % list late, 5 % delist early; 0.2 % random halts.
    alive = np.ones((T, n), bool)
    late = rng.random(n) < 0.15
    alive[: , late] &= np.arange(T)[:, None] >= rng.integers(1, T // 2, late.sum())
    early = rng.random(n) < 0.05
    alive[:, early] &= np.arange(T)[:, None] < rng.integers(T // 2, T, early.sum())
    alive &= rng.random((T, n)) > 0.002

    market = np.where(rng.random(n) < 0.35, "KOSPI", "KOSDAQ")
    tick = np.array([f"{100000 + 7 * i:06d}"[-6:] for i in range(n)])
    names = np.array([f"SYN{i:04d}" for i in range(n)])
    ti, si = np.nonzero(alive)
    c = np.round(close[ti, si])
    sh = shares[si] * np.where(split[si] & (ti >= split_day[si]), 5, 1)
    return pd.DataFrame({
        "date": dates[ti], "ticker": tick[si], "name": names[si], "market": market[si],
        "open": c, "high": c, "low": c, "close": c,
        "volume": rng.integers(10_000, 5_000_000, len(ti)),
        "shares": sh, "mktcap": c * sh,
    })


def write_parquet(df: pd.DataFrame) -> list[dict]:
    if OUT.exists():
        shutil.rmtree(OUT)
    df = df.astype(SCHEMA).sort_values(["date", "ticker"]).reset_index(drop=True)
    files = []
    for year, part in df.groupby(df["date"].dt.year):
        d = OUT / f"year={year}"
        d.mkdir(parents=True)
        f = d / "part-0.parquet"
        # date32 (not ns timestamps), which Spark reads as DATE
        part.assign(date=part["date"].dt.date).to_parquet(f, index=False)
        files.append({"path": str(f.relative_to(ROOT)), "rows": len(part),
                      "sha256": sha256_bytes(f.read_bytes())})
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--begin", default="20230102", help="YYYYMMDD (inclusive)")
    ap.add_argument("--end", default="20261106", help="YYYYMMDD (inclusive)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    ap.add_argument("--synthetic", action="store_true", help="generate offline data instead of calling the API")
    ap.add_argument("--n-stocks", type=int, default=2500, help="synthetic universe size")
    ap.add_argument("--seed", type=int, default=410)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "synthetic" if args.synthetic else ENDPOINT,
        "begin": args.begin,
        "end": args.end,
    }
    t0 = time.perf_counter()

    if args.synthetic:
        df = synthetic(args.begin, args.end, args.n_stocks, args.seed)
        manifest.update(seed=args.seed, n_stocks=args.n_stocks)
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        key = os.environ.get("DATA_GO_KR_API_KEY", "").strip()
        if not key:
            print("DATA_GO_KR_API_KEY is not set (.env). Use --synthetic for offline data.", file=sys.stderr)
            return 2
        CACHE.mkdir(parents=True, exist_ok=True)
        days = [d.strftime("%Y%m%d") for d in pd.bdate_range(args.begin, args.end)]
        frames, pages, n_net, done = [], [], 0, 0
        with ThreadPoolExecutor(args.workers) as ex:
            # Workers return compact typed frames, and each finished future is
            # dropped right away: holding the raw JSON records of every day
            # until the pool closes peaked at ~4.6 GB RSS for 1,000 days.
            futs = {ex.submit(fetch_day_frame, key, d): d for d in days}
            for fut in as_completed(futs):
                frame, pg, cached = fut.result()
                del futs[fut]
                pages.extend(pg)
                n_net += not cached
                if frame is not None:  # holidays return no items
                    frames.append(frame)
                done += 1
                if done % 50 == 0 or done == len(days):
                    print(f"  {done}/{len(days)} days ({n_net} fetched from network)", flush=True)
        df = pd.concat(frames, ignore_index=True)
        manifest["requests"] = {"days": len(days), "trading_days": len(frames), "network": n_net,
                                "pages": sorted(pages, key=lambda p: (p["basDt"], p["pageNo"]))}

    df = df.dropna(subset=["close"]).drop_duplicates(["date", "ticker"])
    manifest["output"] = {"rows": len(df), "tickers": int(df["ticker"].nunique()),
                          "dates": int(df["date"].nunique()), "files": write_parquet(df)}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}/: {len(df):,} rows, {df['ticker'].nunique():,} tickers, "
          f"{df['date'].nunique()} days in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
