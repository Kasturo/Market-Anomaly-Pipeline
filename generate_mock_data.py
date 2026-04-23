# """
# generate_mock_data.py
# =====================
# Generate realistic synthetic OHLCV market data for offline demo/testing.
# Useful when Yahoo Finance is unavailable or you want reproducible results.

# Produces data/{TICKER}_raw.csv for each configured ticker.

# Usage:
#   python generate_mock_data.py
#   python generate_mock_data.py --tickers AAPL TSLA NVDA --days 504
# """

# import argparse
# from datetime import datetime, timedelta
# from pathlib import Path

# import numpy as np
# import pandas as pd

# ROOT_DIR = Path(__file__).resolve().parent
# DATA_DIR = ROOT_DIR / "data"
# DATA_DIR.mkdir(parents=True, exist_ok=True)

# # Per-ticker simulation: start price, daily drift, daily volatility
# TICKER_PARAMS = {
#     "AAPL":  {"start": 175.0, "drift": 0.0003, "vol": 0.018},
#     "MSFT":  {"start": 380.0, "drift": 0.0004, "vol": 0.016},
#     "GOOGL": {"start": 140.0, "drift": 0.0002, "vol": 0.020},
#     "JPM":   {"start": 185.0, "drift": 0.0002, "vol": 0.015},
#     "SPY":   {"start": 470.0, "drift": 0.0003, "vol": 0.010},
#     "TSLA":  {"start": 250.0, "drift": 0.0002, "vol": 0.035},
#     "NVDA":  {"start": 800.0, "drift": 0.0006, "vol": 0.030},
#     "AMZN":  {"start": 185.0, "drift": 0.0003, "vol": 0.022},
#     "QQQ":   {"start": 440.0, "drift": 0.0003, "vol": 0.014},
#     "IWM":   {"start": 220.0, "drift": 0.0002, "vol": 0.016},
# }

# DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY"]
# DEFAULT_DAYS    = 365   # calendar days (produces ~252 trading days)
# BASE_VOLUME     = 50_000_000


# def simulate_ticker(
#     ticker:    str,
#     params:    dict,
#     dates:     pd.DatetimeIndex,
#     n_anomalies: int = 10,
#     seed:      int   = 42,
# ) -> pd.DataFrame:
#     """
#     Simulate OHLCV data using Geometric Brownian Motion with injected anomalies.

#     Anomalies are randomly chosen days where:
#       - Returns are amplified 4-9x in a random direction
#       - Volume is inflated 3-6x above baseline
#     """
#     rng = np.random.default_rng(seed)
#     n   = len(dates)

#     # Geometric Brownian Motion returns
#     returns = rng.normal(params["drift"], params["vol"], n)

#     # Inject anomalous days
#     anom_idx = rng.choice(n, size=min(n_anomalies, n), replace=False)
#     returns[anom_idx] += (
#         rng.choice([-1, 1], size=len(anom_idx))
#         * rng.uniform(0.04, 0.09, size=len(anom_idx))
#     )

#     # Price path
#     close = params["start"] * np.exp(np.cumsum(returns))
#     high  = close * (1 + np.abs(rng.normal(0, 0.008, n)))
#     low   = close * (1 - np.abs(rng.normal(0, 0.008, n)))
#     open_ = close * (1 + rng.normal(0, 0.005, n))

#     # Ensure OHLC consistency
#     high  = np.maximum(high,  np.maximum(open_, close))
#     low   = np.minimum(low,   np.minimum(open_, close))

#     # Volume: log-normal baseline plus spikes on anomaly days
#     volume = (BASE_VOLUME * rng.lognormal(0, 0.5, n)).astype(int)
#     volume[anom_idx] = (volume[anom_idx] * rng.integers(3, 7, size=len(anom_idx))).astype(int)

#     df = pd.DataFrame(
#         {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
#         index=dates,
#     )
#     df.index.name = "Date"
#     return df


# def generate_mock_data(
#     tickers:     list  = DEFAULT_TICKERS,
#     days:        int   = DEFAULT_DAYS,
#     end_date:    datetime = None,
#     n_anomalies: int   = 10,
# ) -> dict:
#     """
#     Generate and save mock OHLCV CSVs for the given tickers.
#     Returns a dict of {ticker: DataFrame}.
#     """
#     if end_date is None:
#         end_date = datetime.today()
#     start_date = end_date - timedelta(days=days)
#     dates = pd.bdate_range(start_date, end_date)   # business days only

#     print(f"\n[MOCK DATA] Generating {len(tickers)} tickers | "
#           f"{len(dates)} trading days | {start_date.date()} to {end_date.date()}")

#     result = {}
#     for i, ticker in enumerate(tickers):
#         if ticker not in TICKER_PARAMS:
#             print(f"  WARN {ticker}: no simulation params defined; using AAPL defaults")

#         params = TICKER_PARAMS.get(ticker, TICKER_PARAMS["AAPL"])
#         df     = simulate_ticker(ticker, params, dates, n_anomalies=n_anomalies, seed=i * 7 + 42)
#         path   = DATA_DIR / f"{ticker}_raw.csv"
#         df.to_csv(path)
#         result[ticker] = df
#         print(f"  OK {ticker:<6} {len(df):4d} rows  "
#               f"close: ${df['Close'].iloc[0]:.2f} to ${df['Close'].iloc[-1]:.2f}  "
#               f"anomalies injected: {n_anomalies}")

#     print(f"\n  Saved to {DATA_DIR}/\n")
#     return result


# def _parse_args():
#     parser = argparse.ArgumentParser(
#         prog="generate_mock_data",
#         description="Generate realistic synthetic OHLCV market data for offline demo/testing.",
#     )
#     parser.add_argument(
#         "--tickers", nargs="+", default=DEFAULT_TICKERS, metavar="TICKER",
#         help=f"Ticker symbols to generate (default: {' '.join(DEFAULT_TICKERS)}). "
#              f"Supported: {', '.join(TICKER_PARAMS.keys())}",
#     )
#     parser.add_argument(
#         "--days", type=int, default=DEFAULT_DAYS,
#         help="Calendar days of history to generate (default: 365, about 252 trading days)",
#     )
#     parser.add_argument(
#         "--anomalies", type=int, default=10,
#         help="Number of anomalous trading days to inject per ticker (default: 10)",
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = _parse_args()
#     generate_mock_data(
#         tickers     = [t.upper() for t in args.tickers],
#         days        = args.days,
#         n_anomalies = args.anomalies,
#     )
