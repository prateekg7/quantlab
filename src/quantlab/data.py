from pathlib import Path

import pandas as pd


def load_prices(path: str | Path, column: str = "adj_close") -> pd.Series:
    """Load a price series from a CSV with a `date` column."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")

    if column not in df.columns:
        raise ValueError(f"CSV must contain the column '{column}'.")
    prices = df[column].dropna().sort_index().astype("float64")
    if len(prices) < 2:
        raise ValueError("CSV must contain at least two rows of data.")
    if prices.index.has_duplicates:
        raise ValueError("CSV must contain unique dates.")
    return prices
