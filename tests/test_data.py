from pathlib import Path

import pandas as pd
import pytest

from quantlab.data import load_prices

DATA = Path(__file__).parent / "data"


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_loads_tiny_csv() -> None:
    prices = load_prices(DATA / "tiny_prices.csv")
    assert isinstance(prices, pd.Series)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.dtype == "float64"
    assert len(prices) == 5
    assert prices.iloc[0] == 100.0
    assert prices.iloc[-1] == 103.02


def test_sorts_unsorted_dates(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,adj_close\n2024-01-03,102\n2024-01-02,100\n")
    prices = load_prices(f)
    assert prices.index.is_monotonic_increasing
    assert prices.iloc[0] == 100.0


def test_drops_nan_prices(tmp_path: Path) -> None:
    f = write_csv(
        tmp_path / "p.csv",
        "date,adj_close\n2024-01-02,100\n2024-01-03,\n2024-01-04,101\n",
    )
    prices = load_prices(f)
    assert len(prices) == 2
    assert not prices.isna().any()


def test_ignores_extra_columns(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,adj_close,volume\n2024-01-02,100,5\n2024-01-03,101,6\n")
    assert len(load_prices(f)) == 2


def test_custom_column(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,close\n2024-01-02,100\n2024-01-03,101\n")
    assert load_prices(f, column="close").iloc[1] == 101.0


def test_fewer_than_two_rows_raises(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,adj_close\n2024-01-02,100\n")
    with pytest.raises(ValueError):
        load_prices(f)


def test_all_nan_raises(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,adj_close\n2024-01-02,\n2024-01-03,\n")
    with pytest.raises(ValueError):
        load_prices(f)


def test_duplicate_dates_raise(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,adj_close\n2024-01-02,100\n2024-01-02,101\n")
    with pytest.raises(ValueError):
        load_prices(f)


def test_missing_column_raises(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,close\n2024-01-02,100\n2024-01-03,101\n")
    with pytest.raises(ValueError):
        load_prices(f)


def test_integer_prices_become_float(tmp_path: Path) -> None:
    f = write_csv(tmp_path / "p.csv", "date,adj_close\n2024-01-02,100\n2024-01-03,101\n")
    assert load_prices(f).dtype == "float64"
