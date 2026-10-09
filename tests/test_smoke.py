import numpy as np
import pandas as pd
from scipy.stats import norm


def test_scientific_stack_imports() -> None:
    assert np.isclose(norm.cdf(0.0), 0.5)
    assert pd.Series([1, 2, 3]).sum() == 6
