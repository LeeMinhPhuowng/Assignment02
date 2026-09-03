"""Sklearn transformer: treat clinically impossible zeros as missing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ZeroMarker(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN on selected columns. Pregnancies is left unchanged."""

    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        present = [col for col in self.columns if col in frame.columns]
        frame[present] = frame[present].replace(0, np.nan)
        return frame
