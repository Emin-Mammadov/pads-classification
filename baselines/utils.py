# utils.py
import numpy as np, pandas as pd
from scipy import sparse

def balanced_weights(y, gender):
    g = pd.Series(1, index=y).groupby([y, gender]).size()
    tgt = g.mean()
    return np.array([tgt / g[(yi, gi)] for yi, gi in zip(y, gender)])

def to_dense_if_needed(X):
    return X.toarray() if sparse.issparse(X) else X
