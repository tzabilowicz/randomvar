import numpy as np
from typing import Sequence

def mean(s: Sequence):
    return np.mean(s)

def variance(s: Sequence):
    return np.var(s)

def standard_deviation(s: Sequence):
    return np.std(s)

def median(s: Sequence):
    return np.median(s)

def skew(s: Sequence):
    x = np.asarray(s)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma == 0:
        return 0.0

    return np.mean((x - mu) ** 3) / (sigma ** 3)

def kurtosis(s: Sequence):
    x = np.asarray(s)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma == 0:
        return 0.0

    k = np.mean((x - mu) ** 4) / (sigma ** 4)

    # Excess kurtosiss
    return k - 3