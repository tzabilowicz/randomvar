import numpy as np
from typing import Sequence

def mean_difference(s1: Sequence, s2: Sequence) -> float:
    return np.mean(s1) - np.mean(s2)

def mean_ratio(s1: Sequence, s2: Sequence) -> float:
    return np.mean(s1) / np.mean(s2)

def variance_difference(s1: Sequence, s2: Sequence) -> float:
    return np.var(s1) - np.var(s2)

def varaince_ratio(s1: Sequence, s2: Sequence) -> float:
    return np.var(s1) / np.var(s2)

def median_difference(s1: Sequence, s2: Sequence) -> float:
    return np.median(s1) - np.median(s2)

def t_statistic(s1: Sequence, s2: Sequence):
    n1, n2 = len(s1), len(s2)
    m1, m2 = np.mean(s1), np.mean(s2)
    v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    
    se = np.sqrt(v1/n1 + v2/n2)
    return (m1 - m2) / se