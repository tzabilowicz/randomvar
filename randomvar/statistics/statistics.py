""" One and Two Sample Statistics"""

import numpy as np
from typing import Sequence

# One sample statistics
def mean(s: Sequence) -> float:
    """ 
    Sample mean. 
    
    Parameters
    ----------
    s : Sequence
        Sample data.
        
    Returns
    -------
    sample_mean : float
    """
    
    return np.mean(s)

def variance(s: Sequence) -> float:
    """
    Sample variance.
    
    Parameters
    ----------
    s : Sequence
        Sample data.
        
    Returns
    -------
    sample_variance : float
    """
    
    return np.var(s)

def standard_deviation(s: Sequence) -> float:
    """
    Sample standard deviation.
    
    Parameters
    ----------
    s : Sequence
        Sample data.
        
    Returns
    -------
    sample_sd : float
    """
    
    return np.std(s)

def median(s: Sequence) -> float:
    """ 
    Sample median.
    
    Parameters
    ----------
    s : Sequence
        Sample data.
        
    Returns
    -------
    sample_median : float
    """
    
    return np.median(s)

def skew(s: Sequence) -> float:
    """ 
    Sample skew.
    
    Parameters
    ----------
    s : Sequence
        Sample data.
    
    Returns
    -------
    sample_skew : float
    """
    
    x = np.asarray(s)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma == 0:
        return 0.0

    return np.mean((x - mu) ** 3) / (sigma ** 3)

def kurtosis(s: Sequence) -> float:
    """
    Sample kurtosis.
    
    Parameters
    ----------
    s : Sequence
        Sample data.
    
    Returns
    -------
    sample_kurtosis : float
    """
    
    x = np.asarray(s)
    mu = np.mean(x)
    sigma = np.std(x)

    if sigma == 0:
        return 0.0

    k = np.mean((x - mu) ** 4) / (sigma ** 4)

    # Excess kurtosiss
    return k - 3

# Two sample statistics
def mean_difference(s1: Sequence, s2: Sequence) -> float:
    """
    Mean difference test statistic.
    
    Parameters
    ----------
    s1, s1: Sequence
        Samples to compute the test statistic.
    
    Returns
    -------
    mean(s1) - mean(s2)
    """
    
    return np.mean(s1) - np.mean(s2)

def mean_ratio(s1: Sequence, s2: Sequence) -> float:
    """
    Mean ratio test statistic.
    
    Parameters
    ----------
    s1, s1: Sequence
        Samples to compute the test statistic.
    
    Returns
    -------
    mean(s1) / mean(s2)
    """
    
    return np.mean(s1) / np.mean(s2)

def variance_difference(s1: Sequence, s2: Sequence) -> float:
    """
    Variance difference test statistic.
    
    Parameters
    ----------
    s1, s1: Sequence
        Samples to compute the test statistic.
    
    Returns
    -------
    var(s1) - var(s2)
    """
    
    return np.var(s1) - np.var(s2)

def varaince_ratio(s1: Sequence, s2: Sequence) -> float:
    """
    Variance ratio test statistic.
    
    Parameters
    ----------
    s1, s1: Sequence
        Samples to compute the test statistic.
    
    Returns
    -------
    var(s1) / var(s2)
    """
    
    return np.var(s1) / np.var(s2)

def median_difference(s1: Sequence, s2: Sequence) -> float:
    """
    Median difference test statistic.
    
    Parameters
    ----------
    s1, s1: Sequence
        Samples to compute the test statistic.
    
    Returns
    -------
    median(s1) - median(s2)
    """
    
    return np.median(s1) - np.median(s2)

def t_statistic(s1: Sequence, s2: Sequence):
    """
    t-statistic.
    
    Parameters
    ----------
    s1, s1: Sequence
        Samples to compute the test statistic.
    
    Returns
    -------
    t-statistic for s1 and s2
    """
    
    n1, n2 = len(s1), len(s2)
    m1, m2 = np.mean(s1), np.mean(s2)
    v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    
    se = np.sqrt(v1/n1 + v2/n2)
    return (m1 - m2) / se