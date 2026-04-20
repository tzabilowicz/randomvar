""" Bootstrap Distribution """

import numpy as np
from typing import Callable, Optional, Sequence

def bootstrap(
    sample: Sequence,
    T: Callable[[Sequence], float],
    N: int = 1_000,
    lower_ci: float = 0.025,
    upper_ci: float = 0.975,
    rng: Optional[np.random.Generator] = None
):
    """
    Compute the one sample distribution for a test statistic and
    sample data. Default bootstrap confidence interval is 95%.
    
    Parameters
    ----------
    sample : Sequence
        Sample data.
    T : function(sample)
        Test statistic.
    N : int
        Number of simulations.
    lower_ci : float (default = 0.025)
        Lower confidence interval bound.
    upper_ci : float (default = 0.975)
        Upper confidence interval bound.
    rng : np.random.Generator (defualt = None)
        Random number generator.
        If None is used, the default (unseeded)
        random number generator is used.
    
    Returns
    -------
    list, float, float, list
    BS distribution, BS mean, BS standard error, 
    Confidence Interval
    """
    
    s_size = len(sample)
    
    if s_size < 1:
        raise ValueError("Must have at least one value in sample.")
    
    if rng is None:
        rng = np.random.default_rng()
    
    T_bs = np.zeros(N)
    
    # Generate the bootstrap distribution
    for n in range(N):
        s = rng.choice(sample, size=s_size, replace=True)
        
        T_bs[n] = T(s)
    
    # Compute the confidence interval
    ci = np.quantile(T_bs, q=[lower_ci, upper_ci])
        
    return T_bs, np.mean(T_bs), np.std(T_bs), ci

def bootstrap_two_sample(
    sample1: Sequence,
    sample2: Sequence,
    T: Callable[[Sequence, Sequence], float],
    N: int = 1_000,
    lower_ci: float = 0.025,
    upper_ci: float = 0.975,
    rng: Optional[np.random.Generator] = None
):
    """
    Compute the two sample distribution for a test statistic and
    sample data. Default bootstrap confidence interval is 95%.
    
    Parameters
    ----------
    sample1 : Sequence
        Sample data.
    sample2: Sequence
        Sample data.
    T : function(sample1, sample2)
        Test statistic.
    N : int
        Number of simulations.
    lower_ci : float (default = 0.025)
        Lower confidence interval bound.
    upper_ci : float (default = 0.975)
        Upper confidence interval bound.
    rng : np.random.Generator (defualt = None)
        Random number generator.
        If None is used, the default (unseeded)
        random number generator is used.
    
    Returns
    -------
    list, float, float, list
    BS distribution, BS mean, BS standard error, 
    Confidence Interval
    """
    
    sample1_size = len(sample1)
    sample2_size = len(sample2)
    
    if sample1_size < 1 or sample2_size < 1:
        raise ValueError("Must have at least one value in sample1 and sample2.")

    if rng is None:
        rng = np.random.default_rng()
    
    T_bs = np.zeros(N)

    # Generate the bootstrap distribution
    for n in range(N):
        t_sample1 = rng.choice(sample1, size=sample1_size, replace=True)
        t_sample2 = rng.choice(sample2, size=sample2_size, replace=True)
        
        T_bs[n] = T(t_sample1, t_sample2)
    
    # Compute the confidence interval
    ci = np.quantile(T_bs, q=[lower_ci, upper_ci])
        
    return T_bs, np.mean(T_bs), np.std(T_bs), ci

def bootstrap_block(
    sample: Sequence,
    T: Callable[[Sequence], float],
    block_size: int,
    N: int = 1_000,
    lower_ci: float = 0.025,
    upper_ci: float = 0.975,
    rng: Optional[np.random.Generator] = None
):
    """
    Compute the one sample distribution for a test statistic and
    sample data. Default bootstrap confidence interval is 95%.
    This bootstrap is computed using overlapping blocks generated
    from the samples, ensuring the data's sequential structure is
    not changed.
    
    Parameters
    ----------
    sample : Sequence
        Sample data.
    T : function(sample)
        Test statistic.
    block_size : int
        Number of samples in a block.
    N : int
        Number of simulations.
    lower_ci : float (default = 0.025)
        Lower confidence interval bound.
    upper_ci : float (default = 0.975)
        Upper confidence interval bound.
    rng : np.random.Generator (defualt = None)
        Random number generator.
        If None is used, the default (unseeded)
        random number generator is used.
    
    Returns
    -------
    list, float, float, list
    BS distribution, BS mean, BS standard error, 
    Confidence Interval
    """
    
    s_size = len(sample)
    
    if s_size < 1:
        raise ValueError("Must have at least one value in the sample.")
    
    # Validate block sizing
    if block_size < 1:
        raise ValueError("Block size must be > 1.")
    if block_size > s_size:
        raise ValueError("Block size must be <= sample size.")
    
    if rng is None:
        rng = np.random.default_rng()
        
    # Construct overlapping blocks
    blocks = np.array([sample[i:i + block_size] for i in range(s_size - block_size + 1)])
    n_blocks = len(blocks)
    
    T_bs = np.zeros(N)
    
    for n in range(N):
        # Get the sample blocks
        k = int(np.ceil(s_size / block_size))
        b_idx = rng.integers(0, n_blocks, size=k)
        s = blocks[b_idx].reshape(-1)[:s_size]
        
        T_bs[n] = T(s)

    # Compute the confidence interval
    ci = np.percentile(T_bs, q=[lower_ci, upper_ci])
    
    return T_bs, np.mean(T_bs), np.std(T_bs), ci

def bootstrap_block_two_sample(
    sample1: Sequence,
    sample2: Sequence,
    T: Callable[[Sequence, Sequence], float],
    block_size: int,
    N: int = 1_000,
    lower_ci: float = 0.025,
    upper_ci: float = 0.975,
    rng: Optional[np.random.Generator] = None
):
    """
    Compute the two sample distribution for a test statistic and
    sample data. Default bootstrap confidence interval is 95%.
    This bootstrap is computed using overlapping blocks generated
    from the samples, ensuring the data's sequential structure is
    unchanged.
    
    Parameters
    ----------
    sample1 : Sequence
        Sample data.
    sample2: Sequence
    T : function(sample1, sample2)
        Test statistic.
    block_size : int
        Number of samples in a block.
    N : int
        Number of simulations.
    lower_ci : float (default = 0.025)
        Lower confidence interval bound.
    upper_ci : float (default = 0.975)
        Upper confidence interval bound.
    rng : np.random.Generator (defualt = None)
        Random number generator.
        If None is used, the default (unseeded)
        random number generator is used.
    
    Returns
    -------
    list, float, float, list
    BS distribution, BS mean, BS standard error, 
    Confidence Interval
    """
    
    sample1_size = len(sample1)
    sample2_size = len(sample2)
    
    if sample1_size < 1 or sample2_size < 1:
        raise ValueError("Must have at least one value in sample1 and sample2.")
    
    # Validate block sizing
    if block_size < 1:
        raise ValueError("Block size must be > 1.")
    if block_size > sample1_size or block_size > sample2_size:
        raise ValueError("Block size must be <= sample sizes.")
    
    if rng is None:
        rng = np.random.default_rng()
        
    # Construct overlapping blocks
    block_sample1 = np.array([sample1[i:i + block_size] for i in range(sample1_size - block_size + 1)])
    block_sample2 = np.array([sample2[i:i + block_size] for i in range(sample2_size - block_size + 1)])
    
    T_bs = np.zeros(N)
    
    for n in range(N):
        # Get the sample blocks
        k1 = int(np.ceil(sample1_size / block_size))
        k2 = int(np.ceil(sample2_size / block_size))
        
        b1_idx = rng.integers(0, len(block_sample1), size=k1)
        b2_idx = rng.integers(0, len(block_sample2), size=k2)
        
        t_sample1 = block_sample1[b1_idx].reshape(-1)[:sample1_size]
        t_sample2 = block_sample2[b2_idx].reshape(-1)[:sample2_size]
        
        T_bs[n] = T(t_sample1, t_sample2)
    
    # Compute the confidence interval
    ci = np.percentile(T_bs, q=[lower_ci, upper_ci])
    
    return T_bs, np.mean(T_bs), np.std(T_bs), ci