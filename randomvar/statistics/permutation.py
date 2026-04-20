""" Permutation Testing """

import numpy as np
from typing import Callable, Optional, Sequence

def permutation_test(
    sample1: Sequence,
    sample2: Sequence, 
    T: Callable[[Sequence, Sequence], float],
    N: int = 1_000,
    two_sided: bool = True,
    rng: Optional[np.random.Generator] = None
):
    """
    Conduct a permutation test for sample data and a 
    pre-defined test statistic T.
    
    Parameters
    ----------
    sample1 : Sequence
        Sample data 1.
    sample2 : Sequence
        Sample data 2.
    T : function(sample1, sample2)
        Function computing the test statistic.
    N : int (default = 1,000)
        Number simulations.
    two_sided : bool (default = True)
        True if two sided test; false for one-sided.
    
    Returns
    -------
    One-Sided Test (two_sided=False):
        Observed test statistic, left p-val, right p-val
    Two-Sided Test (two_sided=True):
        Observed test statistic, p-val
    """
    
    if len(sample1) < 1 or len(sample2) < 1:
        raise ValueError("Must have at least one value in sample1 and sample2.")
    
    if rng is None:
        rng = np.random.default_rng()
    
    pool = np.concatenate((sample1, sample2))
    
    T_obs = T(sample1, sample2)
    T_perm = np.zeros(N)
    
    # Generate the permutation distribution
    for n in range(N):
        perm = rng.permutation(pool)
        t_sample1 = perm[:len(sample1)]
        t_sample2 = perm[len(sample1):]
        
        # Compute the permutated test statistic
        T_perm[n] = T(t_sample1, t_sample2)
    
    if two_sided:
        # Two-sided p-value
        count = np.sum(np.abs(T_perm) >= T_obs)
        p_val = (count + 1) / (len(T_perm) + 1)
        
        return T_obs, p_val
    else:
        # One-sided p-value
        count_r = np.sum(T_perm >= abs(T_obs))
        p_val_r = (count_r + 1) / (len(T_perm) + 1)
        
        count_l = np.sum(T_perm <= T_obs)
        p_val_l = (count_l + 1) / (len(T_perm) + 1)
        
        return T_obs, p_val_l, p_val_r