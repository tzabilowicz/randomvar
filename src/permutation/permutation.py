import numpy as np
from typing import Callable, Sequence

def permutation_test(
    s1: Sequence,
    s2: Sequence, 
    T: Callable[[Sequence, Sequence], float],
    N: int = 10_000,
    two_sided: bool = True
):
    """
    Conduct a permutation test for sample data and a 
    pre-defined test statistic T.
    
    Parameters
    ----------
    s1 : Sequence
        Sample data 1.
    s2 : Sequence
        Sample data 2.
    T : function(s1, s2)
        Function defining the test statistic.
    N : int (default = 10,000)
        Number of random resamples.
    two_sided : bool (default = True)
        True if two sided test; false for one-sided.
    
    Returns
    -------
    One-Sided Test (two_sided=False):
        Observed test statistic, left p-val, right p-val
    Two-Sided Test (two_sided=True):
        Observed test statistic, p-val
    """
    
    pool = np.concatenate((s1, s2))
    
    T_obs = T(s1, s2)
    T_perm = np.zeros(N)
    
    # Generate the permutation distribution
    for n in range(N):
        perm = np.random.permutation(pool)
        t_sample1 = perm[:len(s1)]
        t_sample2 = perm[len(s1):]
        
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