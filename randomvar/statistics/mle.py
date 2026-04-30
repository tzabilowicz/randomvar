""" 
Maximum Likelihood Estimators 

MLE for all distributions in the randomvar distribution library.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Sequence

# Continuous maximum likelohood estimates
def mle_cauchy():
    pass

def mle_exponential(x: Sequence) -> float:
    pass

def mle_gamma():
    pass

def mle_normal(x: Sequence) -> float:
    pass

def mle_uniform():
    pass

# Discrete maximum likelihood estimates
def mle_bernoulli(x: Sequence) -> float:
    """
    Maximum Likelihood Estimate - Bernoulli
    
    Parameters
    ----------
    x : Sequence
        Series of Bernoulli trials assumed to be samples
        independently and identically.
    
    Returns
    -------
    float
    p MLE
    """
    
    return np.mean(x)

def mle_binomial(x: Sequence) -> float:
    return np.sum(x) / len(x)

def mle_geometric(x: Sequence) -> float:
    return len(x) / np.sum(x)

def mle_hypergeometric():
    pass

def mle_multinomial():
    pass

def mle_negativebinomial():
    pass

def mle_poisson():
    pass