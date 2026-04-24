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

def mle_exponential(x: Sequence, method: Optional[str] = "BFGS") -> float:
    """
    Maximum Likelihood Estimate - Exponential
    
    Parameters
    ----------
    x : Sequence
        Sample data.
    method : str (default = BFGS)
        Optimization solver. See scipy.optimize.minimize for
        more information on available solvers.
    
    Returns
    -------
    float
    lambda MLE
    """
    
    x = np.array(x)
    
    def neg_log_likelihood(param, x):
        """ Negative log-likelihood function """
        
        lam = param
        if lam <= 0:
            return np.inf
        
        n = len(x)
        
        log_likelihood = n * np.log(lam) - lam * np.sum(x)
        return -log_likelihood
    
    # Run the optimization
    result = minimize(
        fun=neg_log_likelihood,
        x0=[1/np.mean(x)],
        args=(x,),
        method=method,
    )
    
    if not result:
        raise RuntimeError(result.message)

    lam_hat = result.x[0]
    
    return lam_hat

def mle_gamma():
    pass

def mle_normal(x: Sequence, method: Optional[str] = "BFGS") -> float:
    """
    Maximum Likelihood Estimate - Normal
    
    Parameters
    ----------
    x : Sequence
        Sample data.
    method : str (default = BFGS)
        Optimization solver. See scipy.optimize.minimize for
        more information on available solvers.
    
    Returns
    -------
    float, float
    mu MLE, sigma MLE
    """
    
    x = np.array(x)
    
    # Define normal likelihood function
    def neg_log_likelihood(params, x):
        """ Negative log-likelihood function """
        
        mu, sigma = params
        if sigma <= 0:
            return np.inf

        sigma_sq = sigma ** 2
        n = len(x)
        
        log_likelihood = -(n / 2) * np.log(2 * np.pi * sigma_sq) - (1 / (2 * sigma_sq)) * np.sum((x - mu) ** 2)
        return -log_likelihood
    
    # Run the optimization
    result = minimize(
        fun=neg_log_likelihood,
        x0=[np.mean(x), np.std(x, ddof=0)],
        args=(x,),
        method=method
    )
    
    # Optimization failure
    if not result:
        raise RuntimeError(result.message)
    
    mu_hat = result.x[0]
    sigma_hat = result.x[1]
    
    return mu_hat, sigma_hat

def mle_uniform():
    pass

# Discrete maximum likelihood estimates
def mle_bernoulli():
    pass

def mle_binomial():
    pass

def mle_geometric():
    pass

def mle_hypergeometric():
    pass

def mle_multinomial():
    pass

def mle_negativebinomial():
    pass

def mle_poisson():
    pass