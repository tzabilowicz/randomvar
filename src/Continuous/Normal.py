""" Normal Distribution """

import math
import random

from Continuous.ContinuousDistribution import ContinuousDistribution

class Normal(ContinuousDistribution):
    """ Normal Distribution
    
    Continuous normal distribution.
    Inherits from the COntinuousDistribution for algebraic
    operations.
    @see ContinuousDistribution for more information.
    
    Parameters
    ----------
    mu : float
        Mean of the normal distribution.
    var : float
        Variance of the normal distribution.
    """
    
    def __init__(self, mu: float, var: float) -> None:
        # Validate positive variance
        if var < 0:
            raise ValueError(f"Variance must be positive.")
            
        self._mu = mu
        self._var = var
        
    def __repr__(self) -> str:
        return f'D~Normal({self._mu},{self._var})'
    __str__ = __repr__
    
    def interval(self) -> tuple[float, float]:
        """
        Return the interval of the normal distribution.
        
        Returns
        -------
        intvl : tuple[float, float]
            Interval (-inf, inf) of the normal distribution.
        """
        
        return (-float("inf"), float("inf"))
    
    def sample(self) -> float:
        """
        Generate a random sample from the distribution.
        
        Returns
        -------
        s : float
            Random sample from the distribution.
        """
        
        return random.normalvariate(self._mu, math.sqrt(self._var))
    
    def exp(self) -> float:
        """
        exp - Expectation
        Computes the expectation of the random variable.
        
        Returns
        -------
        e_x : float
            Expected value of X.
            E[X]
        """
        
        return self._mu
    
    def var(self) -> float:
        """
        var - Variance
        Computes the variance of the random variable.
        
        Returns
        -------
        var_x : float
            Variance of X.
            Var[X]
        """
        
        return self._var
    
    def pdf(self, x: float) -> float:
        """
        pdf - Probability Density Function
        Compute the probability density.
        
        Parameters
        ----------
        x : float
            Value of x to compute pdf.
        
        Returns
        -------
        p_x : float
            Probability of the event X = x.
            P(X = x)
        """
        
        return (1 / math.sqrt(2 * math.pi * self._var)) * math.exp(-(x - self._mu)**2/(2 * self._var))
    
    def cdf(self, x: float) -> float:
        pass