""" Discrete HyperGeometric Distribution """

import numpy as np
from utils.utils import choose

from distributions.discrete.DiscreteDistribution import DiscreteDistribution

class Hypergeometric(DiscreteDistribution):
    """ Hypergeometric Distribution
    
    Hypergeometric distribution is the number of k successes in
    n draws without replacement from a finite population of N
    containing K observations with that feature, where each draw
    is either a success or failure.
    Inherits from DiscreteDistribution for algebraic operations.
    @see DiscreteDistribution for more information.
    
    Parameters
    ----------
    N : int
        Population size.
    K : int
        Number of observations in the population with the desired
        features.
    n : int
        Number of draws from the population.
    """
    
    def __init__(self, N: int, K: int, n: int) -> None:
        # Validate parameters
        if N < 0 or K < 0 or n < 0:
            raise ValueError("All parameters N, K, and n must be >= 0.")
        if K > N or n > N:
            raise ValueError("Parameters K and n must be >= N.")
    
        self._N = N
        self._K = K
        self._n = n
    
    def __repr__(self) -> str:
        return f"D~Hypergeometric({self._N},{self._K},{self._n})"
    __str__ = __repr__
    
    def interval(self) -> tuple[int, int]:
        pass
    
    def sample(self) -> int:
        """
        Generate a Hypergeometrix event. A Hypergeometric event is the
        number of k successes in n draws without replacement from a 
        population size N with K observations with the desirable feature.

        Returns
        -------
        s: int, float
            Random sample from the hypergeometric.
        """
        
        return np.random.hypergeometric(
            self._K,
            self._N - self._K,
            self._n
        )
    
    def exp(self) -> float:
        """
        exp - Expectation
        Computes the expectation of the random variable.
        
        Returns
        -------
        e_x: float
            Expected value of X.
            E[X]
        """
        
        return self._n * (self._K / self._N)
    
    def var(self) -> float:
        """
        var - Variance
        Computes the variance of the random variable.
        
        Returns
        -------
        var_x: float
            Variance of X,
            Var[X]
        """
        
        return self._n * \
            (self._K / self._N) * \
            ((self._N - self._K) / self._N) * \
            ((self._N - self._n) / (self._N - 1))
    
    def pmf(self, x: int) -> float:
        if x < 0 or x > self._N:
            return 0.0
        
        return (choose(self._K, x) * choose((self._N - self._K), (self._n - x))) / choose(self._N, self._n)
    
    def cdf(self, x) -> float:
        pass