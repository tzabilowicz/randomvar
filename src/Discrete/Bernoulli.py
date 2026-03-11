""" Bernoulli Distribution """

import random

from Discrete.DiscreteDistribution import DiscreteDistribution

class Bernoulli(DiscreteDistribution):
    """ Bernoulli Distribution

    Bernoulli distribution with probability p of a
    successful event.
    Inherits from DiscreteDistribution for algebraic
    operations.
    @see DiscreteDistribution for more information.

    Parameters
    ----------
    p : float, [0, 1]
        Probability of a successful outcome.
    """

    def __init__(self, p: float) -> None:
        # Validate probability
        if p < 0 or p > 1:
            raise ValueError(f'p must exist in [0, 1]')

        self._p = p

    def __repr__(self) -> str:
        return f"D~Bernoulli({self._p})"
    __str__ = __repr__

    def interval(self) -> tuple[int, int]:
        """
        Return the discrete interval of the Bernoulli distribution.

        Returns
        -------
        intvl : tuple(int, int)
            Interval (0, 1) of the Bernoulli distribution.
        """
        intvl = (0, 1)

        return intvl

    def sample(self) -> int:
        """
        Generate a Bernoulli event. A Bernoulli event either is
        successful (1) or failure (0), goverened by probability p.

        Returns
        -------
        s : int
            Random sample drawn from [0, 1].
        """

        p_s = random.random()
        s = 1 if p_s < self._p else 0

        return s

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

        e_x = self._p
        
        return e_x
    
    def var(self) -> float:
        """
        var - Variance
        Computes the variance of the random variable.
        
        Returns
        -------
        var_x:
            Variance of X.
            Var[X]
        """

        var_x = self._p * (1 - self._p)
        
        return var_x

    def pmf(self, x: int) -> float:
        """
        pmf - Probability Mass Function
        Compute the probability mass at x. If the value of x is not
        0 or 1, the probability of the event is 0.

        Parameters
        ----------
        x : int
            Value of the pmf to compute.

        Returns
        -------
        p_x : float
            Probability of event X = x.
            P(X = x)
        """

        if x != 1 and x != 0:
            p_x = 0
        else:
            p_x = self._p if x == 1 else (1 - self._p)

        return p_x

    def cdf(self, x: int) -> float:
        """
        cdf - Cumulative Distribution Function
        Compute the cumulative probability mass up to and including x.
        @see pmf(x) for additional information.

        Parameters
        ----------
        x : int
            Value of X to compute pmf.

        Returns
        -------
        F_x : float
            Probability of the event X <= x.
            P(X <= x)

        Notes
        -----
        The cdf is an accumulation of the probability mass
        up to and including event X = x.
        """

        F_x = 0 # x < 0

        if x == 0:
            return 1 - self._p
        elif x >= 1:
            F_x = 1

        return F_x