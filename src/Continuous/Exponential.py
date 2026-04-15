""" Continuous Exponential Distribution """

import math
import random

from Continuous.ContinuousDistribution import ContinuousDistribution

class Exponential(ContinuousDistribution):
    """ Continuous Exponential Distribution

    Inherits from the ContinuousDistribution for algebraic
    operations.
    @see ContinuousDistribution for more information.

    Parameters
    ----------
    lambd : float
        Rate parameter; average occurrence of events.
    """

    def __init__(self, lambd: float) -> None:
        # Validate the lambd parameter
        if lambd <= 0:
            raise ValueError(f"Lambda (lambd={lambd}) must be > 0")

        self._lambd = lambd

    def __repr__(self) -> str:
        return f'D~Exponential({self._lambd})'
    __str__ = __repr__

    def interval(self) -> tuple[int, float]:
        """
        Return the continuous interval of the exponential random
        variable.

        Returns
        -------
        intvl : tuple[int, float]
            Interval of the exponential distribution.
        """

        return (0, float('inf'))

    def sample(self) -> float:
        """
        Generate a exponential random sample from the distribution.

        Returns
        -------
        s : float
            Random sample drawn from the distribution.
        """

        return random.expovariate(self._lambd)

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

        return 1 / self._lambd

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

        return 1 / (self._lambd ** 2)

    def pdf(self, x: float) -> float:
        """
        pdf - Probability Density Function
        Compute the probability density at x.

        Parameters
        ----------
        x : float
            Value of X to compute the pdf.

        Returns
        -------
        p_x : float
            Probability of event X = x.
            P(X = x)
        """

        if x < 0:
            return 0.0

        return self._lambd * math.exp(-self._lambd * x)

    def cdf(self, x: float) -> float:
        """
        cdf - Cumulative Distribution Function P(X <= x)
        Compute the cumulative probability density up to and including x.
        @see pdf(x) for additional information

        Parameters
        ----------
        x : float
            Value of X to compute cdf.

        Returns
        -------
        F_x : float
            Probability of the event X <= x.
            P(X <= x)

        Notes
        -----
        The cdf is an accumulation of the probability density
        up to and including event X = x.
        """

        if x < 0:
            return 0.0

        return 1 - math.exp(-self._lambd * x)