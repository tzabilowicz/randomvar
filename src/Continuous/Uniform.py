""" Continuous Uniform Distribution """

import random

from Continuous.ContinuousDistribution import ContinuousDistribution

class Uniform(ContinuousDistribution):
    """ Continuous Uniform Distribution

    Continuous uniform distribution on the range [a, b].
    Inherits from the ContinuousDistribution for algebraic
    operations.
    @see ContinuousDistribution for more information.

    Parameters
    ----------
    a : float
        Inclusive left boundary.
    b : float
        Inclusive right boundary.
    """

    def __init__(self, a: float, b: float) -> None:
        # Validate the interval
        if b < a:
            raise ValueError(f"[a, b] - Right interval (b={b}) must be greater than left interval (a={a}).")

        self._a = a
        self._b = b

    def __repr__(self) -> str:
        return f'D~Uniform({self._a},{self._b})'
    __str__ = __repr__

    def interval(self) -> tuple[float, float]:
        """
        Return the continuous interval of the uniform random variable.

        Returns
        -------
        intvl : tuple[float, float]
            Interval (a, b) of the Uniform distribution.
        """

        return (self._a, self._b)

    def sample(self) -> float:
        """
        Generate a uniform random sample from the interval
        of the distribution on [a, b]. Each value has an equal
        probability of 1/n, where n=(b-a).

        Returns
        -------
        s : float
            Random sample drawn from [a, b].
        """

        return random.uniform(self._a, self._b)

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

        return (self._a + self._b) / 2

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

        return ((self._a - self._b) ** 2) / 12

    def pdf(self, x: float) -> float:
        """
        pdf - Probability Density Function P(X = x)
        Compute the probability density at x. If the value of x is
        outside the interval [a, b], the probability of event X = x
        is 0.

        Parameters
        ----------
        x : float
            Value of X to compute pdf.

        Returns
        -------
        p_x : float
            Probability of event X = x.
            P(X = x)
        """

        if x < self._a or x > self._b:
            return 0.0

        return 1 / abs(self._b - self._a + 1)

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

        if x < self._a:
            return 0.0
        elif x > self._b:
            return 1.0

        return (x - self._a) / (self._b - self._a)