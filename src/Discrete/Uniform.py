""" Discrete Uniform Distribution """

import random

from Discrete.DiscreteDistribution import DiscreteDistribution

class Uniform(DiscreteDistribution):
    """ Discrete Uniform Distribution

    Discrete uniform distribution on the range [a, b].
    Inherits from DiscreteDistribution for algebraic
    operations.
    @see DiscreteDistribution for more information.

    Parameters
    ----------
    a : int
        Inclusive left boundary.
    b : int
        Inclusive right boundary.
    """

    def __init__(self, a: int, b: int) -> None:
        # Validate interval [a, b]
        if b < a:
            raise ValueError(f"[a, b] - Right interval (b={b}) must be greater than left interval (a={a}).")

        self._a = a
        self._b = b

    def __repr__(self) -> str:
        return f'D~Uniform({self._a},{self._b})'
    __str__ = __repr__

    def interval(self) -> tuple[int, int]:
        """
        Return the discrete interval of the uniform random variable.

        Returns
        -------
        intvl : tuple(int, int)
            Interval (a, b) of the Uniform distribution.
        """

        return (self._a, self._b)

    def sample(self) -> int:
        """
        Generate a uniform random sample from the interval
        of the distribution on [a, b]. Each value has equal an
        equal probability of 1/n, where n=(b-a+1).

        Returns
        -------
        s : int
            Random sample drawn from [a, b],
        """

        return random.randint(self._a, self._b)

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

        return ((self._b - self._a + 1) ** 2) / 12

    def pmf(self, x: int) -> float:
        """
        pmf - Probability Mass Function P(X = x)
        Compute the probability mass at x. If the value of x is
        outside interval [a, b], the probability of event X = x
        is 0.

        Parameters
        ----------
        x : int
            Value of X to compute pmf.

        Returns
        -------
        p_x : float
            Probability of event X = x.
            P(X = x)
        """

        if x < self._a or x > self._b:
            return 0.0

        return 1 / abs(self._b - self._a + 1)

    def cdf(self, x: int) -> float:
        """
        cdf - Cumulative Distribution Function P(X <= x)
        Compute the cumulative probability mass up to and including x.
        @see pmf(x) for additional information

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

        if x < self._a:
            return 0.0
        elif x >= self._b:
            return 1.0

        return (x - self._a + 1) / (self._b - self._a + 1)