"""Discrete Geometric Distribution"""

from distributions.discrete.Bernoulli import Bernoulli
from distributions.discrete.DiscreteDistribution import DiscreteDistribution

class Geometric(DiscreteDistribution):
    """ Discrete Geometric Distribution

    Geometric disctribution is the number of failures
    until the first success.
    Inherits from DiscreteDistribution for algebraic
    operations.
    @see DiscreteDistribution for more information.

    Parameters
    ----------
    p: float, [0, 1]
        Probability of a successful outcome.
    """

    def __init__(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must exist in [0, 1]")

        self._p = p

    def __repr__(self):
        return f"D~Geometric({self._p})"
    __str__ = __repr__

    def interval(self) -> tuple[int, float]:
        """
        Return the discrete interval of the Geometric distribution.

        Returns
        -------
        intvl: tuple(int, float)
            Interval (1, inf) of the Geometric distribution.
        """

        return (1, float('inf'))

    def sample(self):
        """
        Generate a Geometric event. A Geometric event is the number of
        failures in a series of Bernoulli trials until the first
        success.

        Returns
        -------
        s: int, float
            Random sample modeling the number of failures until the first
            success.
        """

        # If the probability of success is 0, a successful
        # outcome will never occur
        if self._p == 0:
            return float('inf')

        X = Bernoulli(self._p)

        # Continue generating samples until first success(1)
        s = 1
        while X.sample() == 0:
            s += 1

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

        return 1 / self._p

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

        return (1 - self._p) / (self._p ** 2)

    def pmf(self, x: int) -> float:
        """
        pmf - Probability Mass Function
        Compute the probability mass at x.

        Parameters
        ----------
        x: int
            Value of the pmf to compute.

        Returns
        -------
        p_x: float
            Probability of event X = x.
            P(X = x)
        """

        if x < 1:
            return 0.0

        return ((1 - self._p) ** (x - 1)) * self._p

    def cdf(self, x: int) -> float:
        """
        cdf - Cumulative Distribution Function
        Compute the cumulative probability mass up to and including x.
        @see pmf(x) for more information.

        Parameters
        ----------
        x: int
            Value of X to compute pmf.

        Returns
        -------
        F_x: float
            Probability of the event X <= x.
            P(X <= x)

        Notes
        -----
        The cdf is an accumulation of the probability mass
        up to and including event X = x.
        """

        # 1 to x+1
        return sum(self.pmf(i) for i in range(1, x+1))