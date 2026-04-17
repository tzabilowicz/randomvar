""" Binomial Distribution """

from distributions.discrete.Bernoulli import Bernoulli
from distributions.discrete.DiscreteDistribution import DiscreteDistribution
from utils.utils import factorial

class Binomial(DiscreteDistribution):
    """ Binomial Distribution

    Binomial distribution is the number of successes in
    N independent Bernoulli events.
    Inherits from DiscreteDistribution for algebraic
    operations.
    @see DiscreteDistribution for more information.

    Parameters
    ----------
    n : int
        Number of independent Bernoulli random variables.
    p : float, [0, 1]
        Probability of a successful outcome.
    """

    def __init__(self, n: int, p: float) -> None:
        # Validate probability
        if p < 0 or p > 1:
            raise ValueError('p must exist in [0, 1]')

        # Validate number of trials
        if n < 0:
            raise ValueError(f'Must have a positive number of trials')

        self._p = p
        self._n = n

    def __repr__(self) -> str:
        return f"D~Binomial({self._n},{self._p})"
    __str__ = __repr__

    def interval(self) -> tuple[int, int]:
        """
        Return the discrete interval of the Binomial distribution.

        Returns
        -------
        intvl : tuple(int, int)
            Interval (0, n) of the Binomial distribution.
        """

        return (0, self._n)

    def sample(self) -> int:
        """
        Generate a Binomial event. A Binomial event is a series of
        n independent Bernoulli events. A Binomial event models the
        number of successes in n independent Bernoulli events.

        Returns
        -------
        s : int
            Random sample modeling the number of successes in n
            independent Bernoulli events.
        """

        X = Bernoulli(self._p)
        s = sum([X.sample() for i in range(self._n)])

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

        return self._n * self._p

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

        return self._n * self._p * (1 - self._p)

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

        if x < 0 or x > self._n:
            p_x = 0
        else:
            nCx = factorial(self._n) / (factorial(x) * factorial(self._n - x))
            p_x = nCx * (self._p ** x) * ((1 - self._p) ** (self._n - x))

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

        # x+1 to include x
        return sum(self.pmf(i) for i in range(x+1))