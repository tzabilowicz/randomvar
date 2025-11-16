""" Discrete Uniform Distribution """

import random

from Discrete.DiscreteDistribution import DiscreteDistribution

class Uniform(DiscreteDistribution):
    """ Discrete Uniform Distribution

    Discrete uniform distribution on the range [a, b].
    Inherits from DiscreteDistribution for algebraic
    operators.
    @see DiscreteDistribution for more information.

    Parameters
    ----------
    a : int
        Inclusive left boundary.
    b : int
        Inclusive right boundary.
    """

    def __init__(self, a, b):
        # Validate interval [a, b]
        if b < a:
            raise ValueError(f"[a, b] - Right interval (b={b}) must be greater than left interval (a={a}).")

        self._a = a
        self._b = b

    def __repr__(self):
        return f'D~Uniform({self._a},{self._b})'
    __str__ = __repr__

    def interval(self):
        """
        Return the discrete interval of the uniform random variable.

        Returns
        -------
        intvl : tuple(int, int)
            Interval (a, b) of the Uniform distribution.
        """
        intvl = (self._a, self._b)

        return intvl

    def sample(self):
        """
        Generate a uniform random sample drawn from the interval
        of the distribution, [a, b]. Each value has equal an
        equal probability of 1/n, where n=(b-a).

        Returns
        -------
        s : int
            Random sample drawn from [a, b],
        """
        s = random.randint(self._a, self._b)

        return s

    def pmf(self, x):
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
            p_x = 0
        else:
            p_x = 1 / abs(self._b - self._a + 1)

        return p_x

    def cdf(self, x):
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
            F_x = 0
        elif x > self._b:
            F_x = 1
        else:
            # sum of pmf values on [a, x]
            F_x = sum([self.pmf(i) for i in range(self._a, x+1)])

        return F_x