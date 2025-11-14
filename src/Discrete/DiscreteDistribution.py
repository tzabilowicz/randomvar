""" Default Discrete Distribution """

class DiscreteDistribution:
    """ Default Discrete Distribution

    Default for convolved discrete distributions.

    Parameters
    ----------
    d1 : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion
    d2 : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion

    Attributes
    ----------
    _f : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion
    _g : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion
    """

    def __init__(self, d1, d2):
        self._f = d1
        self._g = d2

    def __repr__(self):
        return f"D~Discrete({self._f},{self._g})"
    __str__ = __repr__

    def __add__(self, other):
        """
        Add the distributions by creating a convolved
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Convolved discrete distribution
        """

        return DiscreteDistribution(self, other)

    def __sub__(self, other):
        return NotImplemented

    def __mul__(self, other):
        return NotImplemented

    def __div__(self, other):
        return NotImplemented

    def interval(self):
        """
        Interval for the convolved discrete distribution.

        Returns
        -------
        intvl : tuple(int, int)
            Interval of the convolved distribution.
        """

        f_a, f_b = self._f.interval()
        g_a, g_b = self._g.interval()

        a = f_a + g_a
        b = f_b + g_b

        intvl = (a, b)

        return intvl

    def pmf(self, x):
        """
        pmf - Probability Mass Function P(X = x)
        Default pmf for convolved discrete distribution.

        Parameters
        ----------
        x : int
            Value of X to compute pmf.

        Returns
        ------
        p_x : float
            Convolved probability of event X = x.
            P(X = x)

        Notes
        -----
        Each individual discrete distribution has an overriding
        probability mass function that specifies its own pmf.
        """

        if self._f is None or self._g is None:
            raise ValueError(f'Distributions (f,g) cannot be ({self._f}, {self._g})')

        f_a, f_b = self._f.interval()
        g_a, g_b = self._g.interval()

        # Construct the convolution interval
        low  = max(f_a, x - g_b)
        high = min(f_b, x - g_a)

        if low > high:
            p_x = 0
        else:
            p_x = sum([self._f.pmf(i) * self._g.pmf(x - i) for i in range(low, high + 1)])

        return p_x

    def cdf(self, x):
        """
        cdf - Cumulative Distribution Function P(X <= x)
        Default cdf for two convolved discrete distributions.
        @see pmf(x) for additional information

        Parameters
        ----------
        x : int
            Value of X to compute pmf.

        Returns
        -------
        F_x : float
            Convolved probability of the event X <= x.
            P(X <= x)

        Notes
        -----
        The cdf is an accumulation of the probability mass
        up to and including event X = x.
        """

        a, b = self.interval()

        if x < a:
            F_x = 0
        elif x > b:
            F_x = 1
        else:
            # sum of pmf values on [a, x]
            F_x = sum([self.pmf(i) for i in range(a, x+1)])