""" Default Discrete Distribution """

class DiscreteDistribution:
    """ General Discrete Distribution

    Base class for all discrete distributions.

    Parameters
    ----------
    d1 : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion 1
    d2 : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion 2

    Attributes
    ----------
    _f : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion
    _g : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion
    """

    def __init__(self, d1, d2) -> None:
        self._f = d1
        self._g = d2

        # Default operation type
        self._oper_type = 'add'

    def __repr__(self) -> str:
        return f"D~Discrete({self._f},{self._g})"
    __str__ = __repr__

    def __add__(self, other) -> "DiscreteDistribution":
        """
        Add the distributions by creating an abstract
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Abstract discrete distribution
        """

        self._oper_type = 'add'
        return DiscreteDistribution(self, other)

    def __sub__(self, other):
        """
        Subtract the distributions by creating an abstract
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Abstract discrete distribution
        """

        self._oper_type = 'sub'
        return DiscreteDistribution(self, other)

    def __mul__(self, other):
        """
        Multiply the distributions by creating an abstract
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Abstract discrete distribution
        """

        self._oper_type = 'mul'
        return DiscreteDistribution(self, other)

    def __div__(self, other):
        """
        Divide the distributions by creating an abstract
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Abstract discrete distribution
        """

        self._oper_type = 'div'
        return DiscreteDistribution(self, other)

    def interval(self) -> tuple[int, int]:
        """
        Interval for the abstract discrete distribution.

        Returns
        -------
        intvl : tuple(int, int)
            Interval of the abstract distribution.
        """

        f_a, f_b = self._f.interval()
        g_a, g_b = self._g.interval()

        # Correctly generate the interval
        if self._oper_type == 'add':
            a = f_a + g_a
            b = f_b + g_b
        elif self._oper_type == 'sub':
            a = f_a - g_a
            b = f_b - g_b
        elif self._oper_type == 'mul':
            a = f_a * g_a
            b = f_b * g_b
        else: # div
            a = f_a / g_a
            b = f_b / g_b

        intvl = (a, b)

        return intvl

    def sample(self):
        return NotImplemented

    def exp(self):
        return NotImplemented

    def var(self):
        return NotImplemented

    def pmf(self, x: int) -> float:
        """
        pmf - Probability Mass Function P(X = x)
        Default pmf for abstract discrete distribution.

        Parameters
        ----------
        x : int
            Value of X to compute pmf.

        Returns
        ------
        p_x : float
            Probability of event X = x.
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

        # Construct the interval
        low  = max(f_a, x - g_b)
        high = min(f_b, x - g_a)

        if low > high:
            p_x = 0
        else:
            p_x = sum([self._f.pmf(i) * self._g.pmf(x - i) for i in range(low, high + 1)])

        return p_x

    def cdf(self, x: int) -> float:
        """
        cdf - Cumulative Distribution Function P(X <= x)
        Default cdf for two abstract discrete distributions.
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

        a, b = self.interval()

        if x < a:
            F_x = 0
        elif x > b:
            F_x = 1
        else:
            # sum of pmf values on [a, x]
            F_x = sum([self.pmf(i) for i in range(a, x+1)])

        return F_x