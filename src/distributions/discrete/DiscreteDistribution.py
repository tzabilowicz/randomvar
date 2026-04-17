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
    oper_type : str
        Operation type for arithmetic between two random variables.

    Attributes
    ----------
    _f : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion
    _g : (Bernoulli, Binomial, DiscreteDistribution, Geometric, Uniform)
        Discrete distributuion
    _oper_type : str
        Operation type for arithmetic between two random variables.
    """

    def __init__(self, d1, d2, oper_type: str) -> None:
        self._f = d1
        self._g = d2

        # Default operation type
        self._oper_type = oper_type

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

        return DiscreteDistribution(self, other, 'add')

    def __sub__(self, other) -> "DiscreteDistribution":
        """
        Subtract the distributions by creating an abstract
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Abstract discrete distribution
        """

        return DiscreteDistribution(self, other, 'sub')

    def __mul__(self, other) -> "DiscreteDistribution":
        """
        Multiply the distributions by creating an abstract
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Abstract discrete distribution
        """

        return DiscreteDistribution(self, other, 'mul')

    def __truediv__(self, other) -> "DiscreteDistribution":
        """
        Divide the distributions by creating an abstract
        discrete distribution.

        Returns
        -------
        DiscreteDistribution
            Abstract discrete distribution
        """

        return DiscreteDistribution(self, other, 'div')

    def interval(self) -> tuple[int, int]:
        """
        Interval for the abstract discrete distribution.

        Returns
        -------
        intvl : tuple(int, int)
            Interval of the abstract distribution. Value returned
            as (lower, upper).
        """
        
        # Validate distributions
        if self._f is None or self._g is None:
            raise ValueError("Member distributions cannot be None.")

        f_a, f_b = self._f.interval()
        g_a, g_b = self._g.interval()
        
        intvl_candidates = []

        # Correctly generate the interval
        if self._oper_type == 'add':
            intvl_candidates = [
                f_a + g_a,
                f_a + g_b,
                f_b + g_a,
                f_b + g_b
            ]
        elif self._oper_type == 'sub':
            intvl_candidates = [
                f_a - g_a,
                f_a - g_b,
                f_b - g_a,
                f_b - g_b
            ]
        elif self._oper_type == 'mul':
            intvl_candidates = [
                f_a * g_a,
                f_a * g_b,
                f_b * g_a,
                f_b * g_b
            ]
        else: # div
            # Ensure no div by 0 error
            if g_a == 0 or g_b == 0:
                raise ZeroDivisionError(f"Invalid endpoints: g_a={g_a} | g_b={g_b}")
            
            intvl_candidates = [
                f_a / g_a,
                f_a / g_b,
                f_b / g_a,
                f_b / g_b
            ]

        intvl = (min(intvl_candidates), max(intvl_candidates))

        return intvl

    def sample(self) -> float:
        """
        Generate a sample from the discrete distribution.
        The sample is a function of the member distributions
        that create the abstract discrete distribution.
        
        Returns
        -------
        s : float
            Sample of the abstract discrete distribution.
        """
        
        if self._f is None or self._g is None:
            raise ValueError("Member distributions cannot be None.")

        f_s = self._f.sample()
        g_s = self._g.sample()
        
        if self._oper_type == 'add':
            s = f_s + g_s
        elif self._oper_type == 'sub':
            s = f_s - g_s
        elif self._oper_type == 'mul':
            s = f_s * g_s
        else: # div
            if g_s == 0:
                raise ZeroDivisionError("Member distribution sample of g is 0")
            
            s = f_s / g_s
        
        return float(s)

    def exp(self) -> float:
        """
        exp - Expectation
        Computes the expectation of the abstract discrete distribution.
        
        Returns 
        -------
        e_x : float
            The expected value of the discrete distribution.
            E[X]
        """

        a, b = self.interval()
        
        if not float(a).is_integer() or not float(b).is_integer():
            return NotImplementedError("E[X]: currently requires a finite, integer valued interval...")
        
        a, b = int(a), int(b) # Ensure the intervals are discrete
        
        prob_vals = [x * self.pmf(x) for x in range(a, b + 1)]
        
        return sum(prob_vals)

    def var(self) -> float:
        """
        var - Variance
        Computes the variance of the discrete distribution.

        Returns
        -------
        var_x:
            Variance of X.
            Var[X]
        """
        
        a, b = self.interval()
        
        if not float(a).is_integer() or not float(b).is_integer():
            return NotImplementedError("Var[X]: currently requires a finite, integer valued interval...")

        a, b = int(a), int(b)
        
        mu = self.exp()
        prob_vals = [self.pmf(x) * ((x - mu) ** 2) for x in range(a, b + 1)]
        
        return sum(prob_vals)

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
            raise ValueError("Member distributions cannot be None.")

        f_a, f_b = self._f.interval()
        g_a, g_b = self._g.interval()

        p_x = 0

        if self._oper_type == 'add':
            lo = min(f_a, x - g_b)
            hi = max(f_b, x - g_a)
            
            if hi >= lo:
                p_x = sum(self._f.pmf(i) * self._g.pmf(x - i) for i in range(lo, hi + 1))
        elif self._oper_type == 'sub':
            lo = min(f_a, x + g_a)
            hi = max(f_b, x + g_b)
            
            if hi >= lo:
                p_x = sum(self._f.pmf(i) * self._g.pmf(i - x) for i in range(lo, hi + 1))
        elif self._oper_type == 'mul':
            total = 0.0

            for i in range(f_a, f_b + 1):
                if i != 0 and x % i == 0:
                    j = x // i

                    if g_a <= j <= g_b:
                        total += self._f.pmf(i) * self._g.pmf(j)
                elif i == 0 and x == 0:
                    for j in range(g_a, g_b + 1):
                        total += self._f.pmf(i) * self._g.pmf(j)

            p_x = total
        else: # div
            total = 0.0

            for j in range(g_a, g_b + 1):
                if j == 0:
                    continue

                if x * j == int(x * j):
                    i = int(x * j)
                    if f_a <= i <= f_b:
                        total += self._f.pmf(i) * self._g.pmf(j)

            return total

        return float(p_x)

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
        
        if not float(a).is_integer() or not float(b).is_integer():
            return NotImplementedError("cdf: currently requires finite, integer valued interval...")

        if x < a:
            F_x = 0
        elif x >= b:
            F_x = 1
        else:
            # sum of pmf values on [a, x]
            F_x = sum([self.pmf(i) for i in range(a, x + 1)])

        return F_x