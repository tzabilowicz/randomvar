# Global Imports
import random

# Package Imports
from Discrete import DiscreteDistribution

class Uniform(DiscreteDistribution.DiscreteDistribution):
    def __init__(self, a, b):
        # Validate the interval [a, b]
        if b < a:
            raise ValueError(f"[a, b] - Right interval (b={b}) must be greater than left interval (a={a}).")

        self._a = a
        self._b = b

    def __repr__(self):
        return f'D~Uniform({self._a}, {self._b})'
    __str__ = __repr__

    def sample(self):
        return random.randint(self._a, self._b)

    def pmf(self, x):
        if x < self._a or x > self._b:
            return 0

        return 1 / (self._b - self._a + 1)

    def cdf(self, x):
        return sum([self.pmf(x) for i in range(x)])