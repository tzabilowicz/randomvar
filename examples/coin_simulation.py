""" Coin Simulation

This example program simulates a single coin flip then
a series of coin flips. It shows how RandomVar can be
used with Monte Carlo simulations.
"""

from Discrete.Bernoulli import Bernoulli
from Discrete.Binomial import Binomial

# Create the random variable (coin; p=0.5 for fair coin)
p = 0.5
X = Bernoulli(p)
print(X)

# Flip the coin
flip = X.sample()
print(f"Flip 1: {flip}")
flip = X.sample()
print(f"Flip 2: {flip}")

# Probability of a particular outcome (pmf)
p_flip = X.pmf(flip)
print(f"P(X={flip})={p_flip}")