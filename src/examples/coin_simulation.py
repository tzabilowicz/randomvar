""" Coin Simulation

This example program simulates a single coin flip then
a series of coin flips. It shows how RandomVar can be
used with Monte Carlo simulations.
"""

from Discrete.Bernoulli import Bernoulli
from Discrete.Binomial import Binomial

# Single coin flip
# ========================
# random variable (coin; p=0.5 for fair coin)
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
print(f"P(X={flip})={p_flip}\n")


# N coin flips
# ========================
# random variable (coin; p=0.5, N=10)
N = 10
Y = Binomial(N, p)
print(Y)

flips = Y.sample()
print(f"Number of heads in {N} flips: {flips}")

# Monte Carlo simulation
# Mean number of heads in N flips
T = 5_000
mus = []
for t in range(T):
    mus.append(Y.sample())

print(f"Mean number of H ({T} trials)={sum(mus)/len(mus):.2f}")