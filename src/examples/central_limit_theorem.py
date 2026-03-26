""" Central Limit Theorem

This example demontrates the central limit theorem with
a discrete and continuous random variable.
"""

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

from Discrete.Geometric import Geometric
from Continuous.Uniform import Uniform

# Example 1 (Discrete)
X = Geometric(p=0.6)

X_bar = []

S = 100   # Number of samples
N = 5_000 # Number of trials
for _ in range(N):
    samples = [X.sample() for _ in range(S)]
    X_bar.append(sum(samples) / len(samples))

# Histogram of sample means
plt.hist(X_bar, bins=20, color="r", alpha=0.65)
plt.title("Histogram of Sample Means (Discrete)")
plt.show()


# Example 2 (Continuous)
Y = Uniform(5.0, 10.0)

X_bar = []

for _ in range(N):
    samples = [Y.sample() for _ in range(S)]
    X_bar.append(sum(samples) / len(samples))

# Histogram of sample means
plt.hist(X_bar, bins=20, color="b", alpha=0.65)
plt.title("Histogram of Sample Means (Continuous)")
plt.show()