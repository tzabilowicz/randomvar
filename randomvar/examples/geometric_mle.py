from distributions.discrete.Geometric import Geometric
from statistics.mle import mle_geometric

N = 1000
p_actual = 0.45

# Generate N geometric samples
X = Geometric(p_actual)
samples = [X.sample() for _ in range(N)]

# Estimate p
p_hat = mle_geometric(samples)

print(f"p actual = {p_actual} | p MLE = {p_hat}")