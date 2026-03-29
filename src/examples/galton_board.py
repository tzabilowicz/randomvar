import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

from Discrete.Bernoulli import Bernoulli

ROWS  = 25
BALLS = 10_000

pos = []
for b in range(BALLS):
    X = Bernoulli(0.5)
    result = [1 if X.sample() == 1 else -1 for _ in range(ROWS)]
    pos.append(sum(result))

bins = np.arange(-26, 28, 2)
plt.hist(pos, bins=bins, edgecolor='white')
plt.title("Galton Board Simulation (Centered Bins)")
plt.show()