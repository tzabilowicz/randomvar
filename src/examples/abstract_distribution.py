""" Example of an abstract distribution and sampling. """

from distributions.discrete.Binomial import Binomial
from distributions.discrete.Uniform import Uniform

N = 25_000 # Number of samples

X = Binomial(15, 0.6)
Y = Uniform(11, 25)

Z_1 = X + Y
Z_2 = X - Y
Z_3 = X * Y
Z_4 = X / Y

# Print the distribution's expectation and variance
print(f"E[Z_1] | Var[Z_1]: {Z_1.exp():.2f} | {Z_1.var():.2f}")
print(f"E[Z_2] | Var[Z_2]: {Z_2.exp():.2f} | {Z_2.var():.2f}")
print(f"E[Z_3] | Var[Z_3]: {Z_3.exp():.2f} | {Z_3.var():.2f}")
print(f"E[Z_4] | Var[Z_4]: {Z_4.exp()} | {Z_4.var()}")

z_1 = [Z_1.sample() for _ in range(N)]
z_2 = [Z_2.sample() for _ in range(N)]
z_3 = [Z_3.sample() for _ in range(N)]
z_4 = [Z_4.sample() for _ in range(N)]

# Plot the samples
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 18))
axs[0].hist(z_1, bins=25, label="Z1", color="blue", alpha=0.6)
axs[1].hist(z_2, bins=25, label="Z2", color="red", alpha=0.6)
axs[2].hist(z_3, bins=25, label="Z3", color="green", alpha=0.6)
axs[3].hist(z_4, bins=25, label="Z4", color="orange", alpha=0.6)
plt.show()