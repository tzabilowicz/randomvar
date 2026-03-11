import matplotlib.pyplot as plt

from Discrete.Geometric import Geometric

X = Geometric(0.25)

# Plot the PMF for values 1 to 20
ys = [X.pmf(i) for i in range(1, 21)]
xs = [i for i in range(1, 21)]

plt.bar(xs, ys)
plt.xticks(xs)
plt.title("PMF of X~Geom(0.25)")
plt.xlabel("Value of X")
plt.ylabel("P(X=x)")
plt.show()