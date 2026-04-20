from statistics.bootstrap import (
    bootstrap,
    bootstrap_two_sample,
)
from statistics.statistics import (
    mean,
    mean_ratio
)

test_data1 = [1, 2, 4, 8, 12, 11, 9, 23, 10, 3, 5, 6, 20]
test_data2 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

T_bs, mu, se, ci = bootstrap(
    test_data1,
    mean,
    N = 5_000
)
print(f"Bootstrap Results: mu={mu} se={se} ci={ci}")

T_bs, mu, se, ci = bootstrap_two_sample(
    test_data1,
    test_data2,
    mean_ratio,
    N = 5_000
)
print(f"Bootstrap Results: mu={mu} se={se} ci={ci}")