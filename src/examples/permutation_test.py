from permutation.permutation import permutation_test
from permutation.test_statistics import mean_difference

observed_1 = [30, 28, 25, 24, 22, 20, 27, 26, 23, 21]
observed_2 = [18, 19, 21, 20, 17, 16, 22, 23, 19, 18]

t_obs, p_val = permutation_test(
    observed_1,
    observed_2,
    mean_difference,
    N = 1000,
    two_sided = True
)

print(f"Hypothesis Test Results: t_obs={t_obs:.5f} p_val={p_val:.5f}")