import unittest
from Discrete import Uniform

class TestDiscreteUniform(unittest.TestCase):
    def testSampleGeneration(self):
        N = 1_000 # Number of samples
        a = 1
        b = 10
        X = Uniform.Uniform(a, b)

        samples = [X.sample() for _ in range(N)]

        for sample in samples:
            # Validate samples are in range [a, b]
            self.assertGreaterEqual(sample, a)
            self.assertLessEqual(sample, b)

    def testProbabilityMassFunction(self):
        a = 1
        b = 10
        X = Uniform.Uniform(a, b)

        # Test interval [l_int, r_int]
        l_int = a - 5
        r_int = b + 5

        # Generate expected pmfs
        expected_pmfs = []
        for i in range(l_int, r_int):
            if i < a or i > b:
                expected_pmfs.append(0)
            else:
                expected_pmfs.append(1 / (b - a + 1))

        # Generate actual pmfs
        actual_pmfs = [X.pmf(i) for i in range(l_int, r_int)]

        self.assertEqual(actual_pmfs, expected_pmfs)

    def testCumulativeDistributionFunction(self):
        pass