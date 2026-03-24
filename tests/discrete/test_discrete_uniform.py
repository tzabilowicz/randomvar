import unittest
from Discrete.Uniform import Uniform

class TestDiscreteUniform(unittest.TestCase):
    def testUniformRepresentation(self):
        X = Uniform(1, 10)

        expected_repr = "D~Uniform(1,10)"
        actual_repr = str(X)

        self.assertEqual(actual_repr, expected_repr)

    def testInvalidUniformRange(self):
        # Invalid positive ranve
        try:
            # Invalid range [b, a]
            X = Uniform(10, 1) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

        # Invalid negative range
        try:
            # Invalid range [-b, -a]
            X = Uniform(-1, -10) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

    def testSampleGeneration(self):
        N = 1_000 # Number of samples
        a = 1
        b = 10
        X = Uniform(a, b)

        samples = [X.sample() for _ in range(N)]

        for sample in samples:
            # Validate samples are in range [a, b]
            self.assertGreaterEqual(sample, a)
            self.assertLessEqual(sample, b)

    def testProbabilityMassFunction(self):
        a = 1
        b = 10
        X = Uniform(a, b)

        # Test interval [l_int, r_int]
        l_int = a - 5
        r_int = b + 5

        # Generate expected pmf values
        expected_pmfs = []
        for i in range(l_int, r_int):
            if i < a or i > b:
                expected_pmfs.append(0)
            else:
                expected_pmfs.append(1 / (b - a + 1))

        # Generate actual pmf values
        actual_pmfs = [X.pmf(i) for i in range(l_int, r_int)]

        self.assertEqual(actual_pmfs, expected_pmfs)

    def testCumulativeDistributionFunction(self):
        X = Uniform(1, 5)

        # Generate expected cdf values
        expected_cdfs = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]

        # Generate actual cdf values [0, 6]
        actual_cdfs = [round(X.cdf(i), 1) for i in range(0, 6+1)]

        self.assertEqual(actual_cdfs, expected_cdfs)