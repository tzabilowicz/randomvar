import unittest
from distributions.discrete.Binomial import Binomial

class TestDiscreteBinomial(unittest.TestCase):
    def testBinomialRepresentation(self):
        X = Binomial(5, 0.5)

        expected_repr = "D~Binomial(5,0.5)"
        actual_repr = str(X)

        self.assertEqual(actual_repr, expected_repr)

    def testInvalidBinomialProbability(self):
        # Test positive probability
        try:
            X = Binomial(1, 1.5) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

        # Test negative probability
        try:
            X = Binomial(1, -0.5) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

    def testInvalidBinomialSuccesses(self):
        # Test invalid number of successes
        try:
            X = Binomial(-1, 1.5) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

    def testSampleGeneration(self):
        N = 10_000 # Number of samples

        X = Binomial(N, 0.5)
        
        actual_s = X.sample()

        self.assertLessEqual(actual_s, N)

    def testProbabilityMassFunction(self):
        X = Binomial(5, 0.5)

        self.assertEqual(X.pmf(-1), 0)
        self.assertEqual(X.pmf(0), 0.03125)
        self.assertEqual(X.pmf(1), 0.15625)
        self.assertEqual(X.pmf(2), 0.3125)
        self.assertEqual(X.pmf(3), 0.3125)
        self.assertEqual(X.pmf(4), 0.15625)
        self.assertEqual(X.pmf(5), 0.03125)
        self.assertEqual(X.pmf(6), 0)

    def testCumulativeDistributionFunction(self):
        X = Binomial(5, 0.5)

        self.assertEqual(X.cdf(-1), 0)
        self.assertEqual(X.cdf(0), 0.03125)
        self.assertEqual(X.cdf(1), 0.1875)
        self.assertEqual(X.cdf(2), 0.5)
        self.assertEqual(X.cdf(3), 0.8125)
        self.assertEqual(X.cdf(4), 0.96875)
        self.assertEqual(X.cdf(5), 1)
        self.assertEqual(X.cdf(6), 1)