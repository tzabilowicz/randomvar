import unittest
from Discrete.Bernoulli import Bernoulli

class TestDiscreteBernoulli(unittest.TestCase):
    def testBernoulliRepresentation(self):
        X = Bernoulli(0.5)

        expected_repr = "D~Bernoulli(0.5)"
        actual_repr = str(X)

        self.assertEqual(actual_repr, expected_repr)

    def testInvalidBernoulliProbability(self):
        # Test positive probability
        try:
            X = Bernoulli(1.5) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

        # Test negative probability
        try:
            X = Bernoulli(-0.5) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

    def testSampleGeneration(self):
        N = 10_000 # Number of samples
        expected_p = 0.5
        X = Bernoulli(expected_p)

        samples = [X.sample() for _ in range(N)]

        actual_p = sum(samples) / len(samples)

        self.assertAlmostEqual(actual_p, expected_p, places=2)

    def testProbabilityMassFunction(self):
        p = 0.6
        X = Bernoulli(p)

        self.assertEqual(X.pmf(-1), 0)
        self.assertEqual(X.pmf(0),  1-p)
        self.assertEqual(X.pmf(1),  p)
        self.assertEqual(X.pmf(2),  0)

    def testCumulativeDistributionFunction(self):
        p = 0.6
        X = Bernoulli(p)

        self.assertEqual(X.cdf(-1), 0)
        self.assertEqual(X.cdf(0),  1-p)
        self.assertEqual(X.cdf(1),  1)
        self.assertEqual(X.cdf(2),  1)