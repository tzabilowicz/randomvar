import unittest
from Continuous.Uniform import Uniform

class TestContinuousUniform(unittest.TestCase):
    def testUniformRepresentation(self):
        X = Uniform(1.0, 10.0)

        expected_repr = "D~Uniform(1.0,10.0)"
        actual_repr = str(X)

        self.assertEqual(actual_repr, expected_repr)

    def testInvalidUniformRange(self):
        # Invalid positive range
        try:
            # Invalid range [b, a]
            X = Uniform(10.0, 1.0) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

        # Invalid negative range
        try:
            # Invalid range [-b, -a]
            X = Uniform(-1.0, -10.0) # Should throw a ValueError
            self.assertEqual(True, False)

        except ValueError as e:
            self.assertEqual(True, True)

    def testSampleGeneration(self):
        N = 1_000 # Number of samples
        a = 1.0
        b = 10.0
        X = Uniform(a, b)

        samples = [X.sample() for _ in range(N)]

        for sample in samples:
            # Validate samples are in range [a, b]
            self.assertGreaterEqual(sample, a)
            self.assertLessEqual(sample, b)

    def testProbabilityDensityFunction(self):
        a = 1.0
        b = 10.0
        X = Uniform(a, b)

        # Test interval [l_int, r_int]
        l_int = a - 5
        r_int = b + 5

        # Generate expected pdf values
        expected_pdfs = []
        for i in range(int(l_int), int(r_int)):
            if i < a or i > b:
                expected_pdfs.append(0)
            else:
                expected_pdfs.append(1 / (b - a))

        # Generate actual pdf values
        actual_pdfs = [X.pdf(i) for i in range(int(l_int), int(r_int))]

        self.assertEqual(actual_pdfs, expected_pdfs)

    def testCumulativeDistributionFunction(self):
        X = Uniform(1, 5)

        # Generate expected cdf values
        expected_cdfs = [0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0]

        # Generate actual cdf values [-1, 6]
        actual_cdfs = [round(X.cdf(i), 1) for i in range(-1, 6+1)]

        self.assertEqual(actual_cdfs, expected_cdfs)