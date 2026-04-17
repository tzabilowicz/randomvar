import unittest
from distributions.discrete.Geometric import Geometric

class TestDiscreteGeometric(unittest.TestCase):
    def testGeometricRepresentation(self):
        X = Geometric(0.5)
        
        expected_repr = "D~Geometric(0.5)"
        actual_repr = str(X)
        
        self.assertEqual(actual_repr, expected_repr)
        
    def testInvalidGeometricProbability(self):
        # Test positive probability
        try:
            X = Geometric(1.5) # Should throw an error
            self.assertTrue(False)
        except ValueError as e:
            self.assertTrue(True)
        
        # Test negative probability
        try:
            X = Geometric(-0.5) # Should throw an error
            self.assertTrue(False)
        except ValueError as e:
            self.assertTrue(True)
    
    def testSampleGeneration(self):
        p = 0.5
        X = Geometric(p)
        
        actual_s = X.sample()
        
        self.assertGreaterEqual(actual_s, 0)
    
    def testProbabilityMassFunction(self):
        X = Geometric(0.5)
        
        self.assertEqual(X.pmf(-1), 0)
        self.assertEqual(X.pmf(0), 0)
        self.assertEqual(X.pmf(1), 0.5)
        self.assertEqual(X.pmf(2), 0.25)
    
    def testCumulativeDistributionFunction(self):
        X = Geometric(0.5)
        
        self.assertEqual(X.cdf(-1), 0)
        self.assertEqual(X.cdf(0), 0)
        self.assertEqual(X.cdf(1), 0.5)
        self.assertEqual(X.cdf(2), 0.75)