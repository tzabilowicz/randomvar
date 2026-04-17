import unittest
from distributions.continuous.Exponential import Exponential

class TestContinuousExponential(unittest.TestCase):
    def testExponentialRepresentation(self):
        X = Exponential(lambd=1/10)
        
        expected_repr = "D~Exponential(0.1)"
        actual_repr = str(X)
        
        self.assertEqual(expected_repr, actual_repr)
    
    def testExponentialLambda(self):
        # Negative lambda (invalid)
        try:
            X = Exponential(-1/10)
            self.assertTrue(False)
        
        except ValueError as e:
            self.assertTrue(True)
        
        # Zero lamda (invalid)
        try:
            X = Exponential(0)
            self.assertTrue(False)
        
        except ValueError as e:
            self.assertTrue(True)
        
        # Positive lambda (valid)
        try:
            X = Exponential(1/10)
            self.assertTrue(True)
        
        except ValueError as e:
            self.assertTrue(False)
    
    def testSampleGeneration(self):
        X = Exponential(1/10)
        
        N = 5_000
        samples = [X.sample() for _ in range(N)]
        
        # Validate all samples are positive
        self.assertTrue(all(s > 0 for s in samples))
    
    def testProbabilityDensityFunction(self):
        pass
    
    def testCumulativeDistributionFunction(self):
        pass