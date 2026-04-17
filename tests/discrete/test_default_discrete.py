import unittest
from distributions.discrete.DiscreteDistribution import DiscreteDistribution

# Convolution testing
#from Discrete.Bernoulli import Bernoulli
#from Discrete.Binomial import Binomial
#from Discrete.Geometric import Geometric
from distributions.discrete.Uniform import Uniform


class TestDefaultDiscrete(unittest.TestCase):
    def testDiscreteRepresentation(self):
        X = Uniform(1, 5)
        Y = Uniform(1, 5)

        D = DiscreteDistribution(X, Y)

        expected_repr = "D~Discrete(D~Uniform(1,5),D~Uniform(1,5))"
        actual_repr = str(D)

        self.assertEqual(actual_repr, expected_repr)