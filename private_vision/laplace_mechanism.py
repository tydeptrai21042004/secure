# laplace_mechanism.py

import numpy as np
from numbers import Real

class LaplaceMechanism:
    def __init__(self, *, epsilon, delta=0.0, sensitivity, random_state=None):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._rng = np.random.default_rng(random_state)

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")
        return float(sensitivity)

    def _check_all(self, value):
        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")
        return True

    def bias(self, value):
        return 0.0

    def variance(self, value):
        self._check_all(0)
        return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(np.pi * unif4)

    def randomise(self, value):
        self._check_all(value)
        scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
        standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(), self._rng.random())
        return value - scale * standard_laplace
