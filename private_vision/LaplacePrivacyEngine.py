# MIT License
#
# Copyright (C) IBM Corporation 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
The classic Laplace mechanism in differential privacy.
"""
from numbers import Real
import numpy as np
from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring
import types
class Laplace(DPMechanism):
    r"""
    The classical Laplace mechanism in differential privacy.

    First proposed by Dwork, McSherry, Nissim and Smith [DMNS16]_, with support for (relaxed)
    :math:`(\epsilon,\delta)`-differential privacy [HLM15]_.

    Samples from the Laplace distribution are generated using 4 uniform variates, as detailed in [HB21]_, to prevent
    against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism. Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism. Must be in [0, 1]. Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism. Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism. To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [DMNS16] Dwork, Cynthia, Frank McSherry, Kobbi Nissim, and Adam Smith. "Calibrating noise to sensitivity in
        private data analysis." Journal of Privacy and Confidentiality 7, no. 3 (2016): 17-51.

    .. [HLM15] Holohan, Naoise, Douglas J. Leith, and Oliver Mason. "Differential privacy in metric spaces: Numerical,
        categorical and functional data under the one roof." Information Sciences 305 (2015): 256-268.

    .. [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy." arXiv preprint
        arXiv:2107.10138 (2021).

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = None

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")
        return float(sensitivity)

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)
        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")
        return True

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the bias of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The bias of the mechanism at `value`.

        """
        return 0.0

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`.

        Parameters
        ----------
        value : float
            The value at which the variance of the mechanism is sought.

        Returns
        -------
        variance : float
            The variance of the mechanism at `value`.

        """
        self._check_all(0)
        return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(np.pi * unif4)

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : float
            The value to be randomised.

        Returns
        -------
        float
            The randomised value.

        """
        self._check_all(value)
        scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
        standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
                                                 self._rng.random())
        return value - scale * standard_laplace
    # def attach(self, optimizer):
    #   autograd_grad_sample.add_hooks(model=self.module, loss_reduction="sum")

    #   # Override zero grad.
    #   def dp_zero_grad(_self, *args, **kwargs):
    #       _self.privacy_engine.zero_grad()

    #   # Override step.
    #   def dp_step(_self, **kwargs):
    #       closure = kwargs.pop("closure", None)

    #       _self.privacy_engine.step(**kwargs)
    #       _self.original_step(closure=closure)
    #       _self.privacy_engine.unlock()  # Only enable creating new grads once parameters are updated.
    #       _self.privacy_engine.steps += 1

    #   def virtual_step(_self, **kwargs):
    #       _self.privacy_engine.virtual_step(**kwargs)

    #   def get_privacy_spent(_self, **kwargs):
    #       _self.privacy_engine.get_privacy_spent(**kwargs)

    #   def get_training_stats(_self, **kwargs):
    #       _self.privacy_engine.get_training_stats(**kwargs)

    #   optimizer.privacy_engine = self

    #   optimizer.original_step = optimizer.step
    #   optimizer.step = types.MethodType(dp_step, optimizer)

    #   optimizer.original_zero_grad = optimizer.zero_grad
    #   optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)

    #   optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

    #   # Make getting info easier.
    #   optimizer.get_privacy_spent = types.MethodType(get_privacy_spent, optimizer)
    #   optimizer.get_training_stats = types.MethodType(get_training_stats, optimizer)

    #   self.module.privacy_engine = self

    #   # Just to be safe, we also override `zero_grad` for module.
    #   self.module.original_zero_grad = self.module.zero_grad
    #   self.module.zero_grad = types.MethodType(dp_zero_grad, self.module)

    #   # For easy detaching.
    #   self.optimizer = optimizer

