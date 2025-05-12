from abc import abstractmethod
from csg.space import Box
from datetime import datetime
import numpy as np


class ContinuousStochasticGradient:

    def __init__(
        self,
        objective_function,
        design_space: Box,
        parameter_space: Box,
        seed: int = None,
    ):
        self.objective_function = objective_function
        self.design_space = design_space
        self.parameter_space = parameter_space
        self.reset(seed)

    def reset(self, seed: int = None):
        if seed is None:
            self.seed = int(1e6 * datetime.now().timestamp())
        self.rng = np.random.default_rng(self.seed)
        self.design_samples = [self.design_space.sample()]
        self.parameter_samples = []
        self.gradients = None

    def step(self, step_size: float = None):
        
        x_n =self.parameter_space.sample(rng=self.rng)
        self.parameter_samples.append()
        
        u_n = self.design_samples[-1]

        grad_u = self.objective_function.grad_u(u_n, x_n)
        self.gradients = np.vstack([self.gradients, grad_u]) # shape (len(design_samples), design_space.dim)
        weights = self._calculate_weights() # shape (len(design_samples), )
        step_size = self._calculate_step_size()

        this_gradient = np.tensordot(weights, self.gradients, axes=(0, 0))

        u_n_plus_1 = u_n - step_size * this_gradient

        self.design_samples.append(u_n_plus_1)
            
    def _calculate_step_size(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _calculate_weights(self) -> np.ndarray:
        pass
