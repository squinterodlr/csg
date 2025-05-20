import numpy as np


class Box:
    """A space representing a box in several dimensions"""

    def __init__(self, dim, lower_bounds, upper_bounds):
        self.dim = dim
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def sample(self, num_samples=1, rng: np.random.Generator | None = None):

        if rng is None:
            return np.random.uniform(
                low=self.lower_bounds, high=self.upper_bounds, size=num_samples
            )
        else:
            return rng.uniform(
                low=self.lower_bounds, high=self.upper_bounds, size=num_samples
            )
