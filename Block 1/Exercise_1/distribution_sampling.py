import numpy as np
from typing import Callable

def inversion_sampling(cdf_inverse: Callable[[float], float], number_sampled_points: int) -> np.ndarray:
    random_variables = np.zeros(number_sampled_points)
    for i in range(number_sampled_points):
        random_variables[i] = cdf_inverse(np.random.random())
    return random_variables    
