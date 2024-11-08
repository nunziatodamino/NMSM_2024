import numpy as np
import scipy.stats as stats
from typing import Callable

def inversion(cdf_inverse: Callable[[float], float], number_sampled_points: int) -> np.ndarray:
    random_variables = np.zeros(number_sampled_points)
    for i in range(number_sampled_points):
        random_variables[i] = cdf_inverse(np.random.random())
    return random_variables    

def rejection_from_normal(target_distribution, c, mean, sigma, number_sampled_points):
    counter = 0
    random_variables = np.zeros(number_sampled_points)
    while counter < number_sampled_points:
        candidate_random_variable = np.random.normal(mean, sigma)
        uniform_random_num = np.random.random()
        acceptance_threshold = target_distribution(candidate_random_variable)/ (c * stats.norm.pdf(candidate_random_variable, mean, sigma))
        if uniform_random_num <= acceptance_threshold:
            random_variables[counter] = candidate_random_variable
            counter+=1
    return random_variables      